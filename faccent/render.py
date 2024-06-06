# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from faccent import objectives, transform, param, utils
from faccent.utils import show, plot,plot_alpha, default_model_input_size, default_model_input_range, default_img_size


def render_vis(
    model,
    objective,
    parameterizer=None,
    transforms=None,
    optimizer=None,
    out_thresholds=range(0,100,2),
    inline_thresholds=range(0,100,10),
    trans_p = 20,
    nb_transforms = 16,
    init_img = None,
    verbose=False,
    preprocess=True,
    progress=True,
    img_tr_obj = None,
    reg_for_img_tr = False,
    model_input_size = None,
    model_input_range = None,
    img_size = None,
    hook_layers = None,
    accent_reg_layer = None,
    accent_reg_alpha = None,
    activation_cutoff = None,
    simple_transforms = None,
    pgd_alpha = None
):

    device = next(model.parameters()).device

    if len(inline_thresholds) >= 1:
        inline_thresholds = list(inline_thresholds)
        inline_thresholds.append(max(out_thresholds))

    if model_input_size is None:
        try:
            model_input_size = model.model_input_size
        except:
            print('warning: arg "model_input_size" not set using default %s'%str(default_model_input_size))
            model_input_size = default_model_input_size
    if model_input_range is None:
        try:
            model_input_range = model.model_input_range
        except:
            print('warning: arg "model_input_range" not set using default %s'%str(default_model_input_range))
            model_input_range = default_model_input_range
    if img_size is None:
        img_size = default_img_size

    if parameterizer is None:
        parameterizer = param.fourier(device=device, img_size = img_size,init_img=init_img)
    if init_img is not None:
        print('initializing parameterization with %s'%str(init_img))
        #parameterizer.img_to_params(init_img)

    #regularization
    if accent_reg_layer is not None:
        if accent_reg_alpha is None:
            print('reg_layer arg set, but not reg_alpha, use reg_alpha=1')
            accent_reg_alpha = 1.
        parameterizer.forward_init_img = True  #when using regularization we must pass the initial image through model on all steps
    if accent_reg_alpha is not None and accent_reg_layer is None:
        print('arg reg_alpha set, but not reg_layer, you must set reg_layer! Applying no regularization.')
    
    params, img_f = parameterizer()
    for p in params: p.requires_grad_(True)

    for p in model.parameters():
        p.requires_grad = False

    if optimizer is None:
        if pgd_alpha is None:
            optimizer = lambda params: torch.optim.Adam(params, lr=.05)
        else:
            optimizer = lambda params: torch.optim.SGD(params, lr=.05)

    optimizer_f = optimizer(params)

    if transforms is None:
        try:
            print('using parameterizer.standard_transforms')
            transforms = parameterizer.standard_transforms.copy()
        except:
            transforms = transform.standard_box_transforms.copy()
    transforms = transforms.copy()
    if preprocess:
        transforms.append(transform.resize(model_input_size))
        transforms.append(transform.range_normalize(model_input_range))
    transform_f = transform.compose(transforms,nb_transforms=nb_transforms)
    if simple_transforms is None:
        simple_transform_f = transform.compose(
                                    [transform.resize(model_input_size),
                                     transform.range_normalize(model_input_range)]
                                    ,nb_transforms=nb_transforms) #EDIT nb_transforms should be 1 here, but doesnt work with hook
    else:
        simple_transform_f = transform.compose(simple_transforms,nb_transforms=nb_transforms)


    if pgd_alpha is not None:
        pgd_init_img = img_f().detach().clone().requires_grad_(False)[:1]
        #pgd_init_img = parameterizer.params_to_img().detach().clone().requires_grad_(False)

    with hook_model(model, img_f, transform_f,nb_transforms=nb_transforms,layers=hook_layers) as hooker:
        hook = hooker.hook_f

        img = img_f()
        #img = parameterizer.params_to_img()

        imgs = []
        img_trs = []
        img_tr  = torch.zeros(img.shape).float().cpu() #this will be 'the transparency' mask, determined by pixel gradients
        losses = []
        img_tr_losses = []
        #param_grads = []
        
        objective = objectives.as_objective(objective)
    
        #set objectives/regularization
        #balanced-alpha regularization
        if accent_reg_layer is not None:
            accent_reg_obj = objectives.l2_compare(accent_reg_layer)

        if verbose or accent_reg_layer is not None:
            model(transform_f(img))
            print("Initial loss: {:.3f}".format(objective(hook)))
        
        if accent_reg_layer is not None:
            #compute balance by gradient ratio
            #reg grads
            reg_loss = accent_reg_obj(hook)
            reg_loss.backward(retain_graph=True)
            reg_grad = 0
            for p in params: 
                reg_grad += float(torch.sum(torch.abs(p.grad.detach().clone())).cpu())

            #img.grad.zero_()
            optimizer_f.zero_grad()
            try:   #EDIT this was failing on colab because params grads didnt exist . . .
                for p in params: p.grad.zero_()
            except:
                pass
                
            #objective grads
            obj_loss = objective(hook)
            obj_loss.backward(retain_graph=True)
            obj_grad = 0
            for p in params: 
                obj_grad += float(torch.sum(torch.abs(p.grad.detach().clone()).cpu()))

            #img.grad.zero_()
            optimizer_f.zero_grad()
            try:  #EDIT this was failing on colab because params grads didnt exist . . .
                for p in params: p.grad.zero_()
            except:
                pass
                
            print('obj/reg ratio: %s'%str(obj_grad/reg_grad))
            print('setting reg balance parameter to this ratio')
            accent_reg_balance = obj_grad/reg_grad
                
        #set image transparency objective
        if img_tr_obj is not None: 
            img_tr_obj = objectives.as_objective(img_tr_obj)
        elif (accent_reg_layer is not None) and not reg_for_img_tr:
            img_tr_obj = objective
            
        #set full objective with regularization
        if accent_reg_layer is not None:
            objective_full = objective - accent_reg_alpha*accent_reg_balance*accent_reg_obj
        else:
            objective_full = objective
        
        
        
        #main optimization loop
        for i in tqdm(range(0, max(out_thresholds) + 1), disable=(not progress)):
            optimizer_f.zero_grad()
            img = img_f()
            #img = parameterizer.params_to_img()
            img.retain_grad()
            #img.grad.zero_()

            try:
                model(transform_f(img))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )

            if img_tr_obj is not None: #compute seperate loss for pixel gradients
                #try:
                img_tr_loss = img_tr_obj(hook)
                img_tr_loss.backward(retain_graph=True)
#                 except RuntimeError as e:
#                     if "modified by an inplace operation" in str(e): pass 
#                     else: raise e

                img_tr = img_tr.detach().clone() + torch.abs(img.grad.detach().clone().cpu())
                img.grad.zero_()
                optimizer_f.zero_grad()
                try:   #EDIT this was failing on colab because params grads didnt exist . . .
                    for p in params: p.grad.zero_()
                except:
                    pass
            
            loss = objective_full(hook)  #compute full objective function loss
            loss.backward()
            #param_grads.append(params[0].grad.clone().cpu())

            if img_tr_obj is None: 
                img_tr = img_tr.detach().clone() + torch.abs(img.grad.detach().clone().cpu())

            if i in out_thresholds:
                imgs.append(img.detach().clone().cpu().requires_grad_(False))
                img_trs.append(img_tr)
                losses.append(-float(loss.detach().clone().cpu()))
                if img_tr_obj is not None:
                    img_tr_losses.append(-float(img_tr_loss.detach().clone().cpu()))
            if i in inline_thresholds:
                if (trans_p is None) or (trans_p in (0,100)):
                    plot(img)
                else:
                    plot_alpha(img,img_tr, p = trans_p)

            if activation_cutoff is not None:
                model(simple_transform_f(img)) #need to compute how the sythesized image activates the feature, without noise
                exit_early = False
                #if img_tr_obj is not None:
                #    if -img_tr_loss > activation_cutoff:
                #        exit_early = True
                #elif -objective(hook) > activation_cutoff:
                if -objective(hook) > activation_cutoff:
                    exit_early = True
                if exit_early:
                    if i not in out_thresholds:
                        imgs.append(img.detach().clone().cpu().requires_grad_(False))
                        img_trs.append(img_tr)
                        losses.append(-float(loss.detach().clone().cpu()))
                        if img_tr_obj is not None:
                            img_tr_losses.append(-float(img_tr_loss.detach().clone().cpu()))

                    if (trans_p is None) or (trans_p in (0,100)):
                        plot(img)
                    else:
                        plot_alpha(img,img_tr, p = trans_p)

                    return imgs, img_trs, losses, img_tr_losses

            optimizer_f.step()

            if pgd_alpha is not None:
                #gen the new image
                img = parameterizer.params_to_img().detach().clone().requires_grad_(False)[:1]
                #project it onto ball around init image if too far away
                pgd_d = img - pgd_init_img
                pgd_d_norm = torch.norm(pgd_d)
                #print(pgd_d_norm)
                if pgd_d_norm > pgd_alpha:
                    print('projecting')
                    img = pgd_init_img + (pgd_alpha/pgd_d_norm)*pgd_d
                    #update the parameterization with the projected image
                    parameterizer.img_to_params(img) 
                    params, img_f = parameterizer()
                    for p in params: p.requires_grad_(True)
                    optimizer_f = optimizer(params)

    return imgs, img_trs, losses, img_tr_losses #, param_grads




def rebatch_transforms(t, nb_transforms=1):
    #adds extra dimension for transform batches
    original_shape = list(t.shape)
    new_shape = [nb_transforms, original_shape[0] // nb_transforms] + original_shape[1:]
    reshaped_t = t.view(*new_shape)
    #n, c, h, w = t.shape
    #b = n // nb_transforms
    #reshaped_t = t.view(nb_transforms, b, c, h, w)
    return reshaped_t


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


class hook_model():
    def __init__(self, model, image_f=None,transform_f=None,nb_transforms=None,layers = None):
        self.image_f = image_f
        self.nb_transforms = nb_transforms
        self.layers = layers
        self.model = model
        self.transform_f = transform_f
        features = OrderedDict()
        # recursive hooking function
        def hook_layers(net, prefix=[], layers = self.layers):
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue

                    layer._forward_hooks.clear()
                    layer._forward_pre_hooks.clear()
                    layer._backward_hooks.clear()
                    if (layers is None) or  ("_".join(prefix + [name]) in layers):
                        features["_".join(prefix + [name])] = ModuleHook(layer)

                    hook_layers(layer, prefix=prefix + [name])

        hook_layers(model)
        self.features = features
    
    def hook_f(self,layer,nb_transforms = None):
        if layer == "input":
            out = self.transform_f(self.image_f())
        elif layer == "labels":
            out = list(self.features.values())[-1].features
        else:
            assert layer in self.features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = self.features[layer].features

        #assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        if nb_transforms is not None:
            return rebatch_transforms(out,nb_transforms=nb_transforms)
        elif self.nb_transforms:
            return rebatch_transforms(out,nb_transforms=self.nb_transforms)
        else:
            return out

    def __exit__(self, *args):
        for feature in self.features: 
            self.features[feature].close()

    def __enter__(self, *args):
        return self


def render_vis_zoomsmooth(
    model,
    objective,
    parameterizer=None,
    optimizer=None,
    out_thresholds=range(0,100,2),
    inline_thresholds=range(0,100,10),
    trans_p = 20,
    nb_transforms = 16,
    init_img = None,
    verbose=False,
    preprocess=True,
    progress=True,
    img_tr_obj = None,
    reg_for_img_tr = False,
    model_input_size = None,
    model_input_range = None,
    box_min_size = .05,
    box_max_size = .99,
    init_box_min_size = .7,
    noise_std=.02,
    img_size = None,
    hook_layers = None,
    accent_reg_layer = None,
    accent_reg_alpha = None,
    activation_cutoff = None,
    simple_transforms = None,
    pgd_alpha = None
):

    device = next(model.parameters()).device

    if len(inline_thresholds) >= 1:
        inline_thresholds = list(inline_thresholds)
        inline_thresholds.append(max(out_thresholds))

    if model_input_size is None:
        try:
            model_input_size = model.model_input_size
        except:
            print('warning: arg "model_input_size" not set using default %s'%str(default_model_input_size))
            model_input_size = default_model_input_size
    if model_input_range is None:
        try:
            model_input_range = model.model_input_range
        except:
            print('warning: arg "model_input_range" not set using default %s'%str(default_model_input_range))
            model_input_range = default_model_input_range
    if img_size is None:
        img_size = default_img_size

    if parameterizer is None:
        parameterizer = param.fourier(device=device, img_size = img_size,init_img=init_img)
    if init_img is not None:
        print('initializing parameterization with %s'%str(init_img))
        #parameterizer.img_to_params(init_img)

    #regularization
    if accent_reg_layer is not None:
        if accent_reg_alpha is None:
            print('reg_layer arg set, but not reg_alpha, use reg_alpha=1')
            accent_reg_alpha = 1.
        parameterizer.forward_init_img = True  #when using regularization we must pass the initial image through model on all steps
    if accent_reg_alpha is not None and accent_reg_layer is None:
        print('arg reg_alpha set, but not reg_layer, you must set reg_layer! Applying no regularization.')
    
    params, img_f = parameterizer()
    for p in params: p.requires_grad_(True)

    for p in model.parameters():
        p.requires_grad = False

    if optimizer is None:
        if pgd_alpha is None:
            optimizer = lambda params: torch.optim.Adam(params, lr=.05)
        else:
            optimizer = lambda params: torch.optim.SGD(params, lr=.05)

    optimizer_f = optimizer(params)



    if pgd_alpha is not None:
        pgd_init_img = img_f().detach().clone().requires_grad_(False)[:1]
        #pgd_init_img = parameterizer.params_to_img().detach().clone().requires_grad_(False)

    with hook_model(model, img_f ,nb_transforms=nb_transforms,layers=hook_layers) as hooker:
        hook = hooker.hook_f

        img = img_f()
        #img = parameterizer.params_to_img()

        imgs = []
        img_trs = []
        img_tr  = torch.zeros(img.shape).float().cpu() #this will be 'the transparency' mask, determined by pixel gradients
        losses = []
        img_tr_losses = []
        #param_grads = []
        
        objective = objectives.as_objective(objective)
    
        #set objectives/regularization
        #balanced-alpha regularization
        if accent_reg_layer is not None:
            accent_reg_obj = objectives.l2_compare(accent_reg_layer)

        if verbose or accent_reg_layer is not None:
            model(transform_f(img))
            print("Initial loss: {:.3f}".format(objective(hook)))
        
        if accent_reg_layer is not None:
            #compute balance by gradient ratio
            #reg grads
            reg_loss = accent_reg_obj(hook)
            reg_loss.backward(retain_graph=True)
            reg_grad = 0
            for p in params: 
                reg_grad += float(torch.sum(torch.abs(p.grad.detach().clone())).cpu())

            #img.grad.zero_()
            optimizer_f.zero_grad()
            try:   #EDIT this was failing on colab because params grads didnt exist . . .
                for p in params: p.grad.zero_()
            except:
                pass
                
            #objective grads
            obj_loss = objective(hook)
            obj_loss.backward(retain_graph=True)
            obj_grad = 0
            for p in params: 
                obj_grad += float(torch.sum(torch.abs(p.grad.detach().clone()).cpu()))

            #img.grad.zero_()
            optimizer_f.zero_grad()
            try:  #EDIT this was failing on colab because params grads didnt exist . . .
                for p in params: p.grad.zero_()
            except:
                pass
                
            print('obj/reg ratio: %s'%str(obj_grad/reg_grad))
            print('setting reg balance parameter to this ratio')
            accent_reg_balance = obj_grad/reg_grad
                
        #set image transparency objective
        if img_tr_obj is not None: 
            img_tr_obj = objectives.as_objective(img_tr_obj)
        elif (accent_reg_layer is not None) and not reg_for_img_tr:
            img_tr_obj = objective
            
        #set full objective with regularization
        if accent_reg_layer is not None:
            objective_full = objective - accent_reg_alpha*accent_reg_balance*accent_reg_obj
        else:
            objective_full = objective
        
        
        transform_step_size = (init_box_min_size - box_min_size)/max(out_thresholds)
        box_min_size = init_box_min_size
        
        #main optimization loop
        for i in tqdm(range(0, max(out_thresholds) + 1), disable=(not progress)):
            
            box_min_size -= transform_step_size
            transform_f = transform.fast_batch_crop(
                box_min_size=box_min_size,
                box_max_size=box_max_size,
                noise_std=0.02,
                img_size= model_input_size,
                img_range = model_input_range,
                nb_crops = nb_transforms)
   
            optimizer_f.zero_grad()
            img = img_f()
            #img = parameterizer.params_to_img()
            img.retain_grad()
            #img.grad.zero_()

            try:
                model(transform_f(img))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )

            if img_tr_obj is not None: #compute seperate loss for pixel gradients
                #try:
                img_tr_loss = img_tr_obj(hook)
                img_tr_loss.backward(retain_graph=True)
#                 except RuntimeError as e:
#                     if "modified by an inplace operation" in str(e): pass 
#                     else: raise e

                img_tr = img_tr.detach().clone() + torch.abs(img.grad.detach().clone().cpu())
                img.grad.zero_()
                optimizer_f.zero_grad()
                try:   #EDIT this was failing on colab because params grads didnt exist . . .
                    for p in params: p.grad.zero_()
                except:
                    pass
            
            loss = objective_full(hook)  #compute full objective function loss
            loss.backward()
            #param_grads.append(params[0].grad.clone().cpu())

            if img_tr_obj is None: 
                img_tr = img_tr.detach().clone() + torch.abs(img.grad.detach().clone().cpu())

            if i in out_thresholds:
                imgs.append(img.detach().clone().cpu().requires_grad_(False))
                img_trs.append(img_tr)
                losses.append(-float(loss.detach().clone().cpu()))
                if img_tr_obj is not None:
                    img_tr_losses.append(-float(img_tr_loss.detach().clone().cpu()))
            if i in inline_thresholds:
                if (trans_p is None) or (trans_p in (0,100)):
                    plot(img)
                else:
                    plot_alpha(img,img_tr, p = trans_p)

            if activation_cutoff is not None:
                model(simple_transform_f(img)) #need to compute how the sythesized image activates the feature, without noise
                exit_early = False
                #if img_tr_obj is not None:
                #    if -img_tr_loss > activation_cutoff:
                #        exit_early = True
                #elif -objective(hook) > activation_cutoff:
                if -objective(hook) > activation_cutoff:
                    exit_early = True
                if exit_early:
                    if i not in out_thresholds:
                        imgs.append(img.detach().clone().cpu().requires_grad_(False))
                        img_trs.append(img_tr)
                        losses.append(-float(loss.detach().clone().cpu()))
                        if img_tr_obj is not None:
                            img_tr_losses.append(-float(img_tr_loss.detach().clone().cpu()))

                    if (trans_p is None) or (trans_p in (0,100)):
                        plot(img)
                    else:
                        plot_alpha(img,img_tr, p = trans_p)

                    return imgs, img_trs, losses, img_tr_losses

            optimizer_f.step()

            if pgd_alpha is not None:
                #gen the new image
                img = parameterizer.params_to_img().detach().clone().requires_grad_(False)[:1]
                #project it onto ball around init image if too far away
                pgd_d = img - pgd_init_img
                pgd_d_norm = torch.norm(pgd_d)
                #print(pgd_d_norm)
                if pgd_d_norm > pgd_alpha:
                    print('projecting')
                    img = pgd_init_img + (pgd_alpha/pgd_d_norm)*pgd_d
                    #update the parameterization with the projected image
                    parameterizer.img_to_params(img) 
                    params, img_f = parameterizer()
                    for p in params: p.requires_grad_(True)
                    optimizer_f = optimizer(params)

    return imgs, img_trs, losses, img_tr_losses #, param_grads
