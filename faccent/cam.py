
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, Iterable, Callable
from collections import OrderedDict
import types
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,Resize,ToTensor
from faccent.utils import default_model_input_range, default_model_input_size, TargetReached, img_to_img_tensor, LargestCenterCrop, image_data,min_max
from faccent.transform import range_normalize
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from faccent.transform import range_normalize, resize
from faccent.utils import img_to_img_tensor
from scipy.ndimage import gaussian_filter
import torch  # Assuming you are using a PyTorch tensor
from matplotlib.patches import Rectangle

class feature_target_saver(nn.Module):
    '''
        takes a model, and adds a target feature to save the outputs of
        layer: str name of layer, use "dict([*self.model.named_modules()])"
        unit: int or list/array the length of the out features dim of the layer (specifying coefficients)

        with feature_target_saver(model, layer,unit) as target_saver:
            ... run images through model
    '''
    def __init__(self, model, layer, unit, kill_forward = True):
        super().__init__()
        self.model = model
        self.layer = layer
        self.unit = unit
        self.target_activations = None
        self.layer_name = layer
        self.layer = OrderedDict([(key.replace('.', '_'), value) for key, value in model.named_modules()])[layer]
        #self.hook = self.layer.register_forward_hook(self.get_target()) #execute on forward pass
        self.hook = None
        self.kill_forward = kill_forward

    def __enter__(self, *args): 
        if self.hook is not None:
            self.hook.remove()
        self.hook = self.layer.register_forward_hook(self.get_target())       
        return self

    def __exit__(self, *args): 
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def get_target(self) -> Callable:
        def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
            #output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
            if isinstance(self.unit,int):
                target_activations = output[:,self.unit]
            else:
                assert len(self.unit) == output.shape[1]
                self.unit = torch.tensor(self.unit).to(output.device).type(output.dtype)
                target_activations = torch.tensordot(output, self.unit, dims=([1],[0]))

            self.target_activations = target_activations


            if self.kill_forward:
                #print('feature target in %s reached.'%self.layer)
                raise TargetReached
        return fn

    def forward(self, x):
        try:
            _ = self.model(x)
        except TargetReached:
            pass
        return self.target_activations



class layer_saver(nn.Module):
    '''
        layer_saver class that allows you to retain outputs of any layer.
        This class uses PyTorch's "forward hooks", which let you insert a function
        that takes the input and output of a module as arguements.
        In this hook function you can insert tasks like storing the intermediate values,
        or as we'll do in the FeatureEditor class, actually modify the outputs.
        Adding these hooks can cause headaches if you don't "remove" them 
        after you are done with them. For this reason, the FeatureExtractor is 
        setup to be used as a context, which sets up the hooks when
        you enter the context, and removes them when you leave:
        with layer_saver(model, layer_name) as saver:
            features = saver(imgs)
        If there's an error in that context (or you cancel the operation),
        the __exit__ function of the feature extractor is executed,
        which we've setup to remove the hooks. This will save you 
        headaches during debugging/development.
    '''    
    def __init__(self, model, layers, retain=True, detach=True, clone=True):
        super().__init__()
        layers = [layers] if isinstance(layers, str) else layers
        self.model = model
        self.layers = layers
        self.detach = detach
        self.clone = clone
        self.device = next(model.parameters()).device 
        self.retain = retain
        self._features = {layer: torch.empty(0) for layer in layers}        
        self.hooks = {}
        
    def hook_layers(self):        
        self.remove_hooks()
        for layer_id in self.layers:
            layer = OrderedDict([(key.replace('.', '_'), value) for key, value in self.model.named_modules()])[layer_id]
            self.hooks[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
    def remove_hooks(self):
        for layer_id in self.layers:
            if self.retain == False:
                self._features[layer_id] = torch.empty(0)
            if layer_id in self.hooks:
                self.hooks[layer_id].remove()
                del self.hooks[layer_id]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self
    
    def __exit__(self, *args): 
        self.remove_hooks()
            
    def save_outputs_hook(self, layer_id):
        def detach(output):
            if isinstance(output, tuple): return tuple([o.detach() for o in output])
            elif isinstance(output, list): return [o.detach() for o in output]
            else: return output.detach()
        def clone(output):
            if isinstance(output, tuple): return tuple([o.clone() for o in output])
            elif isinstance(output, list): return [o.clone() for o in output]
            else: return output.clone()
        def to_device(output, device):
            if isinstance(output, tuple): return tuple([o.to(device) for o in output])
            elif isinstance(output, list): return [o.to(device) for o in output]
            else: return output.to(device)
        def fn(_, __, output):
            if self.detach: output = detach(output)
            if self.clone: output = clone(output)
            if self.device: output = to_device(output, self.device)
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features
    


def layer_activations_from_dataloader(layers,dataloader,model,batch_size=64,retain=True, detach=True, clone=True):
  '''
  dataloader: can be a pytorch dataloader or simply a path to an folder with images and no subfolders
  layers: should be a single layer name or list of layer names, for keys in dict "layers = OrderedDict([*model.named_modules()])"
  model: a pytorch nn model, set the device of model to determine devices for all variables in this function
  '''

  device = next(model.parameters()).device

  #generate dataloader if image_path passed
  if isinstance(dataloader,str):
    kwargs = {'num_workers': 4, 'pin_memory': True, 'sampler':None} if 'cuda' in device.type else {}
    dataloader = DataLoader(image_data(dataloader,
                                            class_folders=False,
                                            rgb=True),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            **kwargs)
  

  layer_activations = {}
  if isinstance(layers,str):
    layers = [layers]
  for i in layers:
    layer_activations[i] = []
  

  for i, data in enumerate(dataloader):
    #if i%int(len(dataloader)/4) == 0:
    #  print(str(i)+'/'+str(len(dataloader)))
    images = data[0].to(device)
    with layer_saver(model, layers, retain=retain, detach=detach, clone=clone) as extractor:
      batch_layer_activations = extractor(images) #all features for layer and all images in batch
      for i in layers:
        layer_activations[i].append(batch_layer_activations[i].detach().to('cpu'))

  for i in layers:     
    layer_activations[i] = torch.cat(layer_activations[i])

  return layer_activations
    

class actgrad_extractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str], concat=True, extract_grads = True):
        super().__init__()
        self.model = model
        self.layers = layers
        self.activations = {layer: None for layer in layers}
        self.extract_grads = extract_grads
        if extract_grads:
            self.gradients = {layer: None for layer in layers}
        self.concat = concat
        

    def __enter__(self, *args):
        #self.remove_all_hooks() 
        self.hooks = {'forward':{},
                      'backward':{}}   #saving hooks to variables lets us remove them later if we want
        for layer_id in self.layers:
            layer = OrderedDict([(key.replace('.', '_'), value) for key, value in self.model.named_modules()])[layer_id]
            self.hooks['forward'][layer_id] = layer.register_forward_hook(self.save_activations(layer_id)) #execute on forward pass
            if self.extract_grads:
                self.hooks['backward'][layer_id] = layer.register_full_backward_hook(self.save_gradients(layer_id))    #execute on backwards pass      
        return self

    def __exit__(self, *args): 
        self.remove_all_hooks()


    def save_activations(self, layer_id: str) -> Callable:
        def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
            #output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
            if (self.activations[layer_id] is None) or (not self.concat):
                self.activations[layer_id] = output.detach().cpu()
            else:
                self.activations[layer_id] = torch.cat((self.activations[layer_id],output.detach().cpu()), dim=0)           
        return fn

    def save_gradients(self, layer_id: str) -> Callable:
        def fn(module, grad_input, grad_output):
            if (self.gradients[layer_id] is None) or (not self.concat):
                self.gradients[layer_id] = grad_output[0].detach().cpu()
            else:
                self.gradients[layer_id] = torch.cat((self.gradients[layer_id],grad_output[0].detach().cpu()), dim=0)
                
        return fn

    def remove_all_hooks(self):
        for layer_id in self.layers:
            self.hooks['forward'][layer_id].remove()
            if self.extract_grads: self.hooks['backward'][layer_id].remove()


def posneg_cam_maps_from_dataloader(data,
                    model,
                    target_layer,
                    unit,
                    cam_layers,
                    position = None,
                    batch_size = 64, 
                    num_workers = 2, 
                    average_grads=False,
                    model_input_range=None,
                    crop=None,
                    model_input_size=None):

  '''
  data: [str to image folder, str to image, or dataloader]
  
  '''
    
  DEVICE = next(model.parameters()).device

  #make data loader
  if isinstance(data,str):
    if crop is None:
        print('crop arg not specified, default is  "True"')
        crop= True
    if model_input_size is None:
        print('model_input_size arg not specified, default is  %s'%str(default_model_input_size))
        model_input_size = default_model_input_size
    if model_input_range is None:
        print('model_input_range arg not specified, default is  %s'%str(default_model_input_range))
        model_input_range = default_model_input_range
    if os.path.isfile(data):
        data = img_to_img_tensor(data, crop=True, size = model_input_size)
        range_norm = range_normalize(img_range = model_input_range)
        data = range_norm(data)
        dataloader = [(data,None)]
    elif os.path.isdir(data):
        transforms = []
        if crop:
            transforms.append(LargestCenterCrop())
        transforms.append(Resize(model_input_size))
        transforms.append(ToTensor())
        transforms.append(range_normalize(model_input_range))
        preprocess = Compose(transforms)

        #dataloader
        kwargs = {'num_workers': num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in DEVICE.type else {}
        dataloader = DataLoader(image_data(data, transform=preprocess),
                            batch_size=batch_size,
                            shuffle=False,
                            **kwargs
                            )        
        
    else:
        data = range_norm(data)
        dataloader = [(data,None)]
        #raise NameError('%s does not exist'%data)

  else:
    dataloader = data

  target_activations = []
  cam_activations = {}
  cam_gradients = {}
  for layer in cam_layers:
      cam_activations[layer] = []
      cam_gradients[layer] = []


  #run data through model, stopping at target and saving activations and gradients throught
  with feature_target_saver(model,target_layer,unit) as target_saver:
      with actgrad_extractor(model,cam_layers) as score_saver:
          for i, data in enumerate(dataloader):

            inputs, label = data
            inputs = inputs.to(DEVICE)

            model.zero_grad() #very import!

            batch_target_activations = target_saver(inputs)
            if position == 'middle':
                batch_target_activations = batch_target_activations[:,batch_target_activations.shape[1]//2,batch_target_activations.shape[2]//2]
            elif position is not None:
                batch_target_activations = batch_target_activations[:,position[0],position[1]]

            #feature collapse
            #loss = loss_f(target_activations)
            loss = torch.sum(batch_target_activations)
            #overall_loss+=loss
            loss.backward()

            target_activations.append(batch_target_activations.detach().cpu())

          activations = score_saver.activations
          gradients = score_saver.gradients

          for l in cam_layers:
              if gradients[l] is None:
                  continue
              cam_activations[l].append(activations[l])
              cam_gradients[l].append(gradients[l])

  for l in cam_layers:
      if cam_activations[l] == []:
          del cam_activations[l]
      else:
          cam_activations[l] = torch.cat(cam_activations[l],dim=0)
      if cam_gradients[l] == []:
          del cam_gradients[l]
      else:
          cam_gradients[l] = torch.cat(cam_gradients[l],dim=0)

  target_activations = torch.cat(target_activations,dim=0)

  rl = torch.nn.ReLU()
  positive_cam_maps={}
  negative_cam_maps={}
  for l in cam_layers:
      if average_grads:
          pos_grads = (torch.mean(rl(cam_gradients[l]),dim=(2,3))).unsqueeze(-1).unsqueeze(-1)
          neg_grads = (torch.mean(rl(-1*cam_gradients[l]),dim=(2,3))).unsqueeze(-1).unsqueeze(-1)
          positive_actgrad_summap = torch.sum(cam_activations[l]*pos_grads, dim=(1))
          negative_actgrad_summap = torch.sum(cam_activations[l]*neg_grads, dim=(1))
      else:
          pos_grads = rl(cam_gradients[l])
          neg_grads = rl(-1*cam_gradients[l])
          positive_actgrad_summap = torch.sum(cam_activations[l]*pos_grads, dim=(1))
          negative_actgrad_summap = torch.sum(cam_activations[l]*neg_grads, dim=(1))

      del pos_grads
      del neg_grads

      #positive_maps.append(bilinear_upsample(positive_actgrad_summap, input_size))
      #negative_maps.append(bilinear_upsample(negative_actgrad_summap, input_size))
      positive_cam_maps[l] = positive_actgrad_summap
      negative_cam_maps[l] = negative_actgrad_summap

  return positive_cam_maps, negative_cam_maps, target_activations



def cam_maps_from_dataloader(data,
                    model,
                    target_layer,
                    unit,
                    cam_layers,
                    position = None,
                    batch_size = 64, 
                    num_workers = 2, 
                    average_grads=False,
                    model_input_range=None,
                    crop=None,
                    model_input_size=None,
                    negative= False,
                    relu_map=False):

    '''
    data: [str to image folder, str to image, or dataloader]

    '''
    DEVICE = next(model.parameters()).device
    if isinstance(cam_layers,str):
        cam_layers = [cam_layers]
        
    
    #make data loader
    if isinstance(data,str):
        if crop is None:
            print('crop arg not specified, default is  "True"')
            crop= True
        if model_input_size is None:
            print('model_input_size arg not specified, default is  %s'%str(default_model_input_size))
            model_input_size = default_model_input_size
        if model_input_range is None:
            print('model_input_range arg not specified, default is  %s'%str(default_model_input_range))
            model_input_range = default_model_input_range
    if os.path.isfile(data):
        data = img_to_img_tensor(data, crop=True, size = model_input_size)
        range_norm = range_normalize(img_range = model_input_range)
        data = range_norm(data)
        dataloader = [(data,None)]
    elif os.path.isdir(data):
        transforms = []
        if crop:
            transforms.append(LargestCenterCrop())
        transforms.append(Resize(model_input_size))
        transforms.append(ToTensor())
        transforms.append(range_normalize(model_input_range))
        preprocess = Compose(transforms)

        #dataloader
        kwargs = {'num_workers': num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in DEVICE.type else {}
        dataloader = DataLoader(image_data(data, transform=preprocess),
                            batch_size=batch_size,
                            shuffle=False,
                            **kwargs
                            )        
        
    else:
        data = range_norm(data)
        dataloader = [(data,None)]
        #raise NameError('%s does not exist'%data)

    target_activations = []
    cam_activations = {}
    cam_gradients = {}
    for layer in cam_layers:
        cam_activations[layer] = []
        cam_gradients[layer] = []


  #run data through model, stopping at target and saving activations and gradients throught
    with feature_target_saver(model,target_layer,unit) as target_saver:
        with actgrad_extractor(model,cam_layers) as score_saver:
            for i, data in enumerate(dataloader):

                inputs, label = data
                inputs = inputs.to(DEVICE)

                model.zero_grad() #very import!

                batch_target_activations = target_saver(inputs)
                if position == 'middle':
                    batch_target_activations = batch_target_activations[:,batch_target_activations.shape[1]//2,batch_target_activations.shape[2]//2]
                elif position is not None:
                    batch_target_activations = batch_target_activations[:,position[0],position[1]]

                #feature collapse
                #loss = loss_f(target_activations)
                loss = torch.sum(batch_target_activations)
                #overall_loss+=loss
                loss.backward()

                target_activations.append(batch_target_activations.detach().cpu())

            activations = score_saver.activations
            gradients = score_saver.gradients


            for l in cam_layers:
                if gradients[l] is None:
                    continue
                cam_activations[l].append(activations[l])
                cam_gradients[l].append(gradients[l])

    for l in cam_layers:
        if cam_activations[l] == []:
            del cam_activations[l]
        else:
            cam_activations[l] = torch.cat(cam_activations[l],dim=0)
        if cam_gradients[l] == []:
            del cam_gradients[l]
        else:
            cam_gradients[l] = torch.cat(cam_gradients[l],dim=0)

    target_activations = torch.cat(target_activations,dim=0)

    rl = torch.nn.ReLU()
    cam_maps={}

    for l in cam_layers:
        if average_grads:
            curr_cam_gradients = torch.mean(cam_gradients[l],dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
        else: 
            curr_cam_gradients = cam_gradients[l]
          
        sum_map = torch.sum(cam_activations[l]*curr_cam_gradients, dim=(1))
        if negative:
            sum_map = -1*sum_map
        if relu_map:
             sum_map = rl(sum_map)
        cam_maps[l] = sum_map

    return cam_maps, target_activations




#a function for returning an opacity value given an input cam_score
def thresholded_linear_map(x, in_min,in_max, out_min =0., out_max = 1.):
    if x <= in_min:
        return out_min
    elif x >= in_max:
        return out_max
    else:
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

thresholded_linear_map = np.vectorize(thresholded_linear_map)

from scipy.interpolate import interp2d,RectBivariateSpline

def bilinear_upsample(array, out_dim):
    if isinstance(out_dim,int):
        out_dim = (out_dim,out_dim)
    original_x = np.linspace(0, 1, array.shape[0])
    original_y = np.linspace(0, 1, array.shape[1])
    target_x = np.linspace(0, 1, out_dim[0])
    target_y = np.linspace(0, 1, out_dim[1])

    interpolator = interp2d(original_x, original_y, array, kind='linear')
    #interpolator = RectBivariateSpline(original_x, original_y, array)
    upsampled_array = interpolator(target_x, target_y)
    return upsampled_array


def posneg_cam_range(pos_maps,neg_maps,percentiles = [50,95],plot=False):
  #plot cam distributions (this isnt strictly necessary)

  # Filter out zeros and get pos/neg data distributions
  #pos_mask = pos_maps.flatten().ne(0)
  #neg_mask = neg_maps.flatten().ne(0)
  #data1 = np.array(pos_maps.flatten()[pos_mask])  
  #data2 = np.array(neg_maps.flatten()[neg_mask])
   
  data1 = np.array(pos_maps.flatten())  
  data2 = np.array(neg_maps.flatten()) 

  pos_range = []
  neg_range = []

  for percentile in percentiles:
    pos_range.append(max(np.percentile(data1, percentile),1e-8))
    neg_range.append(max(np.percentile(data2, percentile),1e-8))

  if plot:
    # Plot your data and the fitted distribution
    plt.hist(data1, bins=100, density=True, color='red', alpha=0.5, label='pos cam')
    plt.hist(data2, bins=100, density=True, color='blue', alpha=0.5, label='neg cam')


    for i in [0,1]:

        plt.axvline(x=pos_range[i], color='r', linestyle='--')
        plt.axvline(x=neg_range[i], color='b', linestyle='--')

        plt.text(pos_range[i], plt.gca().get_ylim()[1]*0.9, f'pos cam {percentiles[i]}%', color='red')
        plt.text(neg_range[i], plt.gca().get_ylim()[1]*0.8, f'neg cam {percentiles[i]}%', color='blue')

    plt.legend()
    plt.show()

  return pos_range, neg_range


def cam_range(maps,percentiles = None,plot=False):
  #plot cam distributions (this isnt strictly necessary)

  # Filter out zeros and get pos/neg data distributions
  #mask = maps.flatten().ne(0)
  #data = np.array(maps.flatten()[mask])  # Replace this with your data
  
  if percentiles is None:
    percentiles = list(range(1,100))

  data = np.array(maps.flatten())

  data_range = []
  for percentile in percentiles:
    data_range.append(max(np.percentile(data, percentile),1e-8))

  if plot:
    # Plot your data and the fitted distribution
    plt.hist(data, bins=100, density=True, color='red', alpha=0.5, label='cam')

    for i in [0,len(percentiles)-1]:
        plt.axvline(x=data_range[i], color='r', linestyle='--')
        plt.text(data_range[i], plt.gca().get_ylim()[1]*0.9, f'cam {percentiles[i]}%', color='red')

    plt.legend()
    plt.show()

  return data_range




def visualize_posneg_cam_map(img,pos_map,neg_map,
                             pos_range=None, neg_range=None, 
                             contrast=.4,lightness=1.5,
                             img_size=(512,512),
                             cmap='jet',
                             crop_bounds = None,
                             save=False):

    if isinstance(img,str):
      img = Image.open(img).resize(img_size).convert("L")
 
    image = np.array(img)
    image = (image - image.min()) / (image.max() - image.min()) #normalize (0-1)

    image = bilinear_upsample(image, img_size)


    #lighten and reduce contrast of original image
    midpoint = np.mean(image)
    shifted_image = image - midpoint
    low_contrast_image = shifted_image * contrast
    image = np.clip(low_contrast_image + midpoint*lightness, 0.0, 1.0)
    # Ensure the image is 3-channel (RGB) so it can be blended with the RGBA images
    image_rgb = np.stack([image, image, image], axis=-1)

    W, H = img_size[0], img_size[1]

    if pos_range is None:
      pos_range = [pos_map.min(),pos_map.max()]
    if neg_range is None:
      neg_range = [neg_map.min(),neg_map.max()]
    

    pos_map = thresholded_linear_map(pos_map,pos_range[0],pos_range[1])
    pos_map = bilinear_upsample(pos_map, img_size)  # Red heatmap
    pos_map = gaussian_filter(pos_map, sigma=2)

    neg_map = thresholded_linear_map(neg_map,neg_range[0],neg_range[1])
    neg_map = bilinear_upsample(neg_map, img_size)  # Blue heatmap
    neg_map = gaussian_filter(neg_map, sigma=2)
    
    diff_map = pos_map-neg_map
    #normalize from (-1,1) to (0,1)
    diff_map = (diff_map+1)/2
    diff_map_orig = diff_map
    
    #colorize
    if isinstance(cmap,str):
        cmap = cm.get_cmap(cmap)
    
    # Apply the colormap to the normalized difference map
    diff_map = cmap(diff_map)
    
    opacity_map = np.maximum(pos_map,neg_map)

    diff_rgb = diff_map[..., :3]  # Ensure this is RGB without alpha

    # Expand the dimensions of opacity_map to match the image RGB for element-wise multiplication
    opacity_map_expanded = np.expand_dims(opacity_map, axis=-1)

    # Apply the overlay considering the opacity
    # Where opacity_map is 1, diff_rgb fully replaces image_rgb; where it's 0, image_rgb remains
    image_with_overlay = image_rgb * (1 - opacity_map_expanded) + diff_rgb * opacity_map_expanded

    if crop_bounds is not None:
        image_with_overlay = image_with_overlay[crop_bounds[0][0]:crop_bounds[0][1],crop_bounds[1][0]:crop_bounds[1][1]]

    # Plot the result
    plt.imshow(image_with_overlay)

    # Hide axis labels and grid
    plt.axis('off')

    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)		

    # Display the plot
    plt.show()
    
    if crop_bounds is not None:
        diff_map_orig = diff_map_orig[crop_bounds[0][0]:crop_bounds[0][1],crop_bounds[1][0]:crop_bounds[1][1]]
        opacity_map = opacity_map[crop_bounds[0][0]:crop_bounds[0][1],crop_bounds[1][0]:crop_bounds[1][1]]

    return  diff_map_orig, opacity_map





#### GRADIENT BASED ANALYTIC RECEPTIVE FIELD  ####
    

def plot_recep_field_and_crop_bounds(recep_field, crop_bounds=None):
    # Plotting the heatmap
    plt.imshow(recep_field, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to a plot

    # Add a rectangle for the crop bounds
    # Rectangle takes (x, y, width, height)
    if crop_bounds is not None:
        top, bottom = crop_bounds[0][0], crop_bounds[0][1]-1
        left, right = crop_bounds[1][0], crop_bounds[1][1]-1
        rect = Rectangle((left, top), right-left, bottom-top, linewidth=1, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()


def find_crop_bounds(tensor, threshold, square=False):
    """
    Find the crop bounds for the tensor where the value exceeds the threshold.
    Optionally returns square crop bounds.

    Parameters:
    tensor (torch.Tensor): The 2D tensor representing the image.
    threshold (float): The threshold value.
    square (bool): If True, returns square crop bounds.

    Returns:
    list: Crop bounds in the format [[left, right], [top, bottom]].
    """
    h, w = tensor.shape

    # Initialize bounds
    left = 0
    right = w - 1
    top = 0
    bottom = h - 1

    # Find left bound
    for i in range(w):
        if torch.any(tensor[:, i] > threshold):
            left = i
            break

    # Find right bound
    for i in range(w-1, -1, -1):
        if torch.any(tensor[:, i] > threshold):
            right = i
            break

    # Find top bound
    for i in range(h):
        if torch.any(tensor[i, :] > threshold):
            top = i
            break

    # Find bottom bound
    for i in range(h-1, -1, -1):
        if torch.any(tensor[i, :] > threshold):
            bottom = i
            break

    # Adjust bounds to make them square, if requested
    if square:
        width = right - left
        height = bottom - top
        if width > height:
            diff = width - height
            half_diff = diff // 2
            top = max(0, top - half_diff)
            bottom = min(h - 1, bottom + diff - half_diff)
        elif height > width:
            diff = height - width
            half_diff = diff // 2
            left = max(0, left - half_diff)
            right = min(w - 1, right + diff - half_diff)

        # Ensure the bounds are still square after adjustment (important for edge cases)
        width = right - left
        height = bottom - top
        if width > height:
            if bottom < h - 1:
                bottom = top + width
            else:
                top = bottom - width
        elif height > width:
            if right < w - 1:
                right = left + height
            else:
                left = right - height

    return [[top, bottom+1],[left, right+1]]




def gradient_based_receptive_field(model,layer,
                                   position='middle',
                                   input_size = (224,224),
                                   batch_size=30,
                                   unit = None,
                                   input_range=(-2,2),
                                   iterations = 10,
                                   device='cuda',
                                   opt_alpha = .1,
                                   integrate = True,
                                   init_img = None,
                                   blur_sigma = 5,
                                   crop_threshold = .15,
                                   square = True,
                                   plot = False):
    
    normalizer = range_normalize(input_range)
    model = model.to(device)
    
    if init_img is None:
        input_img = torch.randn(batch_size,3,input_size[0],input_size[-1]).to(device)
    else:
        resizer = resize(input_size)
        input_img = resizer(init_img)
    input_img = normalizer(input_img)
    input_img.requires_grad_()
    int_gradients = torch.zeros(input_img.shape)
    
    #accumulate gradients
    with layer_saver(model, layer, detach=False, clone=False) as saver:
        
        for i in range(iterations):
            model.zero_grad()

            layer_acts = saver(input_img)[layer]
            if position == 'middle':
                position = (layer_acts.shape[-2]//2,layer_acts.shape[-1]//2)
            position_acts = layer_acts[:,:,position[0],position[-1]]
            if unit is not None:
                position_acts = layer_acts[:,unit,position[0],position[-1]]

            loss = torch.sum(position_acts)
            loss.backward()

            if integrate:
                int_gradients += torch.abs(input_img.grad.clone().detach().cpu())
            else:
                int_gradients = torch.abs(input_img.grad.clone().detach().cpu())

            with torch.no_grad():
                input_img += opt_alpha * input_img.grad

                # Clear the gradients after updating
                input_img.grad.zero_()
                

    recep_field = torch.sum(int_gradients,dim=(0,1))
    recep_field = min_max(gaussian_filter(recep_field, sigma=blur_sigma))
    #get a bounding box around the earliest position from the edges of recep_field
    # where the value is greater than the threshold (crop_threshold*max_value)
    crop_bounds = find_crop_bounds(torch.tensor(recep_field), 
                                   crop_threshold*recep_field.max(), 
                                   square= square)
    if plot:
        plot_recep_field_and_crop_bounds(recep_field, crop_bounds)
        
    return recep_field, crop_bounds


def gradient_based_crop_bounds_cnn_all_positions(model,layer,
                                                   input_size = (224,224),
                                                   batch_size=30,
                                                   input_range=(-2,2),
                                                   iterations = 10,
                                                   device='cuda',
                                                   opt_alpha = .1,
                                                   integrate = True,
                                                   blur_sigma = 5,
                                                   crop_threshold = .1,
                                                   add_padding = False,
                                                ):
    

    dummy_input = torch.randn(1,3,input_size[0],input_size[-1]).to(device)
    _ = model.to(device)
    with layer_saver(model, layer, detach=True, clone=True) as saver:
        model.zero_grad()
        layer_acts = saver(dummy_input)[layer]
        h,w = layer_acts.shape[-2],layer_acts.shape[-1]
        
    pos1 = (h//2-1,w//2-1)
    pos2 = (h//2,w//2)
    pos3 = (0,0)
    
    
    _,crop_bounds_1 = gradient_based_receptive_field(model,layer,
                                                    position=pos1,
                                                    input_size = input_size,
                                                    batch_size=batch_size,
                                                    input_range=input_range,
                                                    iterations = iterations,
                                                    device=device,
                                                    opt_alpha = opt_alpha,
                                                    integrate = integrate,
                                                    init_img = None,
                                                    blur_sigma = blur_sigma,
                                                    crop_threshold = crop_threshold,
                                                    square = True)
    
    _,crop_bounds_2 = gradient_based_receptive_field(model,layer,
                                                    position=pos2,
                                                    input_size = input_size,
                                                    batch_size=batch_size,
                                                    input_range=input_range,
                                                    iterations = iterations,
                                                    device=device,
                                                    opt_alpha = opt_alpha,
                                                    integrate = integrate,
                                                    init_img = None,
                                                    blur_sigma = blur_sigma,
                                                    crop_threshold = crop_threshold,
                                                    square = False)
    
    _,crop_bounds_3 = gradient_based_receptive_field(model,layer,
                                                    position=pos3,
                                                    input_size = input_size,
                                                    batch_size=batch_size,
                                                    input_range=input_range,
                                                    iterations = iterations,
                                                    device=device,
                                                    opt_alpha = opt_alpha,
                                                    integrate = integrate,
                                                    init_img = None,
                                                    blur_sigma = blur_sigma,
                                                    crop_threshold = crop_threshold,
                                                    square = False)
    
    first_step = ((crop_bounds_2[0][1] - crop_bounds_1[0][1])+(crop_bounds_2[1][1] - crop_bounds_1[1][1]))/2
    second_step = ((crop_bounds_2[0][1] - crop_bounds_3[0][1])/pos2[0]+(crop_bounds_2[1][1] - crop_bounds_3[1][1])/pos2[1])/2
    step_size = int((first_step+second_step)/2)
    
    average_crop_bounds = [
                           int((crop_bounds_2[0][0]+crop_bounds_2[1][0])/2),
                           int((crop_bounds_2[0][1]+crop_bounds_2[1][1])/2)
                          ]
    
    
                           
    
    #output is a tensor of shape hxwx2x2, that is the crop bounds corresponding to each position in the actmap
    out = torch.zeros(h,w,2,2,dtype=torch.int)
    
    for i in range(h):
        i_bounds = [average_crop_bounds[0]+step_size*(i-pos2[0]),average_crop_bounds[1]+step_size*(i-pos2[0])]
        if not add_padding: 
            i_bounds = [max(i_bounds[0],0),min(i_bounds[1],input_size[1]-1)]
        for j in range(w):
            j_bounds = [average_crop_bounds[0]+step_size*(j-pos2[1]),average_crop_bounds[1]+step_size*(j-pos2[1])]
            if not add_padding: 
                j_bounds = [max(j_bounds[0],0),min(j_bounds[1],input_size[0]-1)]
            out[i,j] = torch.tensor([j_bounds,i_bounds])    #this is weirdly getting flipped somewhere . . .
    
    #return crop_bounds_1, crop_bounds_2, crop_bounds_3, h,w
    return out
