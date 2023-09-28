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

import numpy as np
import torch
import torch.nn.functional as F
from decorator import decorator
from faccent.objectives_util import _make_arg_str,_extract_act_pos,_T_handle_batch, linconv_reshape, orthogonal_proj


class Objective():

    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model):
        return self.objective_func(model)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other + self(model)
            name = self.name
            description = self.description
        else:
            objective_func = lambda model: self(model) + other(model)
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

    @staticmethod
    def sum(objs):
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other * self(model)
            return Objective(objective_func, name=self.name, description=self.description)
        else:
            # Note: In original Lucid library, objectives can be multiplied with non-numbers
            # Removing for now until we find a good use case
            raise TypeError('Can only multiply by int or float. Received type ' + str(type(other)))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        else:
            raise TypeError('Can only divide by int or float. Received type ' + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


def wrap_objective():
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)
    return inner


def handle_batch(batch=None):
    return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))


@wrap_objective()
def neuron(layer, n_channel, x=None, y=None, batch=None):
    """Visualize a single neuron of a single channel.

    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.

    Odd width & height:               Even width & height:

    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+

    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -layer_t[: ,:, n_channel].mean()
    return inner


@wrap_objective()
def channel(layer, n_channel, batch=None):
    """Visualize a single channel"""
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[: ,:, n_channel].mean()
    return inner

@wrap_objective()
def neuron_weight(layer, weight, x=None, y=None, batch=None):
    """ Linearly weighted channel activation at one location as objective
    weight: a torch Tensor vector same length as channel.
    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        if weight is None:
            return -layer_t.mean()
        else:
            return -(layer_t.squeeze() * weight).mean()
    return inner

@wrap_objective()
def channel_weight(layer, weight, batch=None):
    """ Linearly weighted channel activation as objective
    weight: a torch Tensor vector same length as channel. """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        return -(layer_t * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def localgroup_weight(layer, weight=None, x=None, y=None, wx=1, wy=1, batch=None):
    """ Linearly weighted channel activation around some spot as objective
    weight: a torch Tensor vector same length as channel. """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        if weight is None:
            return -(layer_t[:, :, y:y + wy, x:x + wx]).mean()
        else:
            return -(layer_t[:, :, y:y + wy, x:x + wx] * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def direction(layer, direction, batch=None):
    """Visualize a direction

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model):
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), model(layer)).mean()

    return inner


@wrap_objective()
def direction_neuron(layer,
                     direction,
                     x=None,
                     y=None,
                     batch=None):
    """Visualize a single (x, y) position along the given direction

    Similar to the neuron objective, defaults to the center neuron.

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction_neuron(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model):
        # breakpoint()
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), layer_t).mean()

    return inner


def _torch_blur(tensor, out_c=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[ :  ,  :  ] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().to(device)
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)
    return conv_f(tensor) / conv_f(torch.ones_like(tensor))


@wrap_objective()
def blur_input_each_step():
    """Minimizing this objective is equivelant to blurring input each step.
    Optimizing (-k)*blur_input_each_step() is equivelant to:
    input <- (1-k)*input + k*blur(input)
    An operation that was used in early feature visualization work.
    See Nguyen, et al., 2015.
    """
    def inner(T):
        t_input = T("input")
        with torch.no_grad():
            t_input_blurred = _torch_blur(t_input)
        return -0.5*torch.sum((t_input - t_input_blurred)**2)
    return inner


@wrap_objective()
def channel_interpolate(layer1, n_channel1, layer2, n_channel2):
    """Interpolate between layer1, n_channel1 and layer2, n_channel2.
    Optimize for a convex combination of layer1, n_channel1 and
    layer2, n_channel2, transitioning across the batch.
    Args:
        layer1: layer to optimize 100% at batch=0.
        n_channel1: neuron index to optimize 100% at batch=0.
        layer2: layer to optimize 100% at batch=N.
        n_channel2: neuron index to optimize 100% at batch=N.
    Returns:
        Objective
    """
    def inner(model):
        batch_n = list(model(layer1).shape)[0]
        arr1 = model(layer1)[:, n_channel1]
        arr2 = model(layer2)[:, n_channel2]
        weights = np.arange(batch_n) / (batch_n - 1)
        sum_loss = 0
        for n in range(batch_n):
            sum_loss -= (1 - weights[n]) * arr1[n].mean()
            sum_loss -= weights[n] * arr2[n].mean()
        return sum_loss
    return inner


@wrap_objective()
def alignment(layer, decay_ratio=2):
    """Encourage neighboring images to be similar.
    When visualizing the interpolation between two objectives, it's often
    desirable to encourage analogous objects to be drawn in the same position,
    to make them more comparable.
    This term penalizes L2 distance between neighboring images, as evaluated at
    layer.
    In general, we find this most effective if used with a parameterization that
    shares across the batch. (In fact, that works quite well by itself, so this
    function may just be obsolete.)
    Args:
        layer: layer to penalize at.
        decay_ratio: how much to decay penalty as images move apart in batch.
    Returns:
        Objective.
    """
    def inner(model):
        batch_n = list(model(layer).shape)[0]
        layer_t = model(layer)
        accum = 0
        for d in [1, 2, 3, 4]:
            for i in range(batch_n - d):
                a, b = i, i + d
                arr_a, arr_b = layer_t[a], layer_t[b]
                accum += ((arr_a - arr_b) ** 2).mean() / decay_ratio ** float(d)
        return accum
    return inner


@wrap_objective()
def diversity(layer):
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates the correlation matrix of activations at layer
    for each image, and then penalizes cosine similarity between them. This is
    very similar to ideas in style transfer, except we're *penalizing* style
    similarity instead of encouraging it.

    Args:
        layer: layer to evaluate activation correlations on.

    Returns:
        Objective.
    """
    def inner(model):
        layer_t = model(layer)
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner


@wrap_objective()
def dot_cossim_compare(layer, batch=1,comp_batch=0, cossim_pow=1.):
  def inner(T):
    x = T(layer) # model output, with additional transform dimension
    x1 = x[:,batch]
    x2 = x[:,comp_batch]
    cossim = F.cosine_similarity(x1.view(x1.size(0),-1), x2.view(x2.size(0),-1))
    dot = (x1 * x2).sum(tuple(range(1, (x.dim()-1))))
    return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
  return inner



@wrap_objective()
def dot_seperate_cossim_compare(layer, feature, batch=0,comp_batch=1, cossim_pow=1.,orthogonal=True, comp_layer = None):
  def inner(T, feature=feature):
    x = T(layer)[:,batch]

    #dot product with feature
    if isinstance(feature,int): #unit direction
      dot = x[:,feature] 
    else: #take dot product in desired direction
      feature = linconv_reshape(torch.tensor(feature).to(x.device),x)
      dot = (x * feature).sum(dim=1)
   
    x1 = T(comp_layer or layer)[:,batch]
    x2 = T(comp_layer or layer)[:,comp_batch]
    #import pdb; pdb.set_trace()
    #cosine sim with image
    if orthogonal and (comp_layer == layer):
        x1 = orthogonal_proj(x,feature)
        x2 = orthogonal_proj(x2,feature)
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    cossim = (x1_norm * x2_norm).sum(dim=1)

    #return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
    return -torch.mean(dot) * torch.mean(cossim ** cossim_pow)
  return inner


@wrap_objective()
def l1_compare(layer, batch=0,comp_batch=1):
#   def inner(T):
#     x = T(layer) # model output, with additional transform dimension
#     dot = (x[:,batch] * x[:,comp_batch]).sum(tuple(range(1, (x.dim()-1)))) #dot product 
#     mag = torch.sqrt((x[:,comp_batch]**2).sum(tuple(range(1, (x.dim()-1)))))
#     cossim = dot/(1e-6 + mag)
#     return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
#   return inner
  def inner(T):
    x = T(layer) # model output, with additional transform dimension
    x1 = x[:,batch]
    x2 = x[:,comp_batch]
    return -torch.mean(torch.abs(x1-x2)) #averages across transforms
  return inner


@wrap_objective()
def l2_compare(layer, batch=0,comp_batch=1,p=2):
#   def inner(T):
#     x = T(layer) # model output, with additional transform dimension
#     dot = (x[:,batch] * x[:,comp_batch]).sum(tuple(range(1, (x.dim()-1)))) #dot product 
#     mag = torch.sqrt((x[:,comp_batch]**2).sum(tuple(range(1, (x.dim()-1)))))
#     cossim = dot/(1e-6 + mag)
#     return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
#   return inner
  def inner(T):
    x = T(layer) # model output, with additional transform dimension
    x1 = x[:,batch]
    x2 = x[:,comp_batch]
    distances = torch.norm(x1-x2,dim=1,p=p)
    return -distances.mean()
  return inner

# @wrap_objective()
# def l2_compare(layer, batch=0,comp_batch=1):
# #   def inner(T):
# #     x = T(layer) # model output, with additional transform dimension
# #     dot = (x[:,batch] * x[:,comp_batch]).sum(tuple(range(1, (x.dim()-1)))) #dot product 
# #     mag = torch.sqrt((x[:,comp_batch]**2).sum(tuple(range(1, (x.dim()-1)))))
# #     cossim = dot/(1e-6 + mag)
# #     return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
# #   return inner
#   def inner(T):
#     x = T(layer) # model output, with additional transform dimension
#     x1 = x[:,batch]
#     x2 = x[:,comp_batch]
#     # diffs_squared = (x1-x2) ** 2
#     # l2_distances_squared = diffs_squared.sum(dim=1)
#     # l2_distances = torch.sqrt(l2_distances_squared)

#     return -torch.norm(x1.flatten()-x2.flatten())
#   return inner





def pearson_correlation(x, y):
    x = x.flatten()
    y = y.flatten()
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    cov_xy = torch.mean((x - mean_x) * (y - mean_y))
    std_x = torch.std(x, unbiased=False)
    std_y = torch.std(y, unbiased=False)
    
    return cov_xy / (std_x * std_y)

@wrap_objective()
def dot_seperate_pearson_compare(layer, feature, batch=0,comp_batch=1, pearson_pow=1.,orthogonal=True, comp_layer = None):
  def inner(T, feature=feature):

    x = T(layer)[:,batch]

    #dot product with feature
    if isinstance(feature,int): #unit direction
      dot = x[:,feature] 
    else: #take dot product in desired direction
      feature = linconv_reshape(torch.tensor(feature).to(x.device),x)
      dot = (x * feature).sum(dim=1)
   
    x1 = T(comp_layer or layer)[:,batch]
    x2 = T(comp_layer or layer)[:,comp_batch]

    #pearson with image
    if orthogonal and (comp_layer == layer):
        x1 = orthogonal_proj(x,feature)
        x2 = orthogonal_proj(x2,feature)
    pearson = pearson_correlation(x1,x2)

    #return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
    return -torch.mean(dot) * torch.mean(pearson ** pearson_pow)
  return inner





@wrap_objective()
def cossim_compare(layer, batch=1,comp_batch=0):
#   def inner(T):
#     x = T(layer) # model output, with additional transform dimension
#     dot = (x[:,batch] * x[:,comp_batch]).sum(tuple(range(1, (x.dim()-1)))) #dot product 
#     mag = torch.sqrt((x[:,comp_batch]**2).sum(tuple(range(1, (x.dim()-1)))))
#     cossim = dot/(1e-6 + mag)
#     return torch.mean(-dot * cossim ** cossim_pow) #averages across transforms
#   return inner
  def inner(T):
    x = T(layer) # model output, with additional transform dimension
    x1 = x[:,batch]
    x2 = x[:,comp_batch]
    return -torch.mean(-F.cosine_similarity(x1.view(x1.size(0),-1), x2.view(x2.size(0),-1))) #averages across transforms
  return inner








def as_objective(obj):
    """Convert obj into Objective class.

    Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives are returned unchanged.

    Args:
        obj: string or Objective.

    Returns:
        Objective
    """
    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        return obj
    if isinstance(obj, str):
        layer, chn = obj.split(":")
        layer, chn = layer.strip(), int(chn)
        return channel(layer, chn)
