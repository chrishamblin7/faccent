import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, reduce, repeat
import seaborn as sns
sns.set(font_scale=0.9)
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
import pickle
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter



### General ###

#some default params
default_img_size = (512,512) #size of visualizations
default_model_input_range = (-2,2) # range of values model expects
#NOTE: inceptionv1 range is [-117, 138], imagenet default is roughly [-2,2]
default_model_input_size = (224,224)


### SIMPLE FUNCTIONS ###

def safe_logit(x, eps=1e-6):
		return torch.logit(x.clamp(min=eps, max=1-eps))

def r(n):
	return round(float(n), 3)

def clip_percentile(img, p=0.1):
	return np.clip(img, None, np.percentile(img, 100-p))

def min_max(t):
    return (t - t.min()) / (t.max() - t.min())
 

### preprocessing ###

class LargestCenterCrop(nn.Module):
		def __init__(self):
				super(LargestCenterCrop, self).__init__()

		def forward(self, img: Image.Image) -> Image.Image:
				"""
				Args:
						img (PIL.Image.Image): Image to be cropped.

				Returns:
						PIL.Image.Image: Cropped image.
				"""
				return TF.center_crop(img, min(img.size))

		def __repr__(self):
				return self.__class__.__name__ + "()"

def range_normalize(img_range = default_model_input_range):
		def inner(image_t):
				return image_t*(img_range[1] - img_range[0]) + img_range[0]
		return inner

default_preprocess = Compose([LargestCenterCrop(),
															Resize(default_model_input_size),
															ToTensor(),
															range_normalize(default_model_input_range)])


### Images  ####

totensor = ToTensor()
def img_to_img_tensor(img, 
					  size=default_img_size, 
					  crop=True
										 ):
	'''
	function for preparing PIL images to torch intermediary format.
	pass the output of this function to img_tensor_to_model_input before model
	Args:
		img: PIL image
	Returns:
		tensor of shape (w h c) between (0,1) of size CANVAS_SIZE
	'''
	if isinstance(img,str):
		img = Image.open(os.path.abspath(img))
		if img.mode == 'RGBA':
			img = img.convert('RGB')
		if crop:
			cropper = LargestCenterCrop()
			img = cropper(img)
	if not isinstance(img, torch.Tensor): img = totensor(img)
	if img.dim() == 3: img = img.unsqueeze(0)  # shape now is (1, C, H, W)
	
	# Use interpolate to resize
	img = F.interpolate(img, size=size).clamp(min=0., max=1.)
	return img

def torch_to_numpy(img):
	try:
		img = img.detach().cpu().numpy()
	except:
		img = np.array(img)

	if len(img.shape) == 3:
		if img.shape[0] == 3:
			img = np.moveaxis(img, 0, -1)
	elif len(img.shape) == 4:
		if img.shape[1] == 3:
			img = np.moveaxis(img, 1, -1)

	return img.astype(np.float32)


### PLOTTING ###

def set_size(w,h):
	"""Set matplot figure size"""
	plt.rcParams["figure.figsize"] = [w,h]

def show(img,normalize=True):
	img = torch_to_numpy(img)

	if normalize:
		img -= img.min()
		img /= img.max()

	if len(img.shape)==3:
		plt.imshow(img)
		plt.axis('off')
	else:
		n = img.shape[0]
		for i in range(n):
			plt.subplot(1, n, i + 1)
			plt.imshow(img[i])
			plt.axis('off')
		plt.show()

def plot(t, clip=0.1, normalize=True):
	""" Remove outlier and plot image """
	t = torch_to_numpy(t)
	if len(t.shape)==3:
		t = clip_percentile(t, clip)
		if normalize:
			t -= t.mean(); t /= t.std()
			t -= t.min(); t /= t.max()
		plt.imshow(t)
		plt.axis('off')
	else:
		n = t.shape[0]
		for i in range(n):
			plt.subplot(1, n, i + 1)
			curr_img = clip_percentile(t[i], clip)
			if normalize:
				curr_img -= curr_img.mean(); curr_img /= curr_img.std()
				curr_img -= curr_img.min(); curr_img /= curr_img.max()

			plt.imshow(curr_img)
			plt.axis('off')
		plt.show()



def plot_alpha(img, tr, p=10, save=False, show=True, blur_sigma=2,crop=None):
    """ Remove outlier and plot image (take care of merging the alpha) """
    img = torch_to_numpy(img)
    tr = torch_to_numpy(tr)
    
    # Handle single image
    if len(img.shape) == 3:
        img = img[np.newaxis, ...]
        tr = tr[np.newaxis, ...]

    n = img.shape[0]
	
    for i in range(n):
        plt.subplot(1, n, i + 1)
        curr_img = img[i]
        curr_tr = tr[i]

        if curr_tr.shape[0] == 1:
            curr_tr = np.moveaxis(curr_tr, 0, -1)

        curr_img -= curr_img.mean()
        curr_img /= curr_img.std()
        curr_img -= curr_img.min()
        curr_img /= curr_img.max()

        curr_tr = np.mean(np.array(curr_tr).copy(), -1, keepdims=True)
        curr_tr = clip_percentile(curr_tr, p)
        curr_tr = curr_tr / curr_tr.max()

        # Blur transparency
        curr_tr = curr_tr.squeeze()
        curr_tr = gaussian_filter(curr_tr, sigma=blur_sigma)
        curr_tr = curr_tr[:, :, np.newaxis]

        if crop is None:
            plt.imshow(np.concatenate([curr_img, curr_tr], -1))
        else:
            curr_img = curr_img[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
            curr_tr = curr_tr[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
            plt.imshow(np.concatenate([curr_img, curr_tr], -1))
            
        plt.axis('off')


    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
    else:
        plt.clf()        
        


import ipywidgets as widgets

def accent_widget(imgs, img_trs,tr_step=1,batch=...):
		# Create widgets
		img_slider = widgets.IntSlider(min=0, max=len(imgs)-1, step=1, description='Opt Steps:')
		p_slider = widgets.FloatSlider(min=0, max=100, value=100, step=tr_step, description='Transparency:')
		
		# Function that takes image index and p value, then plots accordingly
		def interact_accent(img_index, trans_p):
				if trans_p == 100:
						plot(imgs[img_index][batch])
				else:
						plot_alpha(imgs[img_index][batch], img_trs[img_index][batch], p=trans_p)
		
		# Interact function creates UI controls for function arguments, and then calls the function with those arguments
		widgets.interact(interact_accent, img_index=img_slider, trans_p=p_slider)


###EXCEPTIONS ###


#Error classes forr breaking forward pass of model
# define Python user-defined exceptions
class ModelBreak(Exception):
	"""Base class for other exceptions"""
	pass

class TargetReached(ModelBreak):
	"""Raised when the output target for a subgraph is reached, so the model doesnt neeed to be run forward any farther"""
	pass  

###DATALOADING###

class image_data(Dataset):

	def __init__(self, root_dir, transform=default_preprocess, label_file_path = None, label_dict_path = None, class_folders=False,select_folders = None,return_image_name=False,rgb=False ):
		
		
		self.root_dir = root_dir
		self.class_folders = class_folders
		self.return_image_name = return_image_name

		if select_folders is not None:
			self.img_names = []
			self.classes = select_folders
			for cl in self.classes:
				files = os.listdir(self.root_dir+'/'+cl)
				full_names = [cl+'/'+s for s in files]
				self.img_names += full_names
		elif not self.class_folders:
			self.img_names = os.listdir(self.root_dir)
			self.img_names.sort()
		else:
			self.img_names = []
			self.classes = os.listdir(self.root_dir)
			for cl in self.classes:
				files = os.listdir(self.root_dir+'/'+cl)
				full_names = [cl+'/'+s for s in files]
				self.img_names += full_names


		self.label_list = None
		self.label_file_path = label_file_path
		if self.label_file_path is not None:
			label_file = open(label_file_path,'r')
			self.label_list = [x.strip() for x in label_file.readlines()]
			label_file.close()

		self.label_dict_path = label_dict_path
		self.label_dict = None
		if self.label_dict_path is not None:
			self.label_dict = pickle.load(open(label_dict_path,'rb'))

		if not transform:
			transform = ToTensor()
		self.transform = transform

		self.rgb = rgb

	def __len__(self):
		return len(self.img_names)

	def get_label_from_name(self,img_name):
		#check for label dict
		if self.label_dict is not None:
			if img_name not in self.label_dict.keys():
				return torch.tensor(9999999)
			else:
				return self.label_dict[img_name]
		else: #assume its a discrete one-hot label	
			if self.label_list is None:
				return torch.tensor(9999999)
			label_name = None
			label_num = None
			for i in range(len(self.label_list)): # see if any label in file name
				if self.label_list[i] in img_name:
					if label_name is None:
						label_name =  self.label_list[i]
						label_num = i
					elif len(self.label_list[i]) > len(label_name):
						label_name = self.label_list[i]
						label_num = i
			target = torch.tensor(9999999)
			if label_num is not None:
				target = torch.tensor(label_num)
			return target      

	def __getitem__(self, idx):

		img_path = os.path.join(self.root_dir,self.img_names[idx])
		img = Image.open(img_path)
		if self.rgb:
			img = img.convert("RGB")
		img = self.transform(img).float()
		if img.shape[0] == 1: #replicate grayscale image
			img = img.repeat(3, 1, 1)
		label = self.get_label_from_name(self.img_names[idx])
		if self.return_image_name:
			return(img,label,self.img_names[idx])
		else:
			return (img,label)
		

"""
Pretty plots option for explanations
"""
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from typing import Optional, Union


def _normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image in [0, 1].

    Parameters
    ----------
    image
        Image to prepare.

    Returns
    -------
    image
        Image ready to be used with matplotlib (in range[0, 1]).
    """
    image = np.array(image, np.float32)

    image -= image.min()
    image /= image.max()

    return image

def _clip_percentile(heatmap: np.ndarray,
                     percentile: float) -> np.ndarray:
    """
    Apply clip according to percentile value (percentile, 100-percentile) of a heatmap
    only if percentile is not None.

    Parameters
    ----------
    heatmap
        Heatmap to clip.

    Returns
    -------
    heatmap_clipped
        Heatmap clipped accordingly to the percentile value.
    """
    assert len(heatmap.shape) == 2 or heatmap.shape[-1] == 1, "Clip percentile is only supposed"\
                                                              "to be applied on heatmap."
    assert 0. <= percentile <= 100., "Percentile value should be in [0, 100]"

    if percentile is not None:
        clip_min = np.percentile(heatmap, percentile)
        clip_max = np.percentile(heatmap, 100. - percentile)
        heatmap = np.clip(heatmap, clip_min, clip_max)

    return heatmap


def plot_attribution(explanation,
                      image: Optional[np.ndarray] = None,
                      cmap: str = "jet",
                      alpha: float = 0.5,
                      clip_percentile: Optional[float] = 0.1,
                      absolute_value: bool = False,
                      save = False,
					  crop=None,
                      **plot_kwargs):
    """
    Displays a single explanation and the associated image (if provided).
    Applies a series of pre-processing to facilitate the interpretation of heatmaps.

    Parameters
    ----------
    explanation
        Attribution / heatmap to plot.
    image
        Image associated to the explanations.
    cmap
        Matplotlib color map to apply.
    alpha
        Opacity value for the explanation.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if image is not None:
        image = _normalize(image)
        if crop is not None:
            image = image[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
        plt.imshow(image)

    if explanation.shape[-1] == 3:
        explanation = np.mean(explanation, -1)

    if absolute_value:
        explanation = np.abs(explanation)

    if clip_percentile:
        explanation = _clip_percentile(explanation, clip_percentile)

    explanation = _normalize(explanation)
    if crop is not None:
        explanation = explanation[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]

    plt.imshow(explanation, cmap=cmap, alpha=alpha, **plot_kwargs)
    plt.axis('off')
    
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
		# Save the figure

    
    

def plot_attribution_with_variable_opacity(explanation, 
                                          image: Optional[np.ndarray] = None,
                                          cmap: str = "jet",
                                          alpha: float = 0.5,
                                          clip_percentile: Optional[float] = 0.1,
                                          ramp: float = 1.,
                                          absolute_value: bool = False,
                                          image_lightening: float = 0.,
                                          **plot_kwargs):

    # If image is provided, display it
    if image is not None:
        image = _normalize(image)
        image+= image_lightening
        image = np.clip(image, 0, 1)
        plt.imshow(image)

    if explanation.shape[-1] == 3:
        explanation = np.mean(explanation, -1)
        #explanation = np.linalg.norm(explanation, ord=2, axis=2)
    
    if absolute_value:
        explanation = np.abs(explanation)
        
    if clip_percentile:
        explanation = _clip_percentile(explanation, clip_percentile)
        
    explanation = _normalize(explanation)
    
    explaination = explanation

    # Get the colormap
    my_cmap = cm.get_cmap(cmap)

    # Apply the colormap like a function to any array:
    img = my_cmap(explanation)

    # Modify alpha channel based on explanation values
    img[..., 3] = (explanation**ramp)*alpha

    # Display the image
    plt.imshow(img)
    plt.axis('off')

    plt.show()
    
    


def plot_attributions(
        explanations: np.ndarray,
        images: Optional[np.ndarray] = None,
        cmap: str = "viridis",
        alpha: float = 0.5,
        clip_percentile: Optional[float] = 0.1,
        absolute_value: bool = False,
        cols: int = 5,
        img_size: float = 2.,
        **plot_kwargs
):
    """
    Displays a series of explanations and their associated images if these are provided.
    Applies pre-processing to facilitate the interpretation of heatmaps.

    Parameters
    ----------
    explanations
        Attributions values to plot.
    images
        Images associated to explanations. If provided, there must be one explanation for each
        image.
    cmap
        Matplotlib color map to apply.
    alpha
        Opacity value for the explanation.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    cols
        Number of columns.
    img_size:
        Size of each subplots (in inch), considering we keep aspect ratio
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if images is not None:
        assert len(images) == len(explanations), "If you provide images, there must be as many " \
                                                 "as explanations."

    rows = ceil(len(explanations) / cols)
    # get width and height of our images
    l_width, l_height = explanations.shape[1:3]

    # define the figure margin, width, height in inch
    margin = 0.3
    spacing = 0.3
    figwidth = cols * img_size + (cols-1) * spacing + 2 * margin
    figheight = rows * img_size * l_height/l_width + (rows-1) * spacing + 2 * margin

    left = margin/figwidth
    bottom = margin/figheight

    fig = plt.figure()
    fig.set_size_inches(figwidth, figheight)

    fig.subplots_adjust(
        left = left,
        bottom = bottom,
        right = 1.-left,
        top = 1.-bottom,
        wspace = spacing/img_size,
        hspace= spacing/img_size * l_width/l_height
    )

    for i, explanation in enumerate(explanations):
        plt.subplot(rows, cols, i+1)

        if images is not None:
            img = _normalize(images[i])
            if img.shape[-1] == 1:
                plt.imshow(img[:,:,0], cmap="Greys")
            else:
                plt.imshow(img)

        plot_attribution(explanation, cmap=cmap, alpha=alpha, clip_percentile=clip_percentile,
                         absolute_value=absolute_value, **plot_kwargs)
