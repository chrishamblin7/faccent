import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
import pickle
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter
import subprocess
import imageio


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
		if img.mode == 'L':
			img = img.convert('RGB')   # Convert grayscale to RGB

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

def show(img,normalize=True, clip=.01, save=None, display=True):
    img = torch_to_numpy(img)

    if len(img.shape)==3:
        if clip is not None:
            img = clip_percentile(img, clip)
        if normalize:
            img -= img.min()
            img /= img.max()
        plt.imshow(img)
        plt.axis('off')
    else:
        n = img.shape[0]
        for i in range(n):
            plt.subplot(1, n, i + 1)
            curr_img = img[i]
            if clip is not None:
                curr_img = clip_percentile(curr_img, clip)
            if normalize:
                curr_img -= curr_img.mean(); curr_img /= curr_img.std()
                curr_img -= curr_img.min(); curr_img /= curr_img.max()

            plt.imshow(curr_img)
            plt.axis('off')

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)		

    if display:
        plt.show()
    else:
        plt.close()
        plt.clf()



def plot(t, clip=0.1, normalize=True):
	""" Remove outlier and plot image """
	t = torch_to_numpy(t)
	if len(t.shape)==3:
		if clip is not None:
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
			curr_img = t[i]
			if clip is not None:
				curr_img = clip_percentile(curr_img, clip)
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
		plt.close()
		


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


def get_crop_bounds(image_size, grid_size, position, spread=1):
	'''
	Returns the bounding box for the crop with consistent size, ensuring all values are integers.
	'''
	# Calculate the size of each grid cell
	cell_height = image_size[0] / grid_size[0]
	cell_width = image_size[1] / grid_size[1]

	# Apply the spread factor
	spread_height = int(cell_height * spread)
	spread_width = int(cell_width * spread)

	# Calculate the top left corner of the grid cell (without spread)
	grid_top = position[0] * cell_height
	grid_left = position[1] * cell_width

	# Calculate the start and end points with spread, ensuring consistent crop size
	h_start = max(0, int(grid_top + cell_height / 2 - spread_height / 2))
	h_end = min(image_size[0], h_start + spread_height)
	w_start = max(0, int(grid_left + cell_width / 2 - spread_width / 2))
	w_end = min(image_size[1], w_start + spread_width)

	return [(h_start, h_end), (w_start, w_end)]





from faccent.modelzoo.inceptionv1 import helper_layers

def redirect_model_relus(model):
	"""
	Replaces all nn.ReLU layers in the model with RedirectedReluLayer.
	"""
	relu = helper_layers.RedirectedReluLayer

	# recursive function to replace ReLU layers
	def replace_relus(net, path=[]):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():
				if isinstance(layer, nn.ReLU):
					print('Replacing ReLU at:', '->'.join(path + [name]))
					net._modules[name] = relu()
				replace_relus(layer, path=path + [name])

	replace_relus(model)



def get_middle_crop_position(img, spread=.5):
	h,w = img.shape[-2], img.shape[-1]
	crop_position = [
		[int(h/2) - int(h/2)*spread,int(h/2) + int(h/2)*spread],
		[int(w/2) - int(w/2)*spread,int(w/2) + int(w/2)*spread],
				 ]
	return crop_position



def prepare_image(img_batch, tr_batch, p=10, blur_sigma=2, use_transparency=True,crop=None):
	"""Prepare a batch of images with transparency for video frame."""

	def torch_to_numpy(tensor):
		"""Convert PyTorch tensor to NumPy array."""
		return tensor.detach().cpu().numpy()

	def clip_percentile(img, clip):
		"""Apply percentile clipping to an image."""
		low, high = np.percentile(img, [clip, 100 - clip])
		img = np.clip(img, low, high)
		return img


	img_batch = torch_to_numpy(img_batch)
	tr_batch = torch_to_numpy(tr_batch)

	try:
		img_batch = img_batch.unsqueeze(0)
		tr_batch = tr_batch.unsqueeze(0)
	except:
		pass

	if len(img_batch.shape) == 4: img_batch = img_batch[0]
	if len(tr_batch.shape) == 4: tr_batch = tr_batch[0]

	# Ensure channel last format (Height, Width, Channels)
	img_batch = np.transpose(img_batch, (1, 2, 0))
	tr_batch = np.transpose(tr_batch, (1, 2, 0))

	# Normalize and clip the image
	img_batch -= img_batch.mean()
	img_batch /= img_batch.std()
	img_batch -= img_batch.min()
	img_batch = img_batch / img_batch.max()

	# Process transparency mask
	tr_batch = np.mean(tr_batch, -1, keepdims=True)
	tr_batch = clip_percentile(tr_batch, p)
	tr_batch = tr_batch / tr_batch.max()

	# Blur transparency
	tr_batch = gaussian_filter(tr_batch.squeeze(), sigma=blur_sigma)
	tr_batch = tr_batch[:, :, np.newaxis]

	# Apply the transparency mask
	if use_transparency:
		white_background = np.ones_like(img_batch)  # White background
		combined = img_batch * tr_batch + white_background * (1 - tr_batch)
	else:
		combined = img_batch


	# Scale to 0-255 and convert to uint8
	combined = (255 * combined).clip(0, 255).astype(np.uint8)

	# Crop if necessary
	if crop:
		combined = combined[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]

	return combined




def images_to_video(imgs, img_trs, video_filename, 
					fps=10, p=30, blur_sigma=2, 
					crop=None, use_transparency=True,
					codec='libx264',
					pixelformat='yuv420p',
					quality=10,
					recode = False):
	"""Convert list of batches of images and transparencies to a video file using imageio."""
	temp_filename = '.'.join(video_filename.split('.'))+'_temp.'+video_filename.split('.')[-1]
	# writer = imageio.get_writer(temp_filename, 
	# 							fps=fps, 
	# 							codec = codec,
	# 							quality=quality,
	# 							pixelformat = pixelformat)
	writer = imageio.get_writer(temp_filename, 
								fps=fps, 
								)

	for img_batch, tr_batch in zip(imgs, img_trs):
		frame = prepare_image(img_batch, tr_batch, p, blur_sigma, crop=crop, use_transparency=use_transparency)
		if frame is not None and frame.size > 0:
			# Convert frame to uint8 if it's not already
			if frame.dtype != np.uint8:
				frame = (255 * frame).clip(0, 255).astype(np.uint8)
			writer.append_data(frame)
		else:
			print("Skipped a frame: Empty or None")

	writer.close()

	if recode:
		ffmpeg_command = [
			'ffmpeg', '-i', temp_filename, '-c:v', 'libx264', '-crf', '22',
			'-pix_fmt', 'yuv420p', '-c:a', 'aac', '-strict', 'experimental',
			video_filename
		]
		# Execute FFmpeg command
		subprocess.run(ffmpeg_command)
		os.remove(temp_filename)
	else:
		subprocess.call('mv %s %s'%(temp_filename,video_filename),shell=True)

	print("Video writing completed.")


def scale_crop_bounds(bounds, original_size, new_size):
	"""
	Scale bounding box coordinates from an original image size to a new size.

	Parameters:
	bounds (list): Bounding box coordinates [[top, bottom], [left, right]]
	original_size (int): Size of the original image (assuming square image)
	new_size (int): Size of the new image (assuming square image)

	Returns:
	list: Scaled bounding box coordinates
	"""
	t, b, l, r = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]

	# Calculate the scaling factor
	scale_factor = new_size / original_size

	# Scale the coordinates
	scaled_t = int(round(t * scale_factor))
	scaled_b = int(round(b * scale_factor))
	scaled_l = int(round(l * scale_factor))
	scaled_r = int(round(r * scale_factor))

	return [[scaled_t, scaled_b], [scaled_l, scaled_r]]


def split_tensor_into_chunks(tensor, b_max):
	b, c, h, w = tensor.shape
	num_chunks = (b + b_max - 1) // b_max  # Calculate the number of chunks needed
	return tensor.chunk(num_chunks, dim=0)  # Split along the batch dimension



def save_png_image_with_transparency(img, tr, filename, p=10, blur_sigma=2, crop=None, normalize=False):
	"""Saves an image with a transparency mask applied to the opacity channel."""
	img = torch_to_numpy(img)
	tr = torch_to_numpy(tr)
	
	assert len(img.shape) == 3

		
	# Normalize the image
	if normalize:
		img -= img.mean()
		img /= img.std()
		img -= img.min()
		img /= img.max()

	# Prepare the transparency mask
	tr = np.mean(np.array(tr).copy(), -1, keepdims=True)
	tr = clip_percentile(tr, p)
	tr = tr / tr.max()
	tr = gaussian_filter(tr.squeeze(), sigma=blur_sigma)
	tr = tr[:, :, np.newaxis]

	if crop is not None:  # Apply cropping if specified
		img = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
		tr = tr[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]

	# Combine the image and the transparency mask
	combined_img = np.concatenate([img, tr], axis=-1)

	# Save the image with transparency
	if filename[-4:] != '.png':
		filename = filename+'.png'
		#print('changing file name to '+filename)
	imageio.imwrite(filename, (combined_img * 255).astype(np.uint8))