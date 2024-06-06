import os
import numpy as np
import torch
import torchvision
from faccent.param_util import create_alpha_mask, center_paste
from torchvision.transforms import CenterCrop
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from faccent.transform import standard_box_transforms, standard_jitter_transforms, box_crop, uniform_gaussian_noise
from faccent.utils import default_model_input_size, safe_logit


from faccent.utils import img_to_img_tensor, default_img_size, default_model_input_size, default_model_input_range

TORCH_VERSION = torch.__version__

### COLORS ###

imagenet_color_correlation = torch.as_tensor(
	  [[0.56282854, 0.58447580, 0.58447580],
	   [0.19482528, 0.00000000,-0.19482528],
	   [0.04329450,-0.10823626, 0.06494176]]
).float()

imagenet_color_correlation_inv = torch.linalg.inv(imagenet_color_correlation)

def recorrelate_colors(images):
	images = rearrange(images, "b c h w -> b h w c")
	b, w, h, c = images.shape
	images_flat = rearrange(images, 'b h w c -> b (h w) c')
	images_flat = torch.matmul(images_flat, imagenet_color_correlation.to(images.device))
	images = rearrange(images_flat, 'b (h w) c -> b h w c', c=c, w=w, h=h)
	images = rearrange(images, "b h w c -> b c h w")
	return images

def decorrelate_colors(images):
	images = rearrange(images, "b c h w -> b h w c") #change channel position
	b, w, h, c = images.shape
	images_flat = rearrange(images, 'b h w c -> b (h w) c')
	images_flat = torch.matmul(images_flat, imagenet_color_correlation_inv.to(images.device))
	images = rearrange(images_flat, 'b (h w) c -> b h w c', c=c, w=w, h=h)
	images = rearrange(images, "b h w c -> b c h w") #change channel position
	return images

### Parameterizations ###
'''
'''


class fourier_phase():    # magnitude constrained

	def __init__(self, 
				 init_img = None, 
				 forward_init_img = False,
				 img_size = default_img_size,
				 device = None,
				 batch_size=1,
				 desaturation = 1.0,
				 seed=None,
				 normalize_img = None,
				 normalize_phase = None,
				 mag_spec_border = False,
				 use_mag_alpha = None,
				 copy_batch = False,
				 mag_alpha_init = 5.,
				 color_decorrelate = True,
				 corr_file_path='/'.join(__file__.split('/')[:-1])+"/clean_decorrelated.npy", 
				 name = 'fourier_phase'):
		
		self.name = name
		self.corr_file_path = corr_file_path
		self.img_size = img_size
		self.batch_size = batch_size
		self.desaturation = desaturation
		self.forward_init_img = forward_init_img
		self.mag_spec_border = mag_spec_border
		self.color_decorrelate = color_decorrelate
		copy_batch=copy_batch
	
		if normalize_img is None:
			if init_img is None: normalize_img = True
			else: normalize_img = False     
		if normalize_phase is None:
			if init_img is None: normalize_phase = True
			else: normalize_phase = False 
		if use_mag_alpha is None:
			if init_img is None: use_mag_alpha = False
			else: use_mag_alpha = True     
		self.normalize_img = normalize_img
		self.normalize_phase = normalize_phase
		self.use_mag_alpha = use_mag_alpha
		self.mag_alpha_init = mag_alpha_init
		self.seed = seed
		self.copy_batch = copy_batch


		if device is None:
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.device = device

		if init_img is None: self.init_img = init_img
		else: self.init_img = img_to_img_tensor(init_img,size=self.img_size).to(self.device)

		#initialize params
		if init_img is None:
			print('no image specified, so intializing with random phase, and magnitude from "clean_decorrelated.npy"')
			self.random_init()
			self.standard_transforms = [box_crop(
												box_min_size=0.05,
												box_max_size=0.5,
												box_loc_std=0.1,
												),
										uniform_gaussian_noise()
										]

		else:
			self.img_to_params(init_img) 
			self.standard_transforms = standard_box_transforms

	def random_init(self, img_size=None):

		if img_size is None:
			img_size = self.img_size

		buffer_shape = (img_size[0], 1 + (img_size[1] // 2))

		if self.seed is not None: np.random.seed(self.seed)
		# Add batch_size as first dimension
		phase = (
			torch.tensor(
				np.random.uniform(size=(self.batch_size, 3) + buffer_shape, low=-np.pi, high=np.pi),
				dtype=torch.float32,
			)
			.float()
			.to(self.device)
		)

		if self.copy_batch:
			single_phase = np.random.uniform(size=(3,) + buffer_shape, low=-np.pi, high=np.pi)
			phase = (
				torch.tensor(single_phase, dtype=torch.float32)
				.float()
				.unsqueeze(0)  # Add a batch dimension at the start
				.repeat(self.batch_size, 1, 1, 1)  # Repeat across the batch dimension
				.to(self.device)
			)
		
		# Load magnitude from a file for each item in the batch
		magnitude = (
			torch.stack([
				torch.tensor(
					np.load(self.corr_file_path),
					dtype=torch.float32,
				)
				.float()
				.to(self.device)
				for _ in range(self.batch_size)
			])
		)
		

		m_shape = magnitude.shape
		magnitude = F.interpolate(
								  magnitude, 
								  buffer_shape, 
								  mode="bilinear", 
								  antialias=True, 
								  align_corners=True,
		) * ((m_shape[-2] * m_shape[-1]) / (buffer_shape[0] * buffer_shape[1]))
		
		self.magnitude = magnitude 
		self.params = [phase]

		if self.use_mag_alpha:
			mag_alpha = self.mag_alpha_init*torch.ones(magnitude.shape).to(self.device)  #we use 6 because it yeilds ~1 after sigmoid is applied
			self.params.append(mag_alpha)


	def img_to_params(self,
					 img
					 ):
		"""convert the image in pixel space (0,1) (h w c) to polar coordinate frequency domain.

		Args:
			img (img tensor, pil image): the image in pixel space.

		Returns:
			magnitude (torch.tensor): the magnitude of the buffer.
			phase (torch.tensor): phase of the buffer.
		"""
		img = img_to_img_tensor(img,size=self.img_size).to(self.device)
		if self.mag_spec_border:
			self.random_init(img_size = (self.img_size[0]*4,self.img_size[1]*4))
			noise_img = self.params_to_img()
			img = center_paste(img, noise_img, blur_radius=int(self.img_size[0]/50))
			#img = center_paste(img, noise_img, blur_radius=int(self.img_size[0]/13))

		img = safe_logit(img)

		if self.color_decorrelate:
			img = decorrelate_colors(img)

		img *= self.desaturation

		# back in frequency domain
		buffer = torch.fft.rfft2(img)
		
		# extract magnitude and phase
		magnitude = torch.abs(buffer)
		phase = torch.atan2(buffer.imag, buffer.real)

		# normalize phase
		if self.normalize_phase:
			phase = phase * (torch.std(phase) + 1e-4)
			phase = phase + torch.mean(phase)


		#repeat for # batch
		phase = phase.repeat(self.batch_size,1,1,1)
		magnitude = magnitude.repeat(self.batch_size,1,1,1)

		self.magnitude = magnitude 
		self.params = [phase]

		if self.use_mag_alpha:
			mag_alpha = self.mag_alpha_init*torch.ones(magnitude.shape).to(self.device)  #we use 6 because it yeilds ~1 after sigmoid is applied
			self.params.append(mag_alpha)

		return self.params
	

	def params_to_img(self):
		"""convert the buffer in frequency domain to spatial domain.

		Args:
			magnitude (torch.tensor): the magnitude of the buffer.
			phase (torch.tensor): phase of the buffer.

		Returns:
			(torch.tensor): the image in pixel space.
		"""

		#input_range = get_input_range()


		##EDIT: This phase normalization was edited out!!!
		# normalize phase
		#phase = self.params - torch.mean(self.params, dim=(1,2,3), keepdim=True)
		#phase = phase / (torch.std(phase, dim=(1,2,3), keepdim=True) + 1e-4)

		phase= self.params[0]
		magnitude = self.magnitude
		if self.use_mag_alpha: 
			magnitude = magnitude * torch.sigmoid(self.params[1])
		# build complex buffer
		buffer = torch.complex(torch.cos(phase) * magnitude, torch.sin(phase) * magnitude)
		
		# back in pixel domain
		img = torch.fft.irfft2(buffer)
		#img = torch.fft.irfftn(buffer, s=self.img_size, norm='ortho')

		img = img / self.desaturation

		# recorrelation require the data to be normalized
		# if self.normalize_img:
		#     img = img - torch.mean(img, dim=(1,2,3), keepdim=True)
		#     img = img / (torch.std(img, dim=(1,2,3), keepdim=True) + 1e-4)
		if self.color_decorrelate:
			img = recorrelate_colors(img)

		if self.normalize_img:
			img = img - torch.mean(img, dim=(1,2,3), keepdim=True)
			img = img / (torch.std(img, dim=(1,2,3), keepdim=True) + 1e-4)

		img = torch.sigmoid(img) 

		if self.mag_spec_border:
			crop = CenterCrop(self.img_size)
			img = crop(img)

		return img
	
	def __call__(self):
		if self.forward_init_img:
			img_f = lambda: torch.concatenate([self.params_to_img(),self.init_img],dim=0)
		else:
			img_f = self.params_to_img
		return self.params, img_f

class pixel():

	def __init__(self, 
				 init_img = None,
				 forward_init_img = False,
				 img_size = default_img_size,
				 sd=.05,
				 batch_size = 1,
				 desaturation = 5.0,
				 device = None,
				 seed = None,
				 normalize_img = False,
				 color_decorrelate = True,
				 name = 'pixel'):
		
		self.sd  = sd
		self.name = name
		self.img_size = img_size
		self.batch_size = batch_size
		self.forward_init_img = forward_init_img
		self.desaturation = desaturation
		self.normalize_img = normalize_img
		self.color_decorrelate = color_decorrelate
		self.seed = seed

		if device is None:
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.device = device

		self.standard_transforms = standard_box_transforms

		if init_img is None: self.init_img = init_img
		else: self.init_img = img_to_img_tensor(init_img,size=self.img_size).to(self.device)

		#initialize params
		if init_img is None:
			print('no image specified, so intializing with random noise"')
			self.random_init()
		else:
			self.img_to_params(init_img)


	def random_init(self):
		"""
		initialize the params as a (c h w) gaussian noise image
		"""

		if self.seed is not None: 
			torch.manual_seed(self.seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed_all(self.seed)

		self.params = (torch.randn(self.batch_size,3,self.img_size[0],self.img_size[1]) * self.sd).to(self.device)



	def img_to_params(self,
					 img
					 ):
		"""
		takes a color correlated rbg image and returns decorrelated params img
		"""
		img = img_to_img_tensor(img,size=self.img_size).to(self.device)

		img = safe_logit(img)

		if self.color_decorrelate:
			img = decorrelate_colors(img)

		img *= self.desaturation

		#repeat for batch #
		img = img.repeat(self.batch_size,1,1,1)
		
		self.params = img

		return self.params
	

	def params_to_img(self):
		"""
		takes a parameterized rgb img tensor, recorrelates the colors, and adds a batch dimension
		"""
		img = self.params
		img = img/self.desaturation

		# recorrelation require the data to be normalized
		if self.normalize_img:
			img = img - torch.mean(img, dim=(1,2,3), keepdim=True)
			img = img / (torch.std(img, dim=(1,2,3), keepdim=True) + 1e-4)
		
		if self.color_decorrelate:
			img = recorrelate_colors(img)
		# remap values to input domain
		img = torch.sigmoid(img) 

		return img
	
	def __call__(self):
		if self.forward_init_img:
			img_f = lambda: torch.concatenate([self.params_to_img(),self.init_img],dim=0)
		else:
			img_f = self.params_to_img
		return [self.params], img_f



class fourier():

	def __init__(self, 
				 init_img = None,
				 forward_init_img = False, 
				 img_size = default_img_size,
				 sd=.01,
				 decay_power=1,
				 batch_size = 1,
				 device = None,
				 seed=None,
				 normalize_img = False,
				 color_decorrelate = True,
				 desaturation = 5.0, #magic constant from the lucid library
				 name = 'fourier'):
		
		self.sd  = sd
		self.decay_power = decay_power
		self.name = name
		self.img_size = img_size
		self.desaturation = desaturation
		self.batch_size = batch_size
		self.forward_init_img = forward_init_img
		self.normalize_img = normalize_img
		self.color_decorrelate = color_decorrelate
		self.seed = seed

		if device is None:
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.device = device

		self.freqs = self.rfft2d_freqs()
		scale = 1.0 / np.maximum(self.freqs, 1.0 / max(self.img_size[0], self.img_size[1])) ** self.decay_power
		self.scale = torch.tensor(scale).float()[None, None, ..., None].to(self.device)

		if init_img is None: self.init_img = init_img
		else: self.init_img = img_to_img_tensor(init_img,size=self.img_size).to(self.device)

		self.standard_transforms = standard_box_transforms

		#initialize params
		if init_img is None:
			print('no image specified, so intializing with random noise"')
			self.random_init()
		else:
			self.img_to_params(init_img)


	def random_init(self):
		"""
		initialize the params as a fourier spectrum with predefined magnitude spectrum
		"""

		if self.seed is not None: 
			torch.manual_seed(self.seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed_all(self.seed)

		init_val_size = (self.batch_size, 3) + self.freqs.shape + (2,) # 2 for imaginary and real components
		self.params = (torch.randn(*init_val_size) * self.sd).to(self.device)



	def img_to_params(self,
					 img
					 ):
		"""
		takes a color correlated rbg image and returns decorrelated params img
		"""
		img = img_to_img_tensor(img,size=self.img_size).to(self.device)
		img = safe_logit(img)

		if self.color_decorrelate:
			img = decorrelate_colors(img)

		img *= self.desaturation

		if TORCH_VERSION >= "1.7.0":
			from torch import fft
			fourier_image = fft.rfftn(img, norm='ortho',s=self.img_size)
			fourier_image = torch.view_as_real(fourier_image)
		else:
			fourier_image = torch.fft.rfftn(img, 2, normalized=True, signal_sizes=self.img_size)
		fourier_image = fourier_image/self.scale
		#repeat for # batch
		fourier_image = fourier_image.repeat(self.batch_size,1,1,1,1)

		self.params = fourier_image

		return self.params


	def params_to_img(self):
		"""
		takes a fourier parameterized image and returns rgb image
		"""

		params = self.params*self.scale
		if TORCH_VERSION >= "1.7.0":
			if type(params) is not torch.complex64:
				params = torch.view_as_complex(params)
			img = torch.fft.irfftn(params, s=self.img_size, norm='ortho')
		else:
			img = torch.irfft(params, 2, normalized=True, signal_sizes=self.img_size)
		img = img[:self.batch_size, :3, :self.img_size[0], :self.img_size[1]]

		img = img / self.desaturation

		# recorrelation require the data to be normalized
		if self.normalize_img:
			img = img - torch.mean(img, dim=(1,2,3), keepdim=True)
			img = img / (torch.std(img, dim=(1,2,3), keepdim=True) + 1e-4)

		if self.color_decorrelate:
			img = recorrelate_colors(img)

		img = torch.sigmoid(img) 
		
		return img


	# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
	def rfft2d_freqs(self):
		"""Computes 2D spectrum frequencies."""
		h,w = self.img_size[0],self.img_size[1]
		fy = np.fft.fftfreq(h)[:, None]
		# when we have an odd input dimension we need to keep one additional
		# frequency and later cut off 1 pixel
		if w % 2 == 1:
			fx = np.fft.fftfreq(w)[: w // 2 + 2]
		else:
			fx = np.fft.fftfreq(w)[: w // 2 + 1]
		return np.sqrt(fx * fx + fy * fy)

	def __call__(self):
		if self.forward_init_img:
			img_f = lambda: torch.concatenate([self.params_to_img(),self.init_img],dim=0)
		else:
			img_f = self.params_to_img
		return [self.params], img_f
	
