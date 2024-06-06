from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchvision.transforms import Normalize
from faccent.utils import default_model_input_size, default_model_input_range
from concurrent.futures import ThreadPoolExecutor
import kornia
from kornia.geometry.transform import translate
KORNIA_VERSION = kornia.__version__


def jitter(d):
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)

    def inner(image_t):
        dx = np.random.choice(d)
        dy = np.random.choice(d)
        return translate(image_t, torch.tensor([[dx, dy]]).float().to(image_t.device))

    return inner


def pad(w, mode="reflect", constant_value=0.5):
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(image_t, [w] * 4, mode=mode, value=constant_value,)

    return inner


def random_scale(scales):
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = [_roundup(scale * d) for d in shp]
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2)

    return inner


def random_rotate(angles, units="degrees"):
    def inner(image_t):
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = torch.ones(b) * alpha
        if KORNIA_VERSION < '0.4.0':
            scale = torch.ones(b)
        else:
            scale = torch.ones(b, 2)
        center = torch.ones(b, 2)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = kornia.get_rotation_matrix2d(center, angle, scale).to(image_t.device)
        rotated_image = kornia.warp_affine(image_t.float(), M, dsize=(h, w))
        return rotated_image

    return inner


def box_crop(
    box_min_size=0.85,
    box_max_size=0.95,
    box_loc_std=0.05,
):
    """returns a function that will take an image and randomly crop it into a batch.

    Args:
        box_min_size (float, optional): minimum box size (as fraction of canevas size). Defaults to 0.10.
        box_max_size (float, optional): minimum box size (as fraction of canevas size). Defaults to 0.35.
        box_loc_std (float, optional): std of box positions (sampled normally around image center). Defaults to 0.2.
        noise_std (float, optional): std of the noise added to the image. Defaults to 0.05.
        model_input_size (tuple, optional): once the crop is made it is interpolated to this size

    Returns:
        (callable): a function that takes an image tensor of shape (b,c,h,w)
                    and returns a batch tensor of shape (b+nb_crops-1,c,h,w).
                    This function operates on the last image in the batch,
                    replacing it with cropped resized versions.
    """

    def inner(image_t):
        """
        Args: 
            x (model input tensor): 
            batch_idx
        Returns:
            batch tensor of shape (b,c,h,w)
        """
        #ensure image formatting is correct
        #x = image_to_tensor(x)

        x = image_t
        im_b, im_c, im_w, im_h = x.shape

        # sample box size uniformely in [min_size, max_size]
        delta_x = torch.rand(1) * (box_max_size - box_min_size) + box_min_size
        delta_y = delta_x
        # sample box x0,y0 in [0,1-box_size/img_size]
        # here we sample around normally around image center,
        # but uniform sampling works also
        x0 = torch.clamp(
            torch.randn(1) * box_loc_std + 0.5, delta_x / 2.0, 1 - delta_x / 2.0
        )
        y0 = torch.clamp(
            torch.randn(1) * box_loc_std + 0.5, delta_y / 2.0, 1 - delta_y / 2.0
        )

        #crop image
        x = x[
                :, 
                :, 
                int((x0 - delta_x / 2.0) * im_w): int((x0 + delta_x / 2.0) * im_w), 
                int((y0 - delta_y / 2.0) * im_h): int((y0 + delta_y / 2.0) * im_h)
             ]

        return x

    return inner


def box_crop_2(
    box_min_size=0.05,
    box_max_size=0.99
):
    """returns a function that will take an image and randomly crop it into a batch.

    Args:
        box_min_size (float, optional): minimum box size (as fraction of canvas size). 
        box_max_size (float, optional): maximum box size (as fraction of canvas size). 

    Returns:
        (callable): a function that takes an image tensor of shape (b,c,h,w)
                    and returns a batch tensor of shape (b,c,h,w).
    """
    def inner(image_t):
        x = image_t
        im_b, im_c, im_w, im_h = x.shape

        # Sample box size uniformly in [min_size, max_size]
        delta_x = torch.rand(1) * (box_max_size - box_min_size) + box_min_size
        delta_y = delta_x

        # Sample top-left corner x0, y0 uniformly from 'in bounds' regions
        max_x0 = 1 - delta_x
        max_y0 = 1 - delta_y
        x0 = torch.rand(1) * max_x0
        y0 = torch.rand(1) * max_y0

        # Crop image
        x = x[
            :, 
            :, 
            int(x0 * im_w): int((x0 + delta_x) * im_w), 
            int(y0 * im_h): int((y0 + delta_y) * im_h)
        ]

        return x

    return inner


def uniform_gaussian_noise(noise_std=0.02):

    def inner(image_t):
        batch_size = image_t.shape[0]
        normal_noise = torch.normal(torch.zeros_like(image_t[0]), std=noise_std).to(image_t.device)
        normal_noise = normal_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        uniform_noise = (torch.rand_like(image_t[0], dtype=torch.float32).to(image_t.device) - 0.5) * noise_std
        uniform_noise = uniform_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        image_t = image_t+normal_noise+uniform_noise

        return image_t
    
    return inner

def color_noise(noise_std=.02):
    def inner(image_t):
        b,c,h,w = image_t.shape
        normal_noise = torch.normal(torch.tensor([0.,0.,0.]), std=noise_std).to(image_t.device)
        normal_noise = normal_noise.view(1,3,1,1).repeat(b, 1, h, w)
        uniform_noise = (torch.rand_like(torch.tensor([0.,0.,0.]), dtype=torch.float32).to(image_t.device) - 0.5) * noise_std
        uniform_noise = uniform_noise.view(1,3,1,1).repeat(b, 1, h, w)
        image_t = image_t+normal_noise+uniform_noise

        return image_t
    
    return inner


##EDIT transformations are not computed in parallel :(
def compose(transforms,nb_transforms=1):
    def inner(x):
        #compose transforms
        def composer(x):
            for transform in transforms:
                x = transform(x)
            return x
        #batch transforms
        if nb_transforms > 1:
            #with ThreadPoolExecutor() as executor:
            #    ys = list(executor.map(composer, [x]*nb_transforms))
            ys = [composer(x) for _ in range(nb_transforms)]
            return torch.concatenate(ys,dim=0) 
        else:
            return composer(x)

    return inner


def _roundup(value):
    return np.ceil(value).astype(int)


def _rads2angle(angle, units):
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle


def imagenet_normalize():
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def inceptionv1_normalize():
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117

def range_normalize(img_range = default_model_input_range):
    def inner(image_t):
        return image_t*(img_range[1] - img_range[0]) + img_range[0]
    return inner

def resize(img_size=default_model_input_size):
    def inner(image_t):
        return F.interpolate(image_t, 
                             size=img_size,
                             mode="bilinear",
                             antialias=True,
                             align_corners=True)
    return inner


standard_jitter_transforms = [
    pad(12, mode="constant", constant_value=0.0),
    jitter(8),
    random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]


standard_box_transforms = [
                           box_crop_2(),
                           uniform_gaussian_noise()
                          ]



def fast_batch_crop(box_min_size=0.05,
                    box_max_size=0.99,
                    noise_std=0.02,
                    img_size=default_model_input_size,
                    img_range = default_model_input_range,
                    nb_crops = 20):

    def inner(x):

        im_b, im_c, im_w, im_h = x.shape

        delta_x = torch.rand(nb_crops) * (box_max_size - box_min_size) + box_min_size
        delta_y = delta_x

        # # Sample top-left corner x0, y0 uniformly from 'in bounds' regions
        max_x0 = 1 - delta_x
        max_y0 = 1 - delta_y
        x0 = torch.rand(nb_crops) * max_x0
        y0 = torch.rand(nb_crops) * max_y0

        # # build boxes
        boxes = torch.stack(
            [
                x0* im_w,
                y0* im_h,
                (x0 + delta_x)* im_w,
                (y0 + delta_y)* im_h,
            ],
            -1,
        )

        boxes = boxes.int()

        #resize
        x = torch.cat(
            [
                F.interpolate(
                    image[None, :, x0:x1, y0:y1],
                    img_size,
                    mode="bilinear",
                    antialias=True,
                    align_corners=True,
                )
                for x0, y0, x1, y1 in boxes  # Inner loop for each box
                for image in x   # Outer loop for each image in batch
            ],
            0,
        )


        #import pdb; pdb.set_trace()

        # Generate normal noise for one image in the batch and repeat for the 3 crops
        normal_noise = torch.normal(torch.zeros(nb_crops, x.shape[1], x.shape[2], x.shape[3]), std=noise_std).to(x.device)
        normal_noise = torch.repeat_interleave(normal_noise,im_b,dim=0)

        # Generate uniform noise for one image in the batch and repeat for the 3 crops
        uniform_noise = ((torch.rand(nb_crops, x.shape[1], x.shape[2], x.shape[3], dtype=torch.float32).to(x.device) - 0.5) * noise_std)
        uniform_noise = torch.repeat_interleave(uniform_noise,im_b,dim=0)

        # Apply the noise to the tensor y
        x += normal_noise
        x += uniform_noise


        x = x*(img_range[1] - img_range[0]) + img_range[0]

        return x

    return inner


