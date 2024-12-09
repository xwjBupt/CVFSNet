from __future__ import division
from configparser import Interpolation
import os
from abc import ABC
from typing import List, Tuple, Union, Literal
import SimpleITK as sitk
import cv2
import numpy as np
import torch
from skimage.filters import threshold_multiotsu
import torch
import torch.nn.functional as F
import random
import torchvision.transforms.functional_tensor as F_t
import math
import numpy as np
import numbers
import collections
import cv2
import torch
import warnings
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as T
import numpy as np
import cv2
import itertools
import operator
from skimage.filters import threshold_multiotsu
import torchio as tio


global_mean_3c = 160.789445663568
global_std_3c = 44.26357490097961


class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        for t in self.transform:
            image = t(image)
        return image


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = (image - global_mean_3c) / (global_std_3c + 1e-7)
        image = image / 255.0
        return image  # CDHW


class Fix_Normalize(object):
    def __init__(self, mean=global_mean_3c, std=global_std_3c):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / (self.std + 1e-7)
        image = image / 255.0
        return image  # CDHW


class Resize(object):
    def __init__(self, t=32, visual=(512, 512)):
        self.t = t
        self.visual = visual

    def __call__(self, image):
        # fix mean and std of 3c the whole dataset
        image = image.unsqueeze(0)
        image = F.interpolate(
            image,
            (self.t, self.visual[0], self.visual[1]),
            mode="trilinear",
            align_corners=True,
        ).contiguous()[0]
        return image  # CDHW


class clip_image(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            # visual norm
            c = image.shape[0]
            gray_int8_img = np.transpose(image[0].int().numpy(), (1, 2, 0))  # H,W,D
            boundaries = threshold_multiotsu(gray_int8_img, classes=3)
            gray_int8_img = np.clip(gray_int8_img, boundaries[0], boundaries[-1])
            gray_int8_img = np.transpose(gray_int8_img, (2, 0, 1))[np.newaxis, ...]
            if c == 3:
                image = torch.cat(
                    [
                        torch.tensor(gray_int8_img, dtype=torch.float32),
                        torch.tensor(gray_int8_img, dtype=torch.float32),
                        torch.tensor(gray_int8_img, dtype=torch.float32),
                    ],
                    dim=0,
                )
                return image
            else:
                return torch.tensor(gray_int8_img, dtype=torch.float32)  # CDHW
        else:
            return image


class RandomRotation(object):
    def __init__(self, degrees=(-10, 10), fill=255, p=0.5):
        self.p = p
        self.trans = T.RandomRotation(
            degrees=degrees, interpolation=T.InterpolationMode.BILINEAR, fill=fill
        )

    def __call__(self, image):
        if random.random() < self.p:
            image = self.trans(image)
        return image  # CDHW


class RandomErode(object):
    def __init__(self, k=5, high=192, low=48, p=0.15, **kwargs):
        self.p = p
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self.low = low
        self.high = high

    def __call__(self, image):
        if random.random() < self.p:
            image = image.numpy()
            # image = np.where(image > self.high, self.high, image)
            # image = np.where(image < self.low, self.low, image)
            for i in range(image.shape[1]):
                image[:, i, ...] = cv2.erode(image[:, i, ...], self.kernel)
            image = torch.tensor(image)
        return image  # CDHW


class RandomDilate(object):
    def __init__(self, k=5, high=192, low=48, p=0.15, **kwargs):
        self.p = p
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self.low = low
        self.high = high

    def __call__(self, image):
        if random.random() < self.p:
            image = image.numpy()
            for i in range(image.shape[1]):
                image[:, i, ...] = cv2.dilate(image[:, i, ...], self.kernel)
            image = torch.tensor(image)
        return image  # CDHW


class TioClamp(object):
    def __init__(self, clamps=(48, 192), p=1):
        self.p = p
        self.trans = tio.Clamp(clamps[0], clamps[1])

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.trans = tio.RandomFlip(axes=("A",), flip_probability=p)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomAnisotropy(object):
    def __init__(self, axes=(0, 1, 2), downsampling=(5, 10), scalars_only=False, p=0.5):
        self.p = p
        self.trans = tio.RandomAnisotropy(
            axes=axes, downsampling=downsampling, scalars_only=scalars_only
        )

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomMotion(object):
    def __init__(
        self,
        degrees=60,
        translation=20,
        num_transforms=10,
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomMotion(
            degrees=degrees, translation=translation, num_transforms=num_transforms
        )

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomGhosting(object):
    def __init__(
        self,
        num_ghosts=(3, 5),
        axes=(0, 1, 2),
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomGhosting(num_ghosts=num_ghosts, axes=axes)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomSpike(object):
    def __init__(
        self,
        num_spikes=3,
        intensity=(1, 2),
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomBiasField(object):
    def __init__(
        self,
        coefficients=0.3,
        order=2,
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomBiasField(coefficients=coefficients, order=order)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomBlur(object):
    def __init__(
        self,
        std=(0, 3),
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomBlur(std=std)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomNoise(object):
    def __init__(
        self,
        std=(0, 0.5),
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomNoise(std=std)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomSwap(object):
    def __init__(
        self,
        patch_size=30,
        num_iterations=30,
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomSwap(
            patch_size=patch_size, num_iterations=num_iterations
        )

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioRandomGamma(object):
    def __init__(
        self,
        log_gamma=(-0.5, 0.5),
        p=0.5,
    ):
        self.p = p
        self.trans = tio.RandomGamma(log_gamma=log_gamma)

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            image = self.trans(image)
            image = image.permute(0, 3, 1, 2)
        return image  # CDHW


class TioZNormalization(object):
    def __init__(
        self,
        p=1,
        div255=False,
    ):
        self.p = p
        self.div255 = div255
        self.trans = tio.ZNormalization()

    def __call__(self, image):  # CDHW
        if random.random() < self.p:
            image = image.permute(0, 2, 3, 1)
            try:
                image = self.trans(image)
            except Exception as e:
                image = (image - global_mean_3c) / (global_std_3c + 1e-7)
            image = image.permute(0, 3, 1, 2)
            if self.div255:
                image = image / 255.0
        return image  # CDHW


class RandomErase(object):
    def __init__(
        self,
        radius_ratio=0.25,
        erase_type="circle",
        p=0.5,
        erase_frame_ratio=0.25,
        throughout=True,
    ):
        self.p = p
        assert erase_type in [
            "circle",
            "rectangle",
            "ellipse",
            "mix",
        ], 'erase_type not in ["circle", "rectangle", "ellipse", "mix"]'
        self.erase_type = erase_type
        self.radius_ratio = radius_ratio
        self.erase_frame_ratio = erase_frame_ratio
        self.throughout = throughout

    def __call__(self, image):
        if random.random() < self.p:
            channel, depth, H, W = image.shape
            if self.throughout:
                erase_frames = list(range(0, depth))
            else:
                erase_frames = random.sample(
                    range(0, depth), int(depth * self.erase_frame_ratio)
                )
            self.radius = int(min(H, W) * self.radius_ratio)
            for erase_frame in erase_frames:
                raw_frame = image[:, erase_frame, ...].numpy()
                raw_frame = np.transpose(raw_frame, (1, 2, 0))
                raw_frame = raw_frame.astype(np.uint8)
                pad_value = int(np.median(raw_frame))
                if channel == 3:
                    pad_value = (pad_value, pad_value, pad_value)
                center = (
                    random.randint(self.radius + 1, H - self.radius - 1),
                    random.randint(self.radius + 1, W - self.radius - 1),
                )
                if self.erase_type == "mix":
                    type_index = random.randint(0, 2)
                    if type_index == 0:
                        type_index = "circle"
                    elif type_index == 1:
                        type_index = "rectangle"
                    else:
                        type_index = "ellipse"
                else:
                    type_index = self.erase_type

                if type_index == "circle":
                    erased_frame = cv2.circle(
                        raw_frame.copy(),
                        center,
                        self.radius,
                        color=pad_value,
                        thickness=-1,
                    )
                elif type_index == "rectangle":
                    p1 = (center[0] - self.radius, center[1] - self.radius)
                    p2 = (center[0] + self.radius, center[1] + self.radius)
                    erased_frame = cv2.rectangle(
                        raw_frame.copy(),
                        p1,
                        p2,
                        color=pad_value,
                        thickness=-1,
                    )
                else:
                    erased_frame = cv2.ellipse(
                        raw_frame.copy(),
                        center,
                        (self.radius, self.radius // 2),
                        0,
                        0,
                        360,
                        pad_value,
                        thickness=-1,
                    )
                erased_frame = np.transpose(erased_frame, (2, 0, 1))
                image[:, erase_frame, ...] = torch.tensor(
                    erased_frame, dtype=torch.float32
                )
        return image  # CDHW


class GaussianBlur(object):
    def __init__(self, kernel_size=(3, 3), p=0.5, **kwargs):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = T.functional.gaussian_blur(image, kernel_size=self.kernel_size)
        return image  # CDHW


class Crop(object):
    def __init__(self, crop=(0.2, 0.2, 0.2, 0.2), p=0.5, **kwargs):
        self.crop = crop
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            y_size, x_size = image.shape[2:]
            pos_x_left = int(x_size * random.uniform(0, self.crop[0]))
            pos_x_right = x_size - int(x_size * random.uniform(0, self.crop[1]))
            pos_y_bottom = int(y_size * random.uniform(0, self.crop[2]))
            pos_y_top = y_size - int(y_size * random.uniform(0, self.crop[3]))
            pos_y_bottom = min(pos_y_bottom, pos_y_top)
            pos_y_top = max(pos_y_bottom, pos_y_top)
            pos_x_left = min(pos_x_left, pos_x_right)
            pos_x_right = max(pos_x_left, pos_x_right)
            return image[..., pos_y_bottom:pos_y_top, pos_x_left:pos_x_right]
        else:
            return image


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = torch.tensor(image).float()
        return image  # CDHW


class Color_Jitter(object):
    def __init__(
        self,
        brightness=[0.85, 1.25],
        contrast=[0.85, 1.25],
        saturation=[0.85, 1.25],
        hue=[-0.15, 0.15],
        p=0.5,
        **kwargs,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        self.color_jitter = T.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue
        )

    def __call__(self, image):
        if random.random() < self.p:
            image = image.transpose(1, 0)
            image = self.color_jitter(image)
            image = image.transpose(1, 0)
        return image


class Autocontrast(object):
    def __init__(self, p):
        self.autocontraster = T.RandomAutocontrast()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.tensor(image.transpose(2, 0, 1))
            image = self.autocontraster(image)
            image = image.numpy().transpose(1, 2, 0)
        return image, label


class Sharpness(object):
    def __init__(self, sharpness_factor=3, p=0.25):
        self.sharpness_adjuster = T.RandomAdjustSharpness(
            sharpness_factor=sharpness_factor, p=1
        )
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.tensor(image.transpose(2, 0, 1))
            image = self.sharpness_adjuster(image)
            image = image.numpy().transpose(1, 2, 0) * 255
        return image, label


class Invert(object):
    def __init__(self, p=0.25):
        self.invert_adjuster = T.RandomInvert()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.uint8)
            image = self.invert_adjuster(image)
            image = image.numpy().transpose(1, 2, 0).astype(np.float32)
        return image, label


class RandomPosterize(object):
    def __init__(self, bits=2, p=0.25):
        self.pos_adjuster = T.RandomPosterize(bits)
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.uint8)
            image = self.pos_adjuster(image)
            image = image.numpy().transpose(1, 2, 0).astype(np.float32)
        return image, label


class RandomSolarize(object):
    def __init__(self, threshold=3, p=0.25):
        self.sola_adjuster = T.RandomSolarize(threshold=threshold)
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.uint8)
            image = self.sola_adjuster(image)
            image = image.numpy().transpose(1, 2, 0).astype(np.float32)
        return image, label


class RandomEqualize(object):
    def __init__(self, p=0.25):
        self.equ_adjuster = T.RandomEqualize()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.uint8)
            image = self.equ_adjuster(image)
            image = image.numpy().transpose(1, 2, 0).astype(np.float32)
        return image, label


class Translate_x_y(object):
    """
    Translate the video along the vertical axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much (relative to the image size) to translate along the
            vertical axis.
    """

    def __init__(self, x_size=0.3, y_size=0.3, p=0.5):
        self.x_size = x_size
        self.y_size = y_size
        self.p = p

    def __call__(self, image):
        image = image.transpose(1, 0)
        if random.random() < self.p:
            if self.x_size > 0:
                translation_offset = self.x_size * image.size(-1)
                image = F_t.affine(
                    image,
                    [1, 0, translation_offset, 0, 1, 0],
                    fill=(0.5, 0.5, 0.5),
                    interpolation="bilinear",
                )
        if random.random() > self.p:
            if self.y_size > 0:
                translation_offset = self.x_size * image.size(-2)
                image = F_t.affine(
                    image,
                    [1, 0, 0, 0, 1, translation_offset],
                    fill=(0.5, 0.5, 0.5),
                    interpolation="bilinear",
                )
        image = image.transpose(1, 0)
        return image


def _translate_x(video: torch.Tensor, factor: float, **kwargs):
    """
    Translate the video along the vertical axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much (relative to the image size) to translate along the
            vertical axis.
    """

    translation_offset = factor * video.size(-1)
    return F_t.affine(
        video,
        [1, 0, translation_offset, 0, 1, 0],
        fill=kwargs["fill"],
        interpolation="bilinear",
    )


def _translate_y(video: torch.Tensor, factor: float, **kwargs):
    """
    Translate the video along the vertical axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much (relative to the image size) to translate along the
            horizontal axis.
    """

    translation_offset = factor * video.size(-2)
    return F_t.affine(
        video,
        [1, 0, 0, 0, 1, translation_offset],
        fill=kwargs["fill"],
        interpolation="bilinear",
    )


def get_ovesample_images(
    file_lists,
    fuse01=False,
    top_k=0.4,
    repeat_thr=1.15,
):
    # 1. For each category c, compute the fraction of images
    #    that contain it: f(c)
    num_images = len(file_lists)
    tmp_list = []
    category_repeat = {}
    class_file_dict = {}
    class_ratio_dict = {}
    class_ratio_dict_before_oversample = {}

    for file_list in file_lists:
        single_label = file_list.split("_")[3]
        if fuse01 and (single_label == "T0" or single_label == "T1"):
            single_label = "T0/1"
        if single_label in class_ratio_dict.keys():
            class_ratio_dict[single_label] += 1
        else:
            class_ratio_dict[single_label] = 1

        if single_label in class_file_dict.keys():
            class_file_dict[single_label].append(file_list)
        else:
            class_file_dict[single_label] = [file_list]

    for k, v in class_ratio_dict.items():
        tmp_list.append(v / num_images)
        class_ratio_dict[k] = v / num_images
        class_ratio_dict_before_oversample[k] = [v, v / num_images]
    print(
        " Before oversample, all {} samples, ratio at {} ".format(
            num_images, class_ratio_dict_before_oversample
        )
    )

    class_ratio_dict = dict(
        sorted(class_ratio_dict.items(), key=operator.itemgetter(1))
    )
    num_classes = len(class_ratio_dict.keys())
    top_k = int(top_k * num_classes)
    max_ratio = max(tmp_list)

    for index, (k, v) in enumerate(class_ratio_dict.items()):
        if index >= top_k:
            category_repeat[k] = 0
        else:
            category_repeat[k] = max(
                math.sqrt(1 / max_ratio),
                math.log(repeat_thr * max_ratio / class_ratio_dict[k]),
            )

    images_after_oversample = []
    class_ratio_dict_after_oversample = {}
    for k, v in class_file_dict.items():
        if category_repeat[k] > 0:
            toadd_num = int(len(class_file_dict[k]) * category_repeat[k])
            tmp_list = toadd_num * class_file_dict[k]
            random.shuffle(tmp_list)
            class_file_dict[k] = class_file_dict[k] + tmp_list[:toadd_num]
        images_after_oversample += class_file_dict[k]
    for k, v in class_file_dict.items():
        class_ratio_dict_after_oversample[k] = [
            len(class_file_dict[k]),
            len(class_file_dict[k]) / len(images_after_oversample),
        ]

    print(
        " After oversample, all {} samples, ratio at {} ".format(
            len(images_after_oversample), class_ratio_dict_after_oversample
        )
    )
    return images_after_oversample
