import random
from PIL import Image
import numpy as np
import numbers
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import Compose, Lambda
import torch
import cv2


def rgb_to_gray(img):
    gray_img = np.empty((1, img.shape[1], img.shape[2], img.shape[3]), np.dtype('float32'))
    for n in range(img.shape[1]):
        trans_img = img[:, n]
        trans_img = np.transpose(trans_img, (1, 2, 0))
        trans_img = Image.fromarray(np.uint8(trans_img))
        trans_img = trans_img.convert('L')
        trans_img = np.expand_dims(np.asarray(trans_img), axis=2)
        trans_img = np.transpose(trans_img, (2, 0, 1))
        gray_img[:, n, :, :] = torch.from_numpy(trans_img.copy())

    return gray_img


def histeq(img):
    gray_img = np.empty((1, img.shape[1], img.shape[2], img.shape[3]), np.dtype('float32'))
    for n in range(img.shape[1]):
        trans_img = img[:, n]
        imhist, bins = np.histogram(trans_img.flatten(), 256)
        cdf = imhist.cumsum()
        cdf = 255.0 * cdf / cdf[-1]
        im = np.interp(trans_img.flatten(), bins[:-1], cdf)
        trans_img = im.reshape(trans_img.shape)
        gray_img[:, n, :, :] = torch.from_numpy(trans_img)

    return gray_img


def normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    channel_len = image.shape[0]

    for i in range(channel_len):
        image[i] = (image[i] - mean[i]) / std[i]

    return image


def to_tensor(img):
    img = torch.from_numpy(img) / 255
    return img


class ToTensor(object):
    """
    img dtype: float32
         type: <class 'numpy.ndarray'>
        (Batch x C x N x H x W)
    """

    def __call__(self, img):
        img = torch.from_numpy(img) / 255
        # img = (img + 1) / 2

        return img


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given numpy.ndarray
    (Batch x C x N x H x W) with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            for n in range(img.shape[1]):
                trans_img = img[:, n]
                trans_img = np.transpose(trans_img, (1, 2, 0))
                trans_img = Image.fromarray(np.uint8(trans_img))
                trans_img = F.hflip(trans_img)
                trans_img = np.transpose(trans_img, (2, 0, 1))
                img[:, n, :, :] = torch.from_numpy(trans_img.copy())

        return img


class CenterCrop(object):
    """
        Crops the given image pack at the center.
    """

    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, img):
        for n in range(img.shape[1]):
            trans_img = img[:, n]
            trans_img = np.transpose(trans_img, (1, 2, 0))
            trans_img = Image.fromarray(np.uint8(trans_img))
            trans_img = F.center_crop(trans_img, self.size)
            trans_img = np.transpose(trans_img, (2, 0, 1))
            img[:, n, :, :] = torch.from_numpy(trans_img.copy())
        return img


class Resize(object):
    """Resize the input image pack to the given size.

    """

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        for n in range(img.shape[1]):
            trans_img = img[:, n]
            trans_img = np.transpose(trans_img, (1, 2, 0))
            trans_img = Image.fromarray(np.uint8(trans_img))
            trans_img = F.resize(trans_img, self.size, self.interpolation)
            trans_img = np.transpose(trans_img, (2, 0, 1))
            img[:, n, :, :] = torch.from_numpy(trans_img.copy())
        return img


class Noise(object):
    """ Add Gaussian noise
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        for n in range(img.shape[1]):
            trans_img = img[:, n]
            shape = trans_img.shape
            gauss = np.random.normal(0, self.sigma, shape)
            noisy_img = trans_img + gauss
            noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
            img[:, n, :, :] = torch.from_numpy(noisy_img.copy())
        return img


class Blurring(object):
    """ Using Gaussian blurring
    """

    def __init__(self, size):
        self.size = int(size)

    def __call__(self, img):
        for n in range(img.shape[1]):
            trans_img = img[:, n]
            trans_img = cv2.GaussianBlur(trans_img, (self.size, self.size), 0)
            img[:, n, :, :] = torch.from_numpy(trans_img.copy())
        return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        for n in range(img.shape[1]):
            trans_img = img[:, n]
            trans_img = np.transpose(trans_img, (1, 2, 0))
            trans_img = Image.fromarray(np.uint8(trans_img))
            trans_img = np.array(transform(trans_img))
            trans_img = np.transpose(trans_img, (2, 0, 1))
            img[:, n, :, :] = torch.from_numpy(trans_img)

        return img

