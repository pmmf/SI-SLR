import torch
from skimage import transform as tf
from skimage.exposure import rescale_intensity
import numpy as np
import copy
import math
from imgaug import augmenters as iaa
import imgaug as ia


class numpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample = sample.transpose((2, 0, 1))
        # sample = sample/255.
        return torch.from_numpy(sample).float()


class Normalize_0_1(object):
    """Normalize range to (0,1)."""

    def __call__(self, sample):
        return sample/255.


class Normalize_1_1(object):
    """Normalize range to (-1,1)."""

    def __call__(self, sample):
        sample = sample*2 - 1.
        return sample


class randomHorizontalFlip(object):
    """Horizontal flipping with 50% of probability."""

    def __call__(self, sample):
        is2flip = np.random.choice([0, 1], 1)

        if is2flip:
            return sample[:, ::-1, :]
        return sample


class randomAffineTransform(object):
    """Geometric transformation - scaling, rotation, shear, translation"""

    def __call__(self, sample):
        sample_t = copy.deepcopy(sample)

        # scaling parameters
        epsilon = np.random.choice([0, 1, 2, 3, 4], 1)
        s = np.float(sample_t.shape[0]/(sample_t.shape[1]-epsilon))
        # rotation parameter
        theta = np.random.choice([-math.pi/18.0, 0, math.pi/18.0], 1)
        # shear parameters
        shear = np.random.choice([-0.1, 0, 0.1], 1)
        # translation parameters
        tx = np.random.choice([0, 1, 2, 3, 4], 1)
        ty = np.random.choice([0, 1, 2, 3, 4], 1)
        # Create Afine transform
        afine_tf = tf.AffineTransform(scale=(s, s),
                                      rotation=theta,
                                      shear=shear,
                                      translation=(tx, ty))
        # apply transformation
        sample_t = tf.warp(sample_t, afine_tf, order=5, mode='reflect')
        sample_t[sample_t < 0] = 0
        sample_t[sample_t > 1] = 1
        return sample_t


class randomColourTransform(object):
    """Colour transformation - histogram streching"""

    def __call__(self, sample):
        sample_t = copy.deepcopy(sample)

        # percentiles
        p_rate = np.random.randint(0, 10)
        p_low = np.percentile(sample_t, p_rate)
        p_rate = np.random.randint(95, 100)
        p_hight = np.percentile(sample_t, p_rate)

        is2transform = np.random.choice([0, 1], 1)
        if is2transform:
            sample_t = rescale_intensity(sample, in_range=(p_low, p_hight))
            sample_t[sample_t < 0] = 0
            sample_t[sample_t > 1] = 1
        return sample_t


class ImgAugTransform(object):
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.Affine(scale={"x": (0.85, 1.15),
                                            "y": (0.85, 1.15)},
                                     translate_percent={"x": (-0.1, 0.1),
                                                        "y": (-0.1, 0.1)},
                                     rotate=(-12, 12),
                                     shear=(-5, 5),
                                     order=[0, 1],
                                     mode='reflect'))])

    def __call__(self, img):
        # img = np.array(img)
        return self.aug.augment_image(img)


if __name__ == '__main__':
    pass
