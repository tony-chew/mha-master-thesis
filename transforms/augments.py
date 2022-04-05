import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    """
    Construct the composition of the augmentations
    """
    def __init__(self,
                 augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self,
                 img,
                 mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class RandomCrop(object):
    """
    Random crop augmentation
    """
    def __init__(self,
                 size,
                 padding=0):
        """
        :param size: input resolution
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))

        else:
            self.size = size

        self.padding = padding

    def __call__(self,
                 img,
                 mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size

        w, h = img.size
        ch, cw = self.size

        if w == cw and h == ch:
            return img, mask

        if w < cw or h < ch:
            pw = cw - w if cw > w else 0
            ph = ch - h if ch > h else 0
            padding = (pw, ph, pw, ph)
            img = ImageOps.expand(img, padding, fill=0)
            mask = ImageOps.expand(mask, padding, fill=250)
            w, h = img.size

            assert img.size == mask.size

        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)

        return (img.crop((x1, y1, x1 + cw, y1 + ch)), mask.crop((x1, y1, x1 + cw, y1 + ch)))


class RandomHorizontallyFlip(object):
    """
    Random horizontal flip augmentation
    """
    def __init__(self,
                 p):
        """
        :param p: chance of flipping the image
        """
        self.p = p

    def __call__(self,
                 img,
                 mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT))

        return img, mask


class RandomScaleCrop(object):
    """
    Scale and crop the image to the given input resolution
    """
    def __init__(self,
                 size):
        """
        :param size: input resolution
        """
        self.size = size
        self.crop = RandomCrop(self.size)

    def __call__(self,
                 img,
                 mask):
        assert img.size == mask.size

        r = random.uniform(0.5, 2.0)
        w, h = img.size
        new_size = (int(w * r), int(h * r))

        return self.crop(img.resize(new_size, Image.BILINEAR), mask.resize(new_size, Image.NEAREST))


class Scale(object):
    """
    Resize the input resolution to the given size
    """
    def __init__(self,
                 size):
        """
        :param size: input resolution
        """
        self.size = size

    def __call__(self,
                 img,
                 mask):
        assert img.size == mask.size

        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask

        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))

        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
