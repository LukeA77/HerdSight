"""Extra training augmentations for D-FINE (Phase 2, plan step G2), registered by name.

gpu/train_dfine.py copies this file into the pinned D-FINE checkout as
src/data/transforms/hs_ops.py and imports it from that package's __init__, so the ops can be named
in a training config like D-FINE's own. Each op takes and returns D-FINE's sample tuple
(PIL image, target dict, dataset); boxes are target["boxes"], absolute x0, y0, x1, y1.

Why these ops (results/baselines/derval_rootcause.json): the farm the Phase 1 model never saw has
mostly dark cattle, often in tree shade and packed together; nadir images have no "up".
  HsRandomRot90     rotate by 90, 180 or 270 degrees (boxes rotated exactly)
  HsRandomVFlip     vertical flip (D-FINE already has the horizontal one)
  HsDarkenShade     lower brightness and contrast, towards shade and dark animals
  HsRandomShadow    darken a soft-edged elliptical patch, like a tree or cloud shadow
  HsRandomGrayscale drop colour, so coat colour matters less
  HsRandomBlur      mild Gaussian blur (motion, focus, compression)
Photometric ops change pixel values only; geometric ops move boxes with the pixels.
"""

import random

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import torch
import torch.nn as nn

from ...core import register
from .._misc import convert_to_tv_tensor


def _split(inputs):
    sample = inputs if len(inputs) > 1 else inputs[0]
    return sample[0], sample[1], sample[2:]


def _set_boxes(target, boxes, w, h):
    target["boxes"] = convert_to_tv_tensor(boxes, key="boxes", spatial_size=(h, w))
    return target


@register()
class HsRandomRot90(nn.Module):
    def __init__(self, p=0.75):
        super().__init__()
        self.p = p

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() >= self.p:
            return (img, target, *rest)
        k = random.choice((1, 2, 3))  # counter-clockwise quarter turns
        w, h = img.size
        b = target.get("boxes")
        if b is not None and len(b):
            x0, y0, x1, y1 = (b[:, i].clone() for i in range(4))
            if k == 1:  # (x, y) -> (y, W - x)
                nb = torch.stack([y0, w - x1, y1, w - x0], 1)
            elif k == 2:  # (x, y) -> (W - x, H - y)
                nb = torch.stack([w - x1, h - y1, w - x0, h - y0], 1)
            else:  # (x, y) -> (H - y, x)
                nb = torch.stack([h - y1, x0, h - y0, x1], 1)
        else:
            nb = b
        img = img.transpose(
            {1: PIL.Image.ROTATE_90, 2: PIL.Image.ROTATE_180, 3: PIL.Image.ROTATE_270}[k]
        )
        nw, nh = img.size
        if nb is not None:
            target = _set_boxes(target, nb, nw, nh)
        return (img, target, *rest)


@register()
class HsRandomVFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() >= self.p:
            return (img, target, *rest)
        w, h = img.size
        b = target.get("boxes")
        if b is not None and len(b):
            nb = torch.stack([b[:, 0], h - b[:, 3], b[:, 2], h - b[:, 1]], 1)
            target = _set_boxes(target, nb, w, h)
        return (img.transpose(PIL.Image.FLIP_TOP_BOTTOM), target, *rest)


@register()
class HsDarkenShade(nn.Module):
    def __init__(self, p=0.5, brightness=(0.45, 0.9), contrast=(0.7, 1.1)):
        super().__init__()
        self.p, self.brightness, self.contrast = p, tuple(brightness), tuple(contrast)

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() < self.p:
            img = PIL.ImageEnhance.Brightness(img).enhance(random.uniform(*self.brightness))
            img = PIL.ImageEnhance.Contrast(img).enhance(random.uniform(*self.contrast))
        return (img, target, *rest)


@register()
class HsRandomShadow(nn.Module):
    def __init__(self, p=0.3, strength=(0.35, 0.7)):
        super().__init__()
        self.p, self.strength = p, tuple(strength)

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() >= self.p:
            return (img, target, *rest)
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        ax, ay = random.uniform(0.15, 0.6) * w, random.uniform(0.15, 0.6) * h
        theta = random.uniform(0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        u = ((xx - cx) * c + (yy - cy) * s) / ax
        v = (-(xx - cx) * s + (yy - cy) * c) / ay
        r = np.sqrt(u * u + v * v)
        soft = np.clip((1.2 - r) / 0.4, 0.0, 1.0)  # 1 inside, fading to 0 at the edge
        factor = 1.0 - soft * (1.0 - random.uniform(*self.strength))
        arr = np.asarray(img.convert("RGB"), np.float32) * factor[..., None]
        img = PIL.Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return (img, target, *rest)


@register()
class HsRandomGrayscale(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() < self.p:
            img = img.convert("L").convert("RGB")
        return (img, target, *rest)


@register()
class HsRandomBlur(nn.Module):
    def __init__(self, p=0.2, radius=(0.5, 1.5)):
        super().__init__()
        self.p, self.radius = p, tuple(radius)

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() < self.p:
            img = img.filter(PIL.ImageFilter.GaussianBlur(random.uniform(*self.radius)))
        return (img, target, *rest)
