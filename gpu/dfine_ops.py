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
  HsCopyPaste       paste 2-6 labelled animals from another training tile as a touching group
  HsScaleJitter     mild scale change (0.8-1.25) at the same tile size
Photometric ops change pixel values only; geometric ops move boxes with the pixels.

Dark-coat, tight-group and outline training (after G2 failed on unseen farms, D-49) adds two ops
that use each training animal's oval (gpu/tile_ovals.py, copied beside this file as hs_ovals.py;
the per-tile file comes from gpu/make_tile_ovals.py). They run first, before any op moves pixels:
  HsCoatDarken      darken and part-desaturate some animals' coats inside their ovals only
  HsOutlinePaste    paste 2-6 animals cut along their ovals (no background patch) as a touching
                    group, some with darkened coats
"""

import json
import os
import random

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import torch
import torch.nn as nn

from ...core import register
from .._misc import convert_to_tv_tensor
from .hs_ovals import darken_coats, oval_alpha, oval_box


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


def _boxes_list(target):
    b = target.get("boxes")
    return [] if b is None or not len(b) else [list(map(float, r)) for r in b.tolist()]


def _update(target, boxes, w, h, keep=None, extra_labels=None):
    """Write boxes back; `keep` drops rows (bool list over the old rows), extra_labels appends."""
    old = len(target["labels"]) if "labels" in target else 0
    keep = keep if keep is not None else [True] * old
    idx = torch.tensor([i for i, k in enumerate(keep) if k], dtype=torch.long)
    for key in ("labels", "area", "iscrowd"):
        if key in target and len(target[key]) == old:
            target[key] = target[key][idx] if old else target[key]
    if extra_labels:
        n = len(extra_labels)
        dtype = target["labels"].dtype if "labels" in target else torch.int64
        target["labels"] = torch.cat(
            [
                target.get("labels", torch.zeros(0, dtype=dtype)),
                torch.tensor(extra_labels, dtype=dtype),
            ]
        )
        new_area = torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes[-n:]])
        if "area" in target:
            target["area"] = torch.cat([target["area"].float(), new_area.float()])
        if "iscrowd" in target:
            target["iscrowd"] = torch.cat(
                [target["iscrowd"], torch.zeros(n, dtype=target["iscrowd"].dtype)]
            )
    t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return _set_boxes(target, t, w, h)


def _feather_mask(w, h, feather):
    """Soft elliptical mask: cattle seen from above fill an ellipse, not their box."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx + 0.5) / w * 2 - 1) ** 2 + ((yy + 0.5) / h * 2 - 1) ** 2)
    a = np.clip((1.0 - r) / max(feather, 1e-3), 0.0, 1.0) * 255
    return PIL.Image.fromarray(a.astype(np.uint8), "L")


def _cover(a, b):
    """Share of box b covered by box a."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    area = (b[2] - b[0]) * (b[3] - b[1])
    return 0.0 if iw <= 0 or ih <= 0 or area <= 0 else iw * ih / area


@register()
class HsCopyPaste(nn.Module):
    """Paste 2-6 labelled animals from another training tile as a tight group (crowding).

    Animals are cut from their boxes with a soft elliptical edge and placed one after another a
    little less than a body apart, so they touch as in a packed herd. A paste is skipped if it
    would cover more than max_cover of any animal already in the tile (which would hide a labelled
    animal), so every animal visible in the result has a box. Donors come from the same training
    set (the dataset passed with each sample), so no validation image is ever used.
    """

    def __init__(self, p=0.5, animals=(2, 6), feather=0.25, max_cover=0.3, tries=10):
        super().__init__()
        self.p, self.animals, self.feather = p, tuple(animals), feather
        self.max_cover, self.tries = max_cover, tries

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        dataset = rest[0] if rest else None
        if random.random() >= self.p or not hasattr(dataset, "load_item"):
            return (img, target, *rest)
        donor = None
        for _ in range(self.tries):
            dimg, dt = dataset.load_item(random.randrange(len(dataset)))
            dboxes = _boxes_list(dt)
            if dboxes:
                donor = (dimg.convert("RGB"), dboxes, dt["labels"].tolist())
                break
        if donor is None:
            return (img, target, *rest)
        dimg, dboxes, dlabels = donor
        img = img.convert("RGB").copy()
        w, h = img.size
        boxes = _boxes_list(target)
        pick = random.sample(range(len(dboxes)), min(len(dboxes), random.randint(*self.animals)))
        x, y = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
        new_labels = []
        for i in pick:
            x0, y0, x1, y1 = dboxes[i]
            bw, bh = int(round(x1 - x0)), int(round(y1 - y0))
            if bw < 4 or bh < 4:
                continue
            px, py = int(round(x - bw / 2)), int(round(y - bh / 2))
            nb = [px, py, px + bw, py + bh]
            step = (bw + bh) / 2 * random.uniform(0.55, 0.9)
            ang = random.uniform(0, 2 * np.pi)
            x, y = x + step * np.cos(ang), y + step * np.sin(ang)
            if nb[0] < 0 or nb[1] < 0 or nb[2] > w or nb[3] > h:
                continue
            if any(_cover(nb, b) > self.max_cover for b in boxes):
                continue
            crop = dimg.crop(
                (int(round(x0)), int(round(y0)), int(round(x0)) + bw, int(round(y0)) + bh)
            )
            img.paste(crop, (px, py), _feather_mask(bw, bh, self.feather))
            boxes.append([float(v) for v in nb])
            new_labels.append(dlabels[i])
        if new_labels:
            target = _update(target, boxes, w, h, extra_labels=new_labels)
        return (img, target, *rest)


@register()
class HsScaleJitter(nn.Module):
    """Mild scale change (0.8-1.25 times) at the same tile size: shrink onto a canvas of the
    image's mean colour, or enlarge and crop. Animals less than min_visible inside are dropped,
    the same rule the tile package uses at tile edges (configs/baselines.yaml)."""

    def __init__(self, p=0.5, scale=(0.8, 1.25), min_visible=0.5):
        super().__init__()
        self.p, self.scale, self.min_visible = p, tuple(scale), min_visible

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        if random.random() >= self.p:
            return (img, target, *rest)
        img = img.convert("RGB")
        w, h = img.size
        s = random.uniform(*self.scale)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        big = img.resize((nw, nh), PIL.Image.BILINEAR)
        if s < 1:
            mean = tuple(int(v) for v in np.asarray(img, np.float32).reshape(-1, 3).mean(0))
            out = PIL.Image.new("RGB", (w, h), mean)
            ox, oy = random.randint(0, w - nw), random.randint(0, h - nh)
            out.paste(big, (ox, oy))
            shift = (ox, oy)
        else:
            ox, oy = random.randint(0, nw - w), random.randint(0, nh - h)
            out = big.crop((ox, oy, ox + w, oy + h))
            shift = (-ox, -oy)
        boxes, keep = [], []
        for b in _boxes_list(target):
            sb = [
                b[0] * s + shift[0],
                b[1] * s + shift[1],
                b[2] * s + shift[0],
                b[3] * s + shift[1],
            ]
            cb = [max(0, sb[0]), max(0, sb[1]), min(w, sb[2]), min(h, sb[3])]
            full = (sb[2] - sb[0]) * (sb[3] - sb[1])
            vis = max(0, cb[2] - cb[0]) * max(0, cb[3] - cb[1])
            ok = full > 0 and vis / full >= self.min_visible
            keep.append(ok)
            if ok:
                boxes.append(cb)
        if keep:
            target = _update(target, boxes, w, h, keep=keep)
            if "area" in target and len(boxes):
                target["area"] = torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        return (out, target, *rest)


_OVAL_FILES = {}


def _ovals_file(path):
    if path not in _OVAL_FILES:
        with open(path) as f:
            _OVAL_FILES[path] = json.load(f)
    return _OVAL_FILES[path]


def _tile_name(dataset, target):
    """The tile's file name relative to the image folder, the key of the oval file."""
    path = target.get("image_path") if isinstance(target, dict) else None
    folder = getattr(dataset, "img_folder", None)
    if not path or not folder:
        return None
    return os.path.relpath(path, folder).replace(os.sep, "/")


@register()
class HsCoatDarken(nn.Module):
    """Darken the coats of some animals inside their ovals; the background is untouched, so the
    model learns dark animals rather than dark photos (unlike HsDarkenShade)."""

    def __init__(self, ovals_file, p=0.5, animal_p=0.7, factor=(0.25, 0.6), desaturate=(0.3, 0.8)):
        super().__init__()
        self.ovals_file, self.p, self.animal_p = ovals_file, p, animal_p
        self.factor, self.desaturate = tuple(factor), tuple(desaturate)

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        dataset = rest[0] if rest else None
        if random.random() >= self.p or dataset is None:
            return (img, target, *rest)
        ovals = _ovals_file(self.ovals_file).get(_tile_name(dataset, target))
        pick = [o for o in (ovals or []) if random.random() < self.animal_p]
        if not pick:
            return (img, target, *rest)
        arr = darken_coats(
            np.asarray(img.convert("RGB")),
            pick,
            random.uniform(*self.factor),
            random.uniform(*self.desaturate),
        )
        return (PIL.Image.fromarray(arr), target, *rest)


@register()
class HsOutlinePaste(nn.Module):
    """Paste 2-6 animals from another training tile, cut along their ovals, as a touching group.

    Unlike HsCopyPaste no background patch comes with them (G2's soft-edged box cut-outs carried
    grass and shadow, a likely source of its extra false animals on empty photos, D-49). Each
    pasted animal's box is its oval's box. Steps between neighbours are a fraction of the body
    size, so the group touches as a packed herd. A paste is skipped if it would cover more than
    max_cover of an animal already there. Donors come from the same training set only.
    """

    def __init__(
        self,
        ovals_file,
        p=0.5,
        animals=(2, 6),
        spacing=(0.45, 0.8),
        max_cover=0.3,
        tries=10,
        dark_p=0.5,
        factor=(0.25, 0.6),
        desaturate=(0.3, 0.8),
    ):
        super().__init__()
        self.ovals_file, self.p, self.animals, self.spacing = (
            ovals_file,
            p,
            tuple(animals),
            tuple(spacing),
        )
        self.max_cover, self.tries, self.dark_p = max_cover, tries, dark_p
        self.factor, self.desaturate = tuple(factor), tuple(desaturate)

    def forward(self, *inputs):
        img, target, rest = _split(inputs)
        dataset = rest[0] if rest else None
        if random.random() >= self.p or not hasattr(dataset, "load_item"):
            return (img, target, *rest)
        table = _ovals_file(self.ovals_file)
        donor = None
        for _ in range(self.tries):
            dimg, dt = dataset.load_item(random.randrange(len(dataset)))
            dovals = table.get(_tile_name(dataset, dt))
            if dovals:
                donor = (np.asarray(dimg.convert("RGB")), dovals, int(dt["labels"][0]))
                break
        if donor is None:
            return (img, target, *rest)
        darr, dovals, label = donor
        if random.random() < self.dark_p:
            darr = darken_coats(
                darr, dovals, random.uniform(*self.factor), random.uniform(*self.desaturate)
            )
        dh, dw = darr.shape[:2]
        base = np.asarray(img.convert("RGB")).astype(np.float32)
        h, w = base.shape[:2]
        boxes = _boxes_list(target)
        pick = random.sample(range(len(dovals)), min(len(dovals), random.randint(*self.animals)))
        x, y = random.uniform(0.2, 0.8) * w, random.uniform(0.2, 0.8) * h
        added = 0
        for i in pick:
            oval = dovals[i]
            bx0, by0, bx1, by1 = oval_box(oval)
            bw, bh = bx1 - bx0, by1 - by0
            if bw < 4 or bh < 4 or bx0 < 0 or by0 < 0 or bx1 > dw or by1 > dh:
                continue  # only whole animals are pasted
            dx, dy = x - oval[0], y - oval[1]  # move the oval centre to (x, y)
            nb = [bx0 + dx, by0 + dy, bx1 + dx, by1 + dy]
            step = (bw + bh) / 2 * random.uniform(*self.spacing)
            ang = random.uniform(0, 2 * np.pi)
            x, y = x + step * np.cos(ang), y + step * np.sin(ang)
            if nb[0] < 0 or nb[1] < 0 or nb[2] > w or nb[3] > h:
                continue
            if any(_cover(nb, b) > self.max_cover for b in boxes):
                continue
            sx0, sy0 = int(np.floor(bx0)), int(np.floor(by0))
            sx1, sy1 = int(np.ceil(bx1)), int(np.ceil(by1))
            tx0, ty0 = sx0 + int(round(dx)), sy0 + int(round(dy))
            if tx0 < 0 or ty0 < 0 or tx0 + (sx1 - sx0) > w or ty0 + (sy1 - sy0) > h:
                continue
            local = [oval[0] - sx0, oval[1] - sy0, oval[2], oval[3], oval[4]]
            a = oval_alpha(sx1 - sx0, sy1 - sy0, local)[..., None]
            patch = darr[sy0:sy1, sx0:sx1].astype(np.float32)
            region = base[ty0 : ty0 + (sy1 - sy0), tx0 : tx0 + (sx1 - sx0)]
            region[:] = region * (1 - a) + patch * a
            boxes.append([float(v) for v in nb])
            added += 1
        if not added:
            return (img, target, *rest)
        out = PIL.Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        target = _update(target, boxes, w, h, extra_labels=[label] * added)
        return (out, target, *rest)
