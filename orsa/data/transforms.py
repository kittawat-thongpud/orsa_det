"""Letterbox resize + augmentation. Boxes carried as normalized cxcywh.

NASA-clean rule: NO magic numbers. Every probability, gain, range and colour
lives in `AugConfig` as a named, documented, config-overridable field. The
dataclass defaults are the single source of truth for the pre-defined params;
a YAML `augment:` section overrides any subset via `AugConfig.from_dict`.

Augmentation pipeline (Ultralytics-style ordering):
    mosaic (4-image, prob) -> mixup (2-image blend, prob) -> hflip -> hsv
    -> letterbox to model size -> normalize to cxcywh.

`Augmentor` owns the whole pipeline. It pulls extra images for mosaic/mixup
through the dataset's `_load_raw(i)` contract:
    _load_raw(i) -> (img_bgr HxWx3 uint8, boxes [n,4] xyxy abs-px, labels [n]).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, fields
import numpy as np
import cv2
import torch

# ----------------------------------------------------------------------------
# default constants (named pre-defined params; overridable through config)
# ----------------------------------------------------------------------------
_LETTERBOX_PAD_COLOR = 114      # neutral grey pad value, Ultralytics convention
_DEFAULT_SIZE = 640


def letterbox(img, new_size: int = _DEFAULT_SIZE, color: int = _LETTERBOX_PAD_COLOR):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=img.dtype)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


@dataclass(frozen=True)
class AugConfig:
    """All augmentation knobs. Defaults = pre-defined params; YAML overrides."""
    hflip_p: float = 0.5            # horizontal-flip probability
    hsv_h: float = 0.015            # HSV hue gain (fraction)
    hsv_s: float = 0.7              # HSV saturation gain (fraction)
    hsv_v: float = 0.4              # HSV value/brightness gain (fraction)
    mosaic_p: float = 0.0           # 4-image mosaic probability (0 disables)
    mixup_p: float = 0.0            # 2-image mixup probability (0 disables)
    mosaic_center_lo: float = 0.5   # mosaic centre lower bound (x model size)
    mosaic_center_hi: float = 1.5   # mosaic centre upper bound (x model size)
    mixup_beta: float = 8.0         # Beta(a,a) param for mixup blend ratio
    min_box_size: float = 2.0       # drop boxes smaller than this (px) post-aug
    pad_color: int = _LETTERBOX_PAD_COLOR

    @classmethod
    def from_dict(cls, d: dict | None) -> "AugConfig":
        d = d or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise KeyError(f"unknown augment keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})


def _hsv(img, hg: float, sg: float, vg: float):
    r = np.random.uniform(-1, 1, 3) * [hg, sg, vg] + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _hflip(img, boxes):
    w0 = img.shape[1]
    img = img[:, ::-1]
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w0 - boxes[:, [2, 0]]
    return img, boxes


def _drop_small(boxes, labels, min_size: float):
    if not len(boxes):
        return boxes, labels
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & \
           ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    return boxes[keep], labels[keep]


class Augmentor:
    """Config-driven augmentation owning the full image pipeline.

    Train mode applies mosaic/mixup/hflip/hsv per their probabilities; eval
    mode (train=False) only letterboxes + normalizes (deterministic)."""

    def __init__(self, size: int = _DEFAULT_SIZE, cfg: AugConfig | None = None,
                 train: bool = True):
        self.size = size
        self.cfg = cfg if cfg is not None else AugConfig()
        self.train = train

    # -- public ------------------------------------------------------------
    def __call__(self, dataset, index):
        """Returns (img[3,S,S] float, boxes cxcywh-norm, labels, orig_hw)."""
        img, boxes, labels = dataset._load_raw(index)
        orig_hw = (img.shape[0], img.shape[1])

        if self.train and self.cfg.mosaic_p > 0 and random.random() < self.cfg.mosaic_p:
            img, boxes, labels = self._mosaic(dataset, index, (img, boxes, labels))
        if self.train and self.cfg.mixup_p > 0 and random.random() < self.cfg.mixup_p:
            img, boxes, labels = self._mixup(dataset, img, boxes, labels)
        return (*self._finalize(img, boxes, labels), orig_hw)

    # -- mosaic ------------------------------------------------------------
    def _mosaic(self, dataset, index, primary):
        """Standard 4-image mosaic onto a 2S x 2S canvas, then carried as-is
        into _finalize which letterboxes 2S -> S."""
        s = self.size
        cfg = self.cfg
        canvas = np.full((s * 2, s * 2, 3), cfg.pad_color, dtype=np.uint8)
        xc = int(random.uniform(cfg.mosaic_center_lo, cfg.mosaic_center_hi) * s)
        yc = int(random.uniform(cfg.mosaic_center_lo, cfg.mosaic_center_hi) * s)
        n = len(dataset)
        indices = [index] + [random.randrange(n) for _ in range(3)]
        all_boxes, all_labels = [], []

        for quadrant, idx in enumerate(indices):
            img, boxes, labels = primary if idx == index and quadrant == 0 \
                else dataset._load_raw(idx)
            h, w = img.shape[:2]
            r = s / max(h, w)               # longest side -> model size
            if r != 1:
                img = cv2.resize(img, (int(round(w * r)), int(round(h * r))),
                                 interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]

            if quadrant == 0:    # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif quadrant == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif quadrant == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:                # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            if len(boxes):
                b = boxes.copy() * r
                b[:, [0, 2]] += padw
                b[:, [1, 3]] += padh
                all_boxes.append(b)
                all_labels.append(labels)

        if all_boxes:
            boxes = np.concatenate(all_boxes, 0)
            labels = np.concatenate(all_labels, 0)
            np.clip(boxes, 0, s * 2, out=boxes)
            boxes, labels = _drop_small(boxes, labels, cfg.min_box_size)
        else:
            boxes = np.zeros((0, 4), np.float32)
            labels = np.zeros((0,), np.int64)
        return canvas, boxes, labels

    # -- mixup -------------------------------------------------------------
    def _mixup(self, dataset, img, boxes, labels):
        """Blend with a second random image resized to match; concat boxes."""
        j = random.randrange(len(dataset))
        img2, boxes2, labels2 = dataset._load_raw(j)
        h2, w2 = img2.shape[:2]
        H, W = img.shape[:2]
        img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_LINEAR)
        if len(boxes2):
            boxes2 = boxes2.copy()
            boxes2[:, [0, 2]] *= W / w2
            boxes2[:, [1, 3]] *= H / h2

        lam = float(np.random.beta(self.cfg.mixup_beta, self.cfg.mixup_beta))
        img = (img.astype(np.float32) * lam +
               img2.astype(np.float32) * (1 - lam)).astype(np.uint8)
        if len(boxes2):
            boxes = np.concatenate([boxes, boxes2], 0) if len(boxes) else boxes2
            labels = np.concatenate([labels, labels2], 0) if len(labels) else labels2
        return img, boxes, labels

    # -- finalize ----------------------------------------------------------
    def _finalize(self, img, boxes, labels):
        cfg = self.cfg
        if self.train and random.random() < cfg.hflip_p and len(boxes):
            img, boxes = _hflip(img, boxes)
        if self.train:
            img = _hsv(img, cfg.hsv_h, cfg.hsv_s, cfg.hsv_v)

        img, r, left, top = letterbox(img, self.size, cfg.pad_color)
        if len(boxes):
            boxes = boxes * r
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / self.size
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / self.size
            bw = (boxes[:, 2] - boxes[:, 0]) / self.size
            bh = (boxes[:, 3] - boxes[:, 1]) / self.size
            boxes = np.stack([cx, cy, bw, bh], 1).clip(0, 1)
        else:
            boxes = np.zeros((0, 4), np.float32)

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR->RGB
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (img, torch.from_numpy(boxes).float(),
                torch.as_tensor(labels, dtype=torch.long))


class Compose:
    """Single-image transform (legacy / no mosaic). Kept for compatibility;
    delegates to Augmentor with mosaic+mixup disabled."""

    def __init__(self, size=_DEFAULT_SIZE, train=True, cfg: AugConfig | None = None):
        base = cfg if cfg is not None else AugConfig()
        # legacy path never composites multiple images
        self.aug = Augmentor(size, AugConfig.from_dict(
            {**{f.name: getattr(base, f.name) for f in fields(AugConfig)},
             "mosaic_p": 0.0, "mixup_p": 0.0}), train)

    def __call__(self, img, boxes, labels):
        return self.aug._finalize(img, np.asarray(boxes, np.float32).reshape(-1, 4),
                                  np.asarray(labels, np.int64).reshape(-1))
