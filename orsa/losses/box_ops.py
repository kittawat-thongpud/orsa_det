"""Box utilities (cxcywh <-> xyxy, IoU, GIoU). Normalized coords."""
from __future__ import annotations
import torch


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], -1)


def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(0) * (b[..., 3] - b[..., 1]).clamp(0)


def box_iou(a, b):
    """a [N,4] b [M,4] xyxy -> iou [N,M], union [N,M]."""
    area_a, area_b = box_area(a), box_area(b)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-7), union


def generalized_box_iou(a, b):
    """GIoU matrix [N,M] for xyxy boxes."""
    iou, union = box_iou(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclose = wh[..., 0] * wh[..., 1]
    return iou - (enclose - union) / enclose.clamp(min=1e-7)
