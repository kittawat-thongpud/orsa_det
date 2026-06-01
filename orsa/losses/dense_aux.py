"""Dense auxiliary loss (TAL-lite, one-to-many) for the train-only AuxDenseHead.

Each pyramid cell whose center lies inside a GT box is a positive for that GT
(center-prior assignment; smallest-area GT wins ties). Positives are supervised
with focal cls + L1 + GIoU on the dense (cxcywh, sigmoid) predictions; all other
cells are cls-background. This is a coarse one-to-many signal that stabilizes the
backbone/neck early and is dropped at inference (`fuse_for_inference`).

aux_dense item per scale: {"logits": [B,C,H,W], "boxes": [B,4,H,W]} (boxes sigmoid cxcywh).
grids per scale: [H*W, 2] normalized (x, y) centers.
"""
from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn.functional as F
from .box_ops import cxcywh_to_xyxy, generalized_box_iou


def _focal(logits, targets, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.sum()


def dense_aux_loss(aux_dense: List[Dict[str, torch.Tensor]], grids: List[torch.Tensor],
                   targets, num_classes: int,
                   w_cls=1.0, w_l1=2.5, w_giou=1.0) -> torch.Tensor:
    device = aux_dense[0]["logits"].device
    B = aux_dense[0]["logits"].shape[0]
    cls_loss = torch.zeros((), device=device)
    l1 = torch.zeros((), device=device)
    giou = torch.zeros((), device=device)
    n_pos = 0

    for lvl, item in enumerate(aux_dense):
        logits = item["logits"]                       # [B,C,H,W]
        boxes = item["boxes"]                          # [B,4,H,W] cxcywh sigmoid
        Bc, C, H, W = logits.shape
        HW = H * W
        grid = grids[lvl].to(device)                   # [HW,2] (x,y)
        log_f = logits.permute(0, 2, 3, 1).reshape(B, HW, C)   # [B,HW,C]
        box_f = boxes.permute(0, 2, 3, 1).reshape(B, HW, 4)    # [B,HW,4]

        for b in range(B):
            tgt = torch.zeros((HW, C), device=device)
            gt = targets[b]["boxes"]                   # [n,4] cxcywh
            labels = targets[b]["labels"]
            if gt.numel():
                xyxy = cxcywh_to_xyxy(gt)              # [n,4]
                areas = (gt[:, 2] * gt[:, 3]).clamp(min=1e-6)
                # [n,HW] cells whose center is inside each GT box
                inside = ((grid[None, :, 0] >= xyxy[:, 0:1]) & (grid[None, :, 0] <= xyxy[:, 2:3]) &
                          (grid[None, :, 1] >= xyxy[:, 1:2]) & (grid[None, :, 1] <= xyxy[:, 3:4]))
                # smallest-area GT wins each cell
                big = areas[:, None].expand(-1, HW).clone()
                big[~inside] = float("inf")
                best_area, best_gt = big.min(0)        # [HW]
                pos = torch.isfinite(best_area)        # [HW] cells with >=1 GT
                if pos.any():
                    pi = pos.nonzero(as_tuple=True)[0]
                    g = best_gt[pi]
                    tgt[pi, labels[g]] = 1.0
                    pb = box_f[b][pi]                  # [P,4] cxcywh
                    tb = gt[g]                         # [P,4] cxcywh
                    l1 = l1 + F.l1_loss(pb, tb, reduction="sum")
                    giou = giou + (1 - torch.diag(generalized_box_iou(
                        cxcywh_to_xyxy(pb), cxcywh_to_xyxy(tb)))).sum()
                    n_pos += pi.numel()
            cls_loss = cls_loss + _focal(log_f[b], tgt)

    n = max(n_pos, 1)
    return w_cls * cls_loss / n + w_l1 * l1 / n + w_giou * giou / n
