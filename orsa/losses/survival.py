"""Token Survival Loss — prevents sparse starvation of small/occluded objects.

For each GT g, S_g = MEAN token score over locations whose center lies inside
box g (averaged across scales that contain >=1 such location), so S_g in [0,1]
regardless of box size. Loss = mean_g max(0, tau_g - S_g)^2. tau_g is larger for
small objects (they get squeezed out of top-K first), pushing the scorer to keep
scores high inside small boxes so their tokens survive selection.

NB: an earlier version summed raw scores over all inside cells, so S_g reached
the tens-to-hundreds while tau was 1.5-4.0 -> deficit was always 0 and the loss
was inert. Mean + [0,1] tau gives a live gradient.
"""
from __future__ import annotations
from typing import List
import torch
from .box_ops import cxcywh_to_xyxy


def survival_loss(dense_scores: List[torch.Tensor], grids: List[torch.Tensor],
                  targets, tau_small=0.6, tau_large=0.25, small_area=0.02) -> torch.Tensor:
    """dense_scores: per scale [B,H,W]; grids: per scale [H*W,2] (x,y) normalized."""
    device = dense_scores[0].device
    B = dense_scores[0].shape[0]
    flat = [s.flatten(1) for s in dense_scores]   # per scale [B, H*W]
    total = torch.zeros((), device=device)
    n_g = 0
    for b in range(B):
        boxes = targets[b]["boxes"]               # [n,4] cxcywh
        if boxes.numel() == 0:
            continue
        xyxy = cxcywh_to_xyxy(boxes)              # [n,4]
        areas = (boxes[:, 2] * boxes[:, 3]).clamp(min=1e-6)
        # tau: small -> tau_small, large -> tau_large (linear in area, clamped)
        t = (areas / small_area).clamp(0, 1)
        tau = tau_small + (tau_large - tau_small) * t   # [n]
        S_sum = torch.zeros(boxes.shape[0], device=device)
        cnt = torch.zeros(boxes.shape[0], device=device)
        for l, grid in enumerate(grids):
            c = grid.to(device)                  # [HW,2]
            inside = ((c[None, :, 0] >= xyxy[:, 0:1]) & (c[None, :, 0] <= xyxy[:, 2:3]) &
                      (c[None, :, 1] >= xyxy[:, 1:2]) & (c[None, :, 1] <= xyxy[:, 3:4])).float()  # [n,HW]
            S_sum = S_sum + (inside * flat[l][b][None, :]).sum(1)
            cnt = cnt + inside.sum(1)
        S = S_sum / cnt.clamp(min=1.0)           # mean score inside box, in [0,1]
        deficit = (tau - S).clamp(min=0)
        total = total + (deficit ** 2).sum()
        n_g += boxes.shape[0]
    return total / max(n_g, 1)
