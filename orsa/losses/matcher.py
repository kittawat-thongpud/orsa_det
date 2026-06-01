"""Hungarian matcher for one-to-one set prediction (main head)."""
from __future__ import annotations
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from .box_ops import cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, c_cls=2.0, c_l1=5.0, c_giou=2.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.c_cls, self.c_l1, self.c_giou = c_cls, c_l1, c_giou
        self.alpha, self.gamma = focal_alpha, focal_gamma

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """pred_logits [B,Q,C], pred_boxes [B,Q,4] cxcywh.
        targets: list of dict{labels[ni], boxes[ni,4] cxcywh}. Returns list of (idx_q, idx_g)."""
        B, Q, C = pred_logits.shape
        indices = []
        for b in range(B):
            tgt = targets[b]
            if tgt["labels"].numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            prob = pred_logits[b].sigmoid()                       # [Q,C]
            tl = tgt["labels"]                                    # [n]
            # focal class cost
            neg = (1 - self.alpha) * (prob ** self.gamma) * (-(1 - prob + 1e-8).log())
            pos = self.alpha * ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
            c_cls = pos[:, tl] - neg[:, tl]                       # [Q,n]
            c_l1 = torch.cdist(pred_boxes[b], tgt["boxes"], p=1)  # [Q,n]
            c_giou = -generalized_box_iou(cxcywh_to_xyxy(pred_boxes[b]), cxcywh_to_xyxy(tgt["boxes"]))
            C_mat = self.c_cls * c_cls + self.c_l1 * c_l1 + self.c_giou * c_giou
            qi, gi = linear_sum_assignment(C_mat.cpu())
            indices.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(gi, dtype=torch.long)))
        return indices
