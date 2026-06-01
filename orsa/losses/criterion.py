"""SetCriterion for ORSA-Det.

Total = main set loss (focal cls + L1 + GIoU, one-to-one)
      + deep-supervision aux (per decoder layer)
      + aux query-group loss (one-to-many, Group-DETR)
      + lambda_surv * survival loss
      + lambda_sparse * gate sparsity (L1 on dense scores)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import HungarianMatcher
from .box_ops import cxcywh_to_xyxy, generalized_box_iou
from .survival import survival_loss
from .dense_aux import dense_aux_loss


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.sum() if reduction == "sum" else loss.mean()


class SetCriterion(nn.Module):
    def __init__(self, num_classes, weights=None, lambda_surv=1.0, lambda_sparse=1e-3,
                 lambda_dense=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        self.w = weights or {"cls": 2.0, "l1": 5.0, "giou": 2.0}
        self.lambda_surv = lambda_surv
        self.lambda_sparse = lambda_sparse
        self.lambda_dense = lambda_dense

    def _set_loss(self, logits, boxes, targets, indices):
        B, Q, C = logits.shape
        tgt_cls = torch.zeros((B, Q, C), device=logits.device)
        l1, giou, n_box = logits.new_zeros(()), logits.new_zeros(()), 0
        for b, (qi, gi) in enumerate(indices):
            if qi.numel() == 0:
                continue
            qi, gi = qi.to(logits.device), gi.to(logits.device)
            tgt_cls[b, qi, targets[b]["labels"][gi]] = 1.0
            pb, tb = boxes[b][qi], targets[b]["boxes"][gi]
            l1 = l1 + F.l1_loss(pb, tb, reduction="sum")
            giou = giou + (1 - torch.diag(generalized_box_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(tb)))).sum()
            n_box += gi.numel()
        n_box = max(n_box, 1)
        cls = sigmoid_focal_loss(logits, tgt_cls) / n_box
        return self.w["cls"] * cls + self.w["l1"] * l1 / n_box + self.w["giou"] * giou / n_box

    def forward(self, outputs, targets):
        logs = {}
        idx = self.matcher(outputs["pred_logits"], outputs["pred_boxes"], targets)
        main = self._set_loss(outputs["pred_logits"], outputs["pred_boxes"], targets, idx)
        logs["loss_main"] = main.detach()
        total = main

        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            ai = self.matcher(aux["pred_logits"], aux["pred_boxes"], targets)
            la = self._set_loss(aux["pred_logits"], aux["pred_boxes"], targets, ai)
            total = total + la
            logs[f"loss_aux{i}"] = la.detach()

        if "aux_group" in outputs:  # one-to-many: match group queries too
            g = outputs["aux_group"]
            gi = self.matcher(g["pred_logits"], g["pred_boxes"], targets)
            lg = self._set_loss(g["pred_logits"], g["pred_boxes"], targets, gi)
            total = total + lg
            logs["loss_group"] = lg.detach()

        if "aux_dense" in outputs and self.lambda_dense > 0:  # TAL-lite one-to-many
            ld = dense_aux_loss(outputs["aux_dense"], outputs["grids"], targets, self.num_classes)
            total = total + self.lambda_dense * ld
            logs["loss_dense"] = ld.detach()

        if "dense_scores" in outputs and self.lambda_surv > 0:
            ls = survival_loss(outputs["dense_scores"], outputs["grids"], targets)
            total = total + self.lambda_surv * ls
            logs["loss_surv"] = ls.detach()
            sparse = sum(s.mean() for s in outputs["dense_scores"]) / len(outputs["dense_scores"])
            total = total + self.lambda_sparse * sparse
            logs["loss_sparse"] = sparse.detach()

        logs["loss_total"] = total.detach()
        return total, logs
