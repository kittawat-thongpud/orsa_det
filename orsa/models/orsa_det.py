"""ORSA-Det 2027 — full model assembly.

backbone (CNN) -> PAN-lite neck -> LEM (P3/P4) -> Sparse Token Bank
 -> Query-lite decoder head (+ aux dense head, train-only).

Outputs a dict consumed by the criterion (training) or postprocessor (inference).
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from .backbone import CSPBackbone
from .neck import PANLite
from .lem import LEMStack
from .sparse_token import SparseTokenBank
from .head import QueryLiteHead, AuxDenseHead


# width/depth presets per scale target (see full_plan §6)
PRESETS = {
    "nano":  dict(width=(16, 32, 64, 128, 256),  depth=(1, 1, 1, 1), embed_dim=96,
                  per_level_k=(192, 96, 48),  num_queries=200, num_layers=3),
    "small": dict(width=(32, 64, 128, 256, 512), depth=(1, 2, 2, 1), embed_dim=128,
                  per_level_k=(256, 160, 96), num_queries=300, num_layers=3),
    "base":  dict(width=(48, 96, 192, 384, 768), depth=(2, 3, 3, 2), embed_dim=192,
                  per_level_k=(384, 224, 128), num_queries=400, num_layers=6),
}


class ORSADet(nn.Module):
    def __init__(self, num_classes: int = 15, scale: str = "small",
                 aux_query_groups: int = 1, use_aux_dense: bool = True, use_ste: bool = True):
        super().__init__()
        cfg = PRESETS[scale]
        self.scale = scale
        self.num_classes = num_classes

        self.backbone = CSPBackbone(cfg["width"], cfg["depth"])
        self.neck = PANLite(self.backbone.out_channels, embed_dim=cfg["embed_dim"])
        self.lem = LEMStack(cfg["embed_dim"], apply_to=(0, 1))
        self.token_bank = SparseTokenBank(cfg["embed_dim"], num_levels=3,
                                          per_level_k=cfg["per_level_k"], use_ste=use_ste)
        self.head = QueryLiteHead(cfg["embed_dim"], num_classes, cfg["num_queries"],
                                  cfg["num_layers"], aux_query_groups=aux_query_groups)
        self.aux_dense = AuxDenseHead(cfg["embed_dim"], num_classes) if use_aux_dense else None

    def forward(self, x: torch.Tensor) -> Dict:
        feats = self.backbone(x)
        feats = self.neck(feats)
        feats = self.lem(feats)
        bank = self.token_bank(feats)
        out = self.head(bank)
        out["dense_scores"] = bank.dense_scores      # for survival loss
        out["grids"] = bank.grids
        out["ref_points"] = bank.ref_points
        if self.training and self.aux_dense is not None:
            out["aux_dense"] = self.aux_dense(list(feats))
        return out

    def fuse_for_inference(self):
        """Drop train-only branches; call before export."""
        self.aux_dense = None
        self.head.aux_query_groups = 0
        self.eval()
        return self


def build_model(num_classes=15, scale="small", **kw) -> ORSADet:
    return ORSADet(num_classes=num_classes, scale=scale, **kw)
