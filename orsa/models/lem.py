"""Local Evidence Module (LEM).

Occlusion-robust local context aggregation on high-res scales (P3/P4).
Pure conv (DWConv + bottleneck + multi-dilation) — NO full self-attention,
so it stays cheap and TensorRT-friendly. Gathers "evidence" from neighbours
to recover partially-occluded objects before token scoring.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import ConvBNAct, DWConv


class LocalEvidenceModule(nn.Module):
    def __init__(self, dim: int, dilations=(1, 2, 3), e: float = 0.5):
        super().__init__()
        c_ = int(dim * e)
        self.proj_in = ConvBNAct(dim, c_, 1, 1)
        # multi-dilation depthwise branches capture local evidence at varied ranges
        self.branches = nn.ModuleList(
            nn.Conv2d(c_, c_, 3, 1, padding=d, dilation=d, groups=c_, bias=False) for d in dilations
        )
        self.bn = nn.BatchNorm2d(c_ * len(dilations))
        self.act = nn.SiLU(inplace=True)
        self.fuse = ConvBNAct(c_ * len(dilations), dim, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  # residual gate, init 0

    def forward(self, x):
        y = self.proj_in(x)
        y = torch.cat([b(y) for b in self.branches], dim=1)
        y = self.act(self.bn(y))
        y = self.fuse(y)
        return x + self.gamma * y  # gated residual; identity at init for stable bring-up


class LEMStack(nn.Module):
    """Apply LEM to selected scales (default P3, P4). P5 left untouched (cheap)."""

    def __init__(self, embed_dim: int, apply_to=(0, 1)):
        super().__init__()
        self.apply_to = set(apply_to)
        self.mods = nn.ModuleDict({str(i): LocalEvidenceModule(embed_dim) for i in apply_to})

    def forward(self, feats):
        out = []
        for i, f in enumerate(feats):
            out.append(self.mods[str(i)](f) if i in self.apply_to else f)
        return tuple(out)
