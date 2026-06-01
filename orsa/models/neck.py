"""PAN-lite neck. Fuses P3/P4/P5 and projects all scales to embed_dim."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNAct, C2f


class PANLite(nn.Module):
    """Top-down + bottom-up fusion, then 1x1 project each scale to embed_dim."""

    def __init__(self, in_channels=(128, 256, 512), embed_dim=128, n=1):
        super().__init__()
        c3, c4, c5 = in_channels

        # top-down
        self.reduce_c5 = ConvBNAct(c5, c4, 1, 1)
        self.td_c4 = C2f(c4 + c4, c4, n=n, shortcut=False)
        self.reduce_c4 = ConvBNAct(c4, c3, 1, 1)
        self.td_c3 = C2f(c3 + c3, c3, n=n, shortcut=False)

        # bottom-up
        self.down_c3 = ConvBNAct(c3, c3, 3, 2)
        self.bu_c4 = C2f(c3 + c4, c4, n=n, shortcut=False)
        self.down_c4 = ConvBNAct(c4, c4, 3, 2)
        self.bu_c5 = C2f(c4 + c4, c5, n=n, shortcut=False)

        # unify to embed_dim
        self.proj3 = ConvBNAct(c3, embed_dim, 1, 1)
        self.proj4 = ConvBNAct(c4, embed_dim, 1, 1)
        self.proj5 = ConvBNAct(c5, embed_dim, 1, 1)
        self.embed_dim = embed_dim

    def forward(self, feats):
        p3, p4, p5 = feats
        # top-down
        u5 = self.reduce_c5(p5)
        p4 = self.td_c4(torch.cat([F.interpolate(u5, size=p4.shape[-2:], mode="nearest"), p4], 1))
        u4 = self.reduce_c4(p4)
        p3 = self.td_c3(torch.cat([F.interpolate(u4, size=p3.shape[-2:], mode="nearest"), p3], 1))
        # bottom-up
        p4 = self.bu_c4(torch.cat([self.down_c3(p3), p4], 1))
        p5 = self.bu_c5(torch.cat([self.down_c4(p4), u5], 1))
        # project
        return self.proj3(p3), self.proj4(p4), self.proj5(p5)
