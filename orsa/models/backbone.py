"""CNN-first backbone for ORSA-Det. Outputs P3 (/8), P4 (/16), P5 (/32)."""
from __future__ import annotations
import torch.nn as nn
from .blocks import ConvBNAct, C2f, SPPF


class CSPBackbone(nn.Module):
    """Lightweight CSP backbone. width/depth scaled by config.

    Channels per scale stored in .out_channels = (c3, c4, c5).
    """

    def __init__(self, width=(32, 64, 128, 256, 512), depth=(1, 2, 2, 1)):
        super().__init__()
        w0, w1, w2, w3, w4 = width
        d1, d2, d3, d4 = depth

        self.stem = ConvBNAct(3, w0, k=3, s=2)            # /2
        self.stage1 = nn.Sequential(ConvBNAct(w0, w1, 3, 2), C2f(w1, w1, n=d1))   # /4
        self.stage2 = nn.Sequential(ConvBNAct(w1, w2, 3, 2), C2f(w2, w2, n=d2))   # /8  -> P3
        self.stage3 = nn.Sequential(ConvBNAct(w2, w3, 3, 2), C2f(w3, w3, n=d3))   # /16 -> P4
        self.stage4 = nn.Sequential(ConvBNAct(w3, w4, 3, 2), C2f(w4, w4, n=d4), SPPF(w4, w4))  # /32 -> P5

        self.out_channels = (w2, w3, w4)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
