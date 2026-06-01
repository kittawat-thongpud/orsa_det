"""Reusable conv blocks for ORSA-Det. All export-friendly (no dynamic shapes)."""
from __future__ import annotations
import torch
import torch.nn as nn


def autopad(k: int, d: int = 1) -> int:
    return d * (k - 1) // 2


class ConvBNAct(nn.Module):
    """Conv2d -> BN -> SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, d: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(ConvBNAct):
    """Depthwise separable conv (depthwise k x k + pointwise 1x1)."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        # depthwise
        super().__init__(c1, c1, k=k, s=s, g=c1, act=act)
        self.pw = ConvBNAct(c1, c2, k=1, s=1, act=act)

    def forward(self, x):
        return self.pw(super().forward(x))


class Bottleneck(nn.Module):
    """Standard residual bottleneck (1x1 -> 3x3)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP block with 2 convs and n bottlenecks (YOLOv8-style, light)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
