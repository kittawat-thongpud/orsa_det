"""Attention helpers: 2D sine positional embedding + optional deformable
sampling over dense pyramids (ablation A-variant). The default decoder uses
standard cross-attention to the sparse token bank (see head.py); this module
provides the building blocks and an export-friendly deformable sampler.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_pos_embed(ref_points: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """Map normalized (x,y) -> [.., dim] sine embedding. dim must be even."""
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    dim_t = torch.arange(half, device=ref_points.device, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / half)
    x = ref_points[..., 0:1] * 2 * math.pi
    y = ref_points[..., 1:2] * 2 * math.pi
    px = x / dim_t
    py = y / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=-1).flatten(-2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([py, px], dim=-1)  # [.., dim]


class DeformableSampler(nn.Module):
    """Export-friendly deformable sampling over a single dense feature map.

    Queries predict `n_points` 2D offsets around their reference point; features
    are bilinearly sampled (grid_sample) and attention-weighted. Used only when
    the deformable ablation is enabled.
    """

    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.n_heads, self.n_points = n_heads, n_points
        self.offsets = nn.Linear(dim, n_heads * n_points * 2)
        self.attn = nn.Linear(dim, n_heads * n_points)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self._reset()

    def _reset(self):
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, query, ref_points, feat_map):
        # query [B,Q,C]; ref_points [B,Q,2]; feat_map [B,C,H,W]
        B, Q, C = query.shape
        v = self.value(feat_map.flatten(2).transpose(1, 2))          # [B,HW,C]
        H, W = feat_map.shape[-2:]
        v = v.transpose(1, 2).reshape(B, C, H, W)
        off = self.offsets(query).view(B, Q, self.n_heads, self.n_points, 2)
        aw = self.attn(query).view(B, Q, self.n_heads, self.n_points).softmax(-1)
        loc = ref_points.view(B, Q, 1, 1, 2) + off * 0.1             # [B,Q,h,p,2] in [0,1]
        grid = (loc * 2 - 1).clamp(-1, 1).view(B, Q * self.n_heads * self.n_points, 1, 2)
        sampled = F.grid_sample(v, grid, mode="bilinear", align_corners=False)  # [B,C,Q*h*p,1]
        sampled = sampled.view(B, C, Q, self.n_heads, self.n_points)
        sampled = sampled.permute(0, 2, 3, 4, 1)                     # [B,Q,h,p,C]
        out = (sampled * aw.unsqueeze(-1)).sum(3).mean(2)            # [B,Q,C]
        return self.out(out)
