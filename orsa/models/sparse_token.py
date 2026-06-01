"""Adaptive Sparse Token Bank.

Scores every spatial location per scale, then keeps a FIXED per-scale top-K
(static shape -> TensorRT-friendly). A learned soft gate (sigmoid) modulates
each surviving token; gradients reach the scorer through the gate (optional
straight-through estimator). Dense score maps are returned for the Token
Survival Loss, which prevents "sparse starvation" of small/occluded objects.

Returned tokens are the keys/values the query-lite head attends to.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
from .blocks import ConvBNAct


@dataclass
class TokenBankOutput:
    tokens: torch.Tensor          # [B, K, C]   gathered + gated token features
    ref_points: torch.Tensor      # [B, K, 2]   normalized (x, y) in [0, 1]
    gates: torch.Tensor           # [B, K, 1]   sigmoid gate of survivors
    level_ids: torch.Tensor       # [K]         scale index per token
    dense_scores: List[torch.Tensor]   # per scale [B, H, W] sigmoid scores (for survival loss)
    grids: List[torch.Tensor]          # per scale [H*W, 2] normalized centers (cached)


class TokenScorer(nn.Module):
    """Per-location objectness/evidence score (1 logit)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(dim, dim, 3, 1, g=dim), nn.Conv2d(dim, 1, 1))

    def forward(self, x):
        return self.net(x)  # [B,1,H,W]


def _make_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [H*W, 2] (x,y)


class SparseTokenBank(nn.Module):
    def __init__(self, embed_dim: int, num_levels: int = 3, per_level_k=(256, 160, 96),
                 use_ste: bool = True):
        super().__init__()
        assert len(per_level_k) == num_levels
        self.per_level_k = list(per_level_k)
        self.use_ste = use_ste
        self.scorers = nn.ModuleList(TokenScorer(embed_dim) for _ in range(num_levels))

    def _gate(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_ste:
            return z
        hard = (z >= 0.5).to(z.dtype)
        return hard + (z - z.detach())  # forward: hard 0/1, backward: grad of z

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> TokenBankOutput:
        B = feats[0].shape[0]
        tokens_all, ref_all, gate_all, level_all = [], [], [], []
        dense_scores, grids = [], []

        for lvl, f in enumerate(feats):
            b, c, h, w = f.shape
            logit = self.scorers[lvl](f).flatten(2).squeeze(1)        # [B, H*W]
            z = torch.sigmoid(logit)                                  # [B, H*W]
            dense_scores.append(z.view(B, h, w))

            grid = _make_grid(h, w, f.device, f.dtype)                # [H*W, 2]
            grids.append(grid)

            k = min(self.per_level_k[lvl], h * w)
            topv, topi = torch.topk(z, k, dim=1)                      # [B, k]

            feat_flat = f.flatten(2).transpose(1, 2)                  # [B, H*W, C]
            idx_c = topi.unsqueeze(-1).expand(-1, -1, c)
            tok = torch.gather(feat_flat, 1, idx_c)                   # [B, k, C]

            gate = self._gate(topv).unsqueeze(-1)                     # [B, k, 1]
            tok = tok * gate                                          # gated tokens

            ref = grid[topi]                                          # [B, k, 2]

            tokens_all.append(tok)
            ref_all.append(ref)
            gate_all.append(gate)
            level_all.append(torch.full((k,), lvl, device=f.device, dtype=torch.long))

        return TokenBankOutput(
            tokens=torch.cat(tokens_all, 1),
            ref_points=torch.cat(ref_all, 1),
            gates=torch.cat(gate_all, 1),
            level_ids=torch.cat(level_all, 0),
            dense_scores=dense_scores,
            grids=grids,
        )
