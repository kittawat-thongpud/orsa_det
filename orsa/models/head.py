"""Query-lite detection head + auxiliary heads.

Main head : N learned queries with learnable reference points (DAB-style),
            cross-attend to the sparse token bank, iterative box refinement.
            One-to-one (Hungarian) matching -> NMS-free.
Aux heads (train-only, removed at inference):
  - aux query group : extra queries, one-to-many (Group-DETR style).
  - aux dense head   : per-location cls+box on dense pyramid, one-to-many (TAL).
"""
from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn
from .blocks import ConvBNAct
from .deformable import sine_pos_embed


class MLP(nn.Module):
    def __init__(self, dim, hidden, out, layers=3):
        super().__init__()
        h = [hidden] * (layers - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([dim] + h, h + [out]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = torch.relu(l(x)) if i < len(self.layers) - 1 else l(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads=8, ffn=512, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, q, q_pos, tokens, tok_pos):
        qk = q + q_pos
        q = q + self.self_attn(qk, qk, q)[0]
        q = self.n1(q)
        q = q + self.cross_attn(q + q_pos, tokens + tok_pos, tokens)[0]
        q = self.n2(q)
        q = q + self.ffn(q)
        return self.n3(q)


class QueryLiteHead(nn.Module):
    def __init__(self, embed_dim=128, num_classes=15, num_queries=300, num_layers=3,
                 n_heads=8, ffn=512, aux_query_groups=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_query_groups = aux_query_groups
        total_q = num_queries * (1 + aux_query_groups)

        self.query_embed = nn.Embedding(total_q, embed_dim)
        self.ref_points = nn.Embedding(total_q, 2)        # learnable anchor centers (sigmoid)
        nn.init.uniform_(self.ref_points.weight, 0.0, 1.0)

        self.layers = nn.ModuleList(DecoderLayer(embed_dim, n_heads, ffn) for _ in range(num_layers))
        self.embed_dim = embed_dim

        # prediction heads (shared across layers; aux deep supervision per layer)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.box_head = MLP(embed_dim, embed_dim, 4, layers=3)
        nn.init.constant_(self.box_head.layers[-1].bias, 0.0)
        # focal prior: P(fg)=0.01 at init -> stable early training (RetinaNet/DETR)
        prior = float(-math.log((1 - 0.01) / 0.01))
        nn.init.constant_(self.class_head.bias, prior)

    def forward(self, bank, return_aux=True):
        B = bank.tokens.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)        # [B,Tq,C]
        ref = self.ref_points.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)  # [B,Tq,2]
        tok_pos = sine_pos_embed(bank.ref_points, self.embed_dim)

        aux_outputs = []
        for layer in self.layers:
            q_pos = sine_pos_embed(ref, self.embed_dim)
            q = layer(q, q_pos, bank.tokens, tok_pos)
            if return_aux:
                aux_outputs.append(self._predict(q, ref))
        out = self._predict(q, ref)

        # split main vs aux-query-group along query dim
        nq = self.num_queries
        result = {"pred_logits": out["pred_logits"][:, :nq],
                  "pred_boxes": out["pred_boxes"][:, :nq]}
        if self.training and return_aux:
            result["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, :nq],
                                      "pred_boxes": a["pred_boxes"][:, :nq]} for a in aux_outputs[:-1]]
            if self.aux_query_groups > 0:
                result["aux_group"] = {"pred_logits": out["pred_logits"][:, nq:],
                                       "pred_boxes": out["pred_boxes"][:, nq:]}
        return result

    def _predict(self, q, ref):
        logits = self.class_head(q)
        delta = self.box_head(q)                              # [B,Tq,4] -> (dx,dy,dw,dh)
        ref_logit = torch.logit(ref.clamp(1e-4, 1 - 1e-4))    # DAB-style: refine in logit space
        cxcy = (ref_logit + delta[..., :2]).sigmoid()
        wh = delta[..., 2:].sigmoid()
        boxes = torch.cat([cxcy, wh], dim=-1)                 # cxcywh, normalized [0,1]
        return {"pred_logits": logits, "pred_boxes": boxes}


class AuxDenseHead(nn.Module):
    """Per-location cls + box (cxcywh) on each dense pyramid scale. One-to-many (TAL)."""

    def __init__(self, embed_dim=128, num_classes=15, num_levels=3):
        super().__init__()
        self.cls = nn.ModuleList(nn.Sequential(ConvBNAct(embed_dim, embed_dim, 3, 1),
                                               nn.Conv2d(embed_dim, num_classes, 1)) for _ in range(num_levels))
        self.reg = nn.ModuleList(nn.Sequential(ConvBNAct(embed_dim, embed_dim, 3, 1),
                                               nn.Conv2d(embed_dim, 4, 1)) for _ in range(num_levels))
        prior = float(-math.log((1 - 0.01) / 0.01))  # focal prior on dense cls
        for m in self.cls:
            nn.init.constant_(m[-1].bias, prior)

    def forward(self, feats: List[torch.Tensor]):
        outs = []
        for i, f in enumerate(feats):
            logits = self.cls[i](f)
            boxes = self.reg[i](f).sigmoid()
            outs.append({"logits": logits, "boxes": boxes})
        return outs
