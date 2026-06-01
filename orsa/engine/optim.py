"""Optimizers for ORSA-Det — config-driven selection (NASA-clean: no magic
numbers; every hyperparameter lives in `OptimConfig`).

Selectable optimizers (`OptimConfig.name`):
  "adamw"        : plain AdamW on all params (bring-up / debug).
  "sgd"          : SGD (+momentum, +nesterov) on all params.
  "muon_hybrid"  : MuSGD-hybrid (baseline-fair) =
      Muon  (Newton-Schulz orthogonalized momentum) on 2D weight matrices,
      SGD   (Nesterov) on norm + bias (1D),
      AdamW on embeddings / reference points / scalars (gamma gates).

`build_optimizer` always returns a 3-tuple of optimizers (None-padded) so the
trainer can iterate uniformly.

Muon ref: Jordan et al. 2024 (MomentUm Orthogonalized by Newton-Schulz).
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from torch.optim.optimizer import Optimizer


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimConfig:
    """All optimizer knobs. Defaults = pre-defined params; YAML overrides."""
    name: str = "adamw"             # adamw | sgd | muon_hybrid
    lr: float = 2e-4                # base lr (others scale off this)
    weight_decay: float = 1e-4
    # AdamW
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    # SGD (when name == "sgd")
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    # muon_hybrid: per-sub-optimizer lr multipliers + momenta
    muon_lr_mult: float = 100.0     # Muon wants a much larger lr
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    hybrid_sgd_lr_mult: float = 5.0
    hybrid_sgd_momentum: float = 0.9
    hybrid_sgd_nesterov: bool = True
    # Newton-Schulz quintic coefficients + numerical epsilon
    ns_a: float = 3.4445
    ns_b: float = -4.7750
    ns_c: float = 2.0315
    ns_eps: float = 1e-7

    @classmethod
    def from_dict(cls, d: dict | None) -> "OptimConfig":
        d = d or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise KeyError(f"unknown optimizer keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_phase(cls, phase: str, lr: float, weight_decay: float) -> "OptimConfig":
        """Legacy bridge: phase 'A' -> adamw, phase 'B' -> muon_hybrid."""
        name = "adamw" if phase.upper() == "A" else "muon_hybrid"
        return cls(name=name, lr=lr, weight_decay=weight_decay)


# ----------------------------------------------------------------------------
# Muon
# ----------------------------------------------------------------------------
@torch.no_grad()
def _zeropower_newtonschulz5(G: torch.Tensor, steps: int,
                             a: float, b: float, c: float, eps: float) -> torch.Tensor:
    """Quintic Newton-Schulz iteration -> approx orthogonalization of G (2D)."""
    assert G.ndim == 2
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, ns_a=3.4445, ns_b=-4.7750, ns_c=2.0315, ns_eps=1e-7):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay,
                                      ns_a=ns_a, ns_b=ns_b, ns_c=ns_c, ns_eps=ns_eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            mom, lr, wd = grp["momentum"], grp["lr"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if grp["nesterov"] else buf
                shp = upd.shape
                upd2d = upd.reshape(shp[0], -1)          # flatten conv kernel -> 2D
                upd2d = _zeropower_newtonschulz5(upd2d, grp["ns_steps"],
                                                 grp["ns_a"], grp["ns_b"],
                                                 grp["ns_c"], grp["ns_eps"])
                upd = upd2d.reshape(shp).to(p.dtype)
                scale = max(1.0, p.shape[0] / p[0].numel()) ** 0.5 if p.ndim > 1 else 1.0
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(upd, alpha=-lr * scale)
        return loss


def build_param_groups(model):
    """Split params: muon(2D weights), sgd(norm/bias 1D), adamw(embed/ref/scalar)."""
    muon, sgd, adamw = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("query_embed" in name or "ref_points" in name or "gamma" in name
                or name.endswith(".pos") or p.ndim == 1 and "embed" in name):
            adamw.append(p)
        elif p.ndim >= 2:
            muon.append(p)
        else:  # norm weight, bias (1D)
            sgd.append(p)
    return muon, sgd, adamw


def _adamw(params, cfg: OptimConfig, lr: float):
    return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay,
                             betas=(cfg.adamw_beta1, cfg.adamw_beta2), eps=cfg.adamw_eps)


def build_optimizer(model, phase=None, lr=None, weight_decay=None, cfg: OptimConfig | None = None):
    """Returns (opt0, opt1, opt2) with None padding.

    Preferred call: build_optimizer(model, cfg=OptimConfig(...)).
    Legacy call:    build_optimizer(model, phase="A", lr=2e-4) — still works.
    """
    if cfg is None:
        cfg = OptimConfig.from_phase(
            phase if phase is not None else "A",
            lr if lr is not None else OptimConfig.lr,
            weight_decay if weight_decay is not None else OptimConfig.weight_decay)

    name = cfg.name.lower()
    if name == "adamw":
        return _adamw(model.parameters(), cfg, cfg.lr), None, None
    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.sgd_momentum,
                              nesterov=cfg.sgd_nesterov, weight_decay=cfg.weight_decay)
        return opt, None, None
    if name == "muon_hybrid":
        muon_p, sgd_p, adamw_p = build_param_groups(model)
        opt_muon = Muon(muon_p, lr=cfg.lr * cfg.muon_lr_mult, momentum=cfg.muon_momentum,
                        nesterov=cfg.muon_nesterov, ns_steps=cfg.muon_ns_steps,
                        weight_decay=cfg.weight_decay,
                        ns_a=cfg.ns_a, ns_b=cfg.ns_b, ns_c=cfg.ns_c, ns_eps=cfg.ns_eps)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.lr * cfg.hybrid_sgd_lr_mult,
                                  momentum=cfg.hybrid_sgd_momentum,
                                  nesterov=cfg.hybrid_sgd_nesterov,
                                  weight_decay=cfg.weight_decay)
        opt_adamw = _adamw(adamw_p, cfg, cfg.lr)
        return opt_muon, opt_sgd, opt_adamw
    raise ValueError(f"unknown optimizer name: {cfg.name!r} "
                     f"(expected adamw | sgd | muon_hybrid)")
