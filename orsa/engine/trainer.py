"""Training loop: AMP bf16, grad accumulation, grad clip, multi-optimizer,
EMA, periodic eval + checkpoint. Tuned for 8GB VRAM (RTX 4070 Laptop)."""
from __future__ import annotations
import math
from copy import deepcopy
from dataclasses import dataclass, fields
import torch
from tqdm import tqdm
from .optim import build_optimizer, OptimConfig
from .evaluator import evaluate
from ..utils.metrics import ConfusionMatrix
from ..utils.report import generate_reports


@dataclass(frozen=True)
class SchedConfig:
    """LR warmup + cosine schedule knobs (no magic numbers in the loop)."""
    warmup_ratio: float = 0.05      # warmup = ratio * total_steps ...
    warmup_max_steps: int = 1000    # ... capped at this many steps
    cosine_min_ratio: float = 0.01  # floor lr = base * this at end of cosine

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class EMAConfig:
    """Model-EMA knobs."""
    enabled: bool = True
    decay: float = 0.9998
    tau: float = 2000.0             # decay-warmup time constant

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


class ModelEMA:
    """EMA with Ultralytics-style decay warmup: decay ramps from ~0 to `decay`
    as decay_t = decay * (1 - exp(-updates / tau)). Early updates track the raw
    model closely (avoids the cold-start where a high fixed decay keeps EMA near
    random init for thousands of steps); late updates use the full decay."""

    def __init__(self, model, decay=0.9998, tau=2000):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)


def _opt_step(opts, model, grad_clip):
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for o in opts:
        if o is not None:
            o.step()
    for o in opts:
        if o is not None:
            o.zero_grad(set_to_none=True)


def cosine_lr(step, total, warmup, base, min_ratio=0.01):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def train(model, criterion, train_loader, val_loader, logger, *, device="cuda",
          epochs=100, accum=8, grad_clip=1.0,
          amp_dtype=torch.bfloat16, eval_every=5, num_classes=15,
          opt_cfg: OptimConfig | None = None, sched_cfg: SchedConfig | None = None,
          ema_cfg: EMAConfig | None = None,
          do_train=True, do_val=True, make_report=True, names=None,
          class_counts=None):
    """do_train / do_val toggle the train and validation phases. make_report
    writes Ultralytics-style PNGs (curves, confusion matrix, per-class AP) to
    the run dir when validation is enabled.

    opt_cfg / sched_cfg / ema_cfg carry all optimizer, schedule and EMA
    hyperparameters (config-driven; defaults applied when None)."""
    opt_cfg = opt_cfg if opt_cfg is not None else OptimConfig()
    sched_cfg = sched_cfg if sched_cfg is not None else SchedConfig()
    ema_cfg = ema_cfg if ema_cfg is not None else EMAConfig()

    model.to(device)
    opts = build_optimizer(model, cfg=opt_cfg)
    opt_list = [o for o in opts if o is not None]
    base_lrs = [g["lr"] for o in opt_list for g in o.param_groups]
    model_ema = (ModelEMA(model, decay=ema_cfg.decay, tau=ema_cfg.tau)
                 if (ema_cfg.enabled and do_train) else None)

    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = min(sched_cfg.warmup_max_steps, int(total_steps * sched_cfg.warmup_ratio))
    best = -1.0
    last_metrics = None

    for epoch in range(epochs if do_train else 0):
        model.train()
        for o in opt_list:
            o.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"train e{epoch}", leave=False)
        for step, (imgs, targets) in enumerate(pbar):
            gstep = epoch * steps_per_epoch + step
            scale = cosine_lr(gstep, total_steps, warmup, 1.0,
                              min_ratio=sched_cfg.cosine_min_ratio)
            for o, bl in zip([g for o in opt_list for g in o.param_groups], base_lrs):
                o["lr"] = bl * scale

            imgs = imgs.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=device != "cpu"):
                out = model(imgs)
                loss, logs = criterion(out, targets)
                loss = loss / accum
            loss.backward()

            if (step + 1) % accum == 0:
                _opt_step(opt_list, model, grad_clip)
                if model_ema:
                    model_ema.update(model)
            cur_lr = opt_list[0].param_groups[0]["lr"]
            logger.log_step(epoch, step, steps_per_epoch, cur_lr, logs)

        if do_val and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            target_model = model_ema.ema if model_ema else model
            metrics = evaluate(target_model, val_loader, device, num_classes, amp_dtype)
            last_metrics = metrics
            logger.log_eval(epoch, metrics)
            ck = {"model": model.state_dict(),
                  "ema": model_ema.ema.state_dict() if model_ema else None,
                  "epoch": epoch, "metrics": metrics}
            logger.save_ckpt(ck, "last.pth")
            if metrics["mAP50-95"] > best:
                best = metrics["mAP50-95"]
                logger.save_ckpt(ck, "best.pth")

    # ---- final Ultralytics-style report ------------------------------------
    if do_val and make_report:
        target_model = model_ema.ema if model_ema else model
        nm = names if names is not None else [str(i) for i in range(num_classes)]
        cm = ConfusionMatrix(num_classes)
        metrics, pc_ap, pc_ap50 = evaluate(
            target_model, val_loader, device, num_classes, amp_dtype,
            confusion=cm, return_per_class=True)
        last_metrics = metrics
        made = generate_reports(
            logger.dir, nm, jsonl_path=logger.dir / "metrics.jsonl",
            confusion=cm, per_class_ap=pc_ap, per_class_ap50=pc_ap50,
            class_counts=class_counts, metrics=metrics)
        print(f"[report] {len(made)} artifacts -> {logger.dir}")
    return best
