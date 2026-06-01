"""Startup banner + dataset class-density summary (Ultralytics-style init log).

log_init(...) prints config, top-level model components with param counts, and is
called once at train start to confirm the code wires up as intended.
class_density(...) counts instances per class over a dataset for the density plot.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
except Exception:  # pragma: no cover
    _console = None


def _fmt(n):
    return f"{n/1e6:.3f}M" if n >= 1e5 else f"{n/1e3:.1f}K" if n >= 1e3 else str(n)


def model_component_rows(model):
    """(name, #params, #trainable, dtype) for each top-level child module."""
    rows = []
    for name, mod in model.named_children():
        ps = list(mod.parameters())
        total = sum(p.numel() for p in ps)
        train = sum(p.numel() for p in ps if p.requires_grad)
        dt = str(ps[0].dtype).replace("torch.", "") if ps else "-"
        rows.append((name, total, train, dt))
    return rows


def class_density(dataset, num_classes):
    """Count instances per class id over dataset. Reads labels directly when the
    dataset exposes them, else falls back to iterating __getitem__."""
    counts = [0] * num_classes
    # fast path: YOLO/IDD expose parse via __getitem__ target labels
    for i in range(len(dataset)):
        try:
            _, target = dataset[i]
            labs = target["labels"]
        except Exception:
            continue
        for c in (labs.tolist() if torch.is_tensor(labs) else list(labs)):
            if 0 <= int(c) < num_classes:
                counts[int(c)] += 1
    return counts


def log_init(model, cfg, *, num_classes, names=None, train_n=None, val_n=None,
             out_dir=None, phase="A", img_size=None):
    """Print + persist the init banner. Returns the path of the saved txt (or None)."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = model_component_rows(model)
    lines = []
    lines.append("=" * 64)
    lines.append("ORSA-Det 2027  |  initialization")
    lines.append("=" * 64)
    lines.append(f"phase={phase}  num_classes={num_classes}  img_size={img_size}")
    if train_n is not None or val_n is not None:
        lines.append(f"train_imgs={train_n}  val_imgs={val_n}")
    lines.append(f"params total={_fmt(total)}  trainable={_fmt(train)}")
    lines.append("-" * 64)
    lines.append(f"{'component':<22}{'params':>12}{'trainable':>12}{'dtype':>10}")
    for nm, tp, tr, dt in rows:
        lines.append(f"{nm:<22}{_fmt(tp):>12}{_fmt(tr):>12}{dt:>10}")
    lines.append("-" * 64)
    lines.append("config:")
    for blk, sub in cfg.items():
        if isinstance(sub, dict):
            lines.append(f"  [{blk}]")
            for k, v in sub.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {blk}: {sub}")
    lines.append("=" * 64)
    text = "\n".join(lines)

    if _console:
        t = Table(title="ORSA-Det model components")
        t.add_column("component"); t.add_column("params", justify="right")
        t.add_column("trainable", justify="right"); t.add_column("dtype", justify="right")
        for nm, tp, tr, dt in rows:
            t.add_row(nm, _fmt(tp), _fmt(tr), dt)
        _console.print(f"[bold green]ORSA-Det 2027[/] init  phase={phase} "
                       f"classes={num_classes} params={_fmt(total)}")
        _console.print(t)
    else:
        print(text)

    if out_dir:
        p = Path(out_dir) / "init.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return str(p)
    return None
