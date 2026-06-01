"""Ultralytics-style reporting plots (matplotlib, no seaborn).

All functions write a PNG to disk and return the path. Headless-safe (Agg).
  - plot_class_distribution : dataset class-density histogram
  - plot_training_curves    : loss components + val mAP vs epoch (from metrics.jsonl)
  - plot_confusion_matrix   : confusion matrix heatmap (raw or normalized)
  - plot_per_class_ap       : per-class AP / AP50 bar chart
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short(names, n=None):
    n = n if n is not None else len(names)
    return [str(x)[:12] for x in names[:n]]


def plot_class_distribution(counts, names, out, title="dataset class density"):
    """counts: list[int] per class. names: list[str]."""
    out = Path(out)
    counts = list(counts)
    names = _short(names, len(counts))
    n = len(counts)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.28), 5))
    ax.bar(range(n), counts, color="#3b7dd8")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("instances")
    ax.set_title(f"{title}  (total={int(sum(counts))})")
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _read_jsonl(path):
    steps, evals = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (evals if r.get("eval") else steps).append(r)
    return steps, evals


def plot_training_curves(jsonl_path, out):
    """Plot per-step loss components (epoch-averaged) and val metrics vs epoch."""
    out = Path(out)
    steps, evals = _read_jsonl(jsonl_path)
    if not steps:
        return None
    loss_keys = sorted({k for r in steps for k in r
                        if k.startswith("loss") or k == "loss"})
    # epoch-average each loss key
    epochs = sorted({r["epoch"] for r in steps})
    curves = {k: [] for k in loss_keys}
    xs = []
    for e in epochs:
        rows = [r for r in steps if r["epoch"] == e]
        xs.append(e)
        for k in loss_keys:
            vals = [r[k] for r in rows if k in r]
            curves[k].append(float(np.mean(vals)) if vals else np.nan)

    metric_keys = ["mAP50", "mAP50-95"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k in loss_keys:
        axes[0].plot(xs, curves[k], label=k, linewidth=1.4)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("training loss (epoch mean)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    if evals:
        ex = [r["epoch"] for r in evals]
        for mk in metric_keys:
            if any(mk in r for r in evals):
                axes[1].plot(ex, [r.get(mk, np.nan) for r in evals],
                             marker="o", label=mk, linewidth=1.6)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mAP")
    axes[1].set_title("validation mAP")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_confusion_matrix(matrix, names, out, normalize=False):
    """matrix: (nc+1, nc+1) ndarray (rows=pred, cols=gt, last=background)."""
    out = Path(out)
    m = np.asarray(matrix, dtype=np.float64)
    nc = m.shape[0] - 1
    labels = _short(names, nc) + ["background"]
    if normalize:
        col = m.sum(0, keepdims=True)
        m = m / np.clip(col, 1e-9, None)
    fig, ax = plt.subplots(figsize=(max(6, nc * 0.5), max(5, nc * 0.45)))
    im = ax.imshow(m, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(nc + 1)); ax.set_yticks(range(nc + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    if nc <= 30:                                     # annotate only if readable
        thr = m.max() / 2.0 if m.max() else 0.5
        for i in range(nc + 1):
            for j in range(nc + 1):
                v = m[i, j]
                if v:
                    ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                            ha="center", va="center", fontsize=6,
                            color="white" if v > thr else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_per_class_ap(per_class_ap, per_class_ap50, names, out):
    """per_class_ap / per_class_ap50: list per class (may contain NaN)."""
    out = Path(out)
    n = len(per_class_ap)
    names = _short(names, n)
    ap = np.nan_to_num(np.asarray(per_class_ap, dtype=np.float64))
    ap50 = np.nan_to_num(np.asarray(per_class_ap50, dtype=np.float64))
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.3), 5))
    ax.bar(x - 0.2, ap50, 0.4, label="AP50", color="#3b7dd8")
    ax.bar(x + 0.2, ap, 0.4, label="AP50-95", color="#e8743b")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("AP"); ax.set_ylim(0, 1)
    mean_ap = float(np.nanmean([v for v in per_class_ap if v == v])) if n else 0.0
    ax.set_title(f"per-class AP  (mean AP50-95={mean_ap:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)
