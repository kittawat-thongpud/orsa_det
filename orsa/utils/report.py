"""End-of-training report: writes all PNGs + a results summary to the run dir.

generate_reports(...) is called once after training (or standalone via
scripts/report.py). Each artifact is optional and guarded so a missing input
just skips that plot instead of crashing the run.
"""
from __future__ import annotations
from pathlib import Path
from . import plots


def generate_reports(out_dir, names, *, jsonl_path=None, confusion=None,
                     per_class_ap=None, per_class_ap50=None, class_counts=None,
                     metrics=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = {}

    if class_counts is not None:
        made["class_distribution"] = plots.plot_class_distribution(
            class_counts, names, out / "class_distribution.png")

    if jsonl_path and Path(jsonl_path).exists():
        p = plots.plot_training_curves(jsonl_path, out / "training_curves.png")
        if p:
            made["training_curves"] = p

    if confusion is not None:
        made["confusion_matrix"] = plots.plot_confusion_matrix(
            confusion.matrix, names, out / "confusion_matrix.png", normalize=False)
        made["confusion_matrix_norm"] = plots.plot_confusion_matrix(
            confusion.matrix, names, out / "confusion_matrix_norm.png", normalize=True)

    if per_class_ap is not None:
        ap50 = per_class_ap50 if per_class_ap50 is not None else [float("nan")] * len(per_class_ap)
        made["per_class_ap"] = plots.plot_per_class_ap(
            per_class_ap, ap50, names, out / "per_class_ap.png")
        _write_per_class_csv(out / "per_class_ap.csv", names, per_class_ap, ap50)

    if metrics is not None:
        _write_results(out / "results.txt", metrics, made)

    return made


def _write_per_class_csv(path, names, ap, ap50):
    lines = ["class,AP50,AP50-95"]
    for i, nm in enumerate(names[:len(ap)]):
        a = ap[i]; a5 = ap50[i]
        lines.append(f"{nm},{'' if a5 != a5 else f'{a5:.4f}'},{'' if a != a else f'{a:.4f}'}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_results(path, metrics, made):
    lines = ["ORSA-Det 2027 - final results", "=" * 40]
    for k, v in metrics.items():
        lines.append(f"{k:<12}= {float(v):.4f}")
    lines.append("-" * 40)
    lines.append("artifacts:")
    for k, p in made.items():
        lines.append(f"  {k}: {Path(p).name}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
