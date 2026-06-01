"""Logging: rich console + TensorBoard + JSONL metrics file + run dir mgmt."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None
try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
except Exception:  # pragma: no cover
    _console = None


class RunLogger:
    def __init__(self, run_dir: str, name: str = "orsa_det"):
        # each run isolated in a timestamped subdir: <run_dir>/<name>_<YYYYmmdd-HHMMSS>
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(run_dir) / f"{name}_{stamp}"
        (self.dir / "ckpt").mkdir(parents=True, exist_ok=True)
        print(f"[run] {self.dir}")
        self.tb = SummaryWriter(str(self.dir / "tb")) if SummaryWriter else None
        self.jsonl = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        self.name = name
        self.t0 = time.time()

    def log_scalars(self, tag: str, scalars: dict, step: int):
        if self.tb:
            for k, v in scalars.items():
                self.tb.add_scalar(f"{tag}/{k}", float(v), step)

    def log_step(self, epoch, step, total_steps, lr, losses: dict):
        rec = {"t": round(time.time() - self.t0, 1), "epoch": epoch, "step": step,
               "lr": lr, **{k: round(float(v), 4) for k, v in losses.items()}}
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        self.log_scalars("train", {"lr": lr, **losses}, epoch * total_steps + step)
        if _console and step % 20 == 0:
            _console.print(f"[cyan]e{epoch}[/] {step}/{total_steps} "
                           f"lr={lr:.2e} " +
                           " ".join(f"{k}={float(v):.3f}" for k, v in losses.items()))

    def log_eval(self, epoch, metrics: dict):
        rec = {"t": round(time.time() - self.t0, 1), "epoch": epoch, "eval": True,
               **{k: round(float(v), 4) for k, v in metrics.items()}}
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        self.log_scalars("val", metrics, epoch)
        if _console:
            t = Table(title=f"eval @ epoch {epoch}")
            t.add_column("metric"); t.add_column("value", justify="right")
            for k, v in metrics.items():
                t.add_row(k, f"{float(v):.4f}")
            _console.print(t)

    def save_ckpt(self, state: dict, name: str):
        path = self.dir / "ckpt" / name
        torch.save(state, path)
        return str(path)

    def close(self):
        if self.tb:
            self.tb.close()
        self.jsonl.close()
