"""Verify the Ultralytics-style reporting pipeline produces real artifacts.

Runs a short overfit on a COCO128 subset, then exercises EVERY reporting path:
  - log_init banner -> init.txt
  - class_density scan
  - per-class AP + confusion matrix via evaluate(..., confusion=, return_per_class=)
  - generate_reports -> all PNGs + csv + results.txt
Asserts each expected file exists and is non-empty.

Usage: python tools/test_report.py [--subset 24] [--steps 80]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.losses import SetCriterion
from orsa.data import YOLODetection, collate_fn
from orsa.engine.optim import build_optimizer
from orsa.engine.evaluator import evaluate
from orsa.utils import RunLogger, log_init, class_density, ConfusionMatrix
from orsa.utils.report import generate_reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/coco128")
    ap.add_argument("--class-map", default="configs/coco128_classes.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--subset", type=int, default=24)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--out", default="runs/_report_test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    amp = torch.bfloat16 if device == "cuda" else torch.float32

    ds = YOLODetection(args.root, "train2017", args.class_map, args.size, train=False)
    nc = ds.num_classes
    names = ds.classes
    sub = Subset(ds, list(range(min(args.subset, len(ds)))))
    loader = DataLoader(sub, batch_size=4, shuffle=True, num_workers=0,
                        collate_fn=collate_fn, drop_last=True)
    eval_loader = DataLoader(sub, batch_size=4, shuffle=False, num_workers=0,
                             collate_fn=collate_fn)

    model = build_model(num_classes=nc, scale="small", aux_query_groups=1,
                        use_aux_dense=True, use_ste=True).to(device)
    crit = SetCriterion(nc, lambda_surv=1.0, lambda_sparse=1e-3, lambda_dense=1.0)
    opt = build_optimizer(model, phase="A", lr=2e-4)[0]
    logger = RunLogger(args.out, "report_test")

    # ---- init banner -------------------------------------------------------
    cfg = {"dataset": {"root": args.root, "num_classes": nc, "img_size": args.size},
           "model": {"scale": "small"}, "train": {"phase": "A", "steps": args.steps}}
    init_txt = log_init(model, cfg, num_classes=nc, names=names,
                        train_n=len(sub), val_n=len(sub), out_dir=logger.dir,
                        phase="A", img_size=args.size)
    print(f"[ok] init -> {init_txt}")

    # ---- class density (subset only, fast) ---------------------------------
    counts = class_density(sub, nc)
    print(f"[ok] density: {sum(counts)} instances over {sum(c>0 for c in counts)} classes")

    # ---- overfit (writes metrics.jsonl via logger) -------------------------
    model.train()
    it = iter(loader)
    for step in range(args.steps):
        try:
            imgs, targets = next(it)
        except StopIteration:
            it = iter(loader); imgs, targets = next(it)
        imgs = imgs.to(device)
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                   for t in targets]
        with torch.autocast(device_type=device.split(":")[0], dtype=amp, enabled=device != "cpu"):
            out = model(imgs); loss, logs = crit(out, targets)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        logger.log_step(0, step, args.steps, 2e-4, logs)
        if step % 20 == 0:
            print(f"  step {step:3d} loss={loss.item():.3f}")
    # fake a couple eval rows so the val curve has points
    m_mid = evaluate(model, eval_loader, device, nc, amp_dtype=amp)
    logger.log_eval(0, m_mid)

    # ---- per-class AP + confusion matrix -----------------------------------
    cm = ConfusionMatrix(nc)
    metrics, pc_ap, pc_ap50 = evaluate(model, eval_loader, device, nc, amp_dtype=amp,
                                       confusion=cm, return_per_class=True)
    logger.log_eval(1, metrics)
    valid = sum(1 for a in pc_ap if a == a)
    print(f"[ok] eval mAP50={metrics['mAP50']:.4f} per-class-AP defined for {valid} classes")
    print(f"[ok] confusion matrix sum={int(cm.matrix.sum())}")

    made = generate_reports(logger.dir, names, jsonl_path=logger.dir / "metrics.jsonl",
                            confusion=cm, per_class_ap=pc_ap, per_class_ap50=pc_ap50,
                            class_counts=counts, metrics=metrics)
    logger.close()

    # ---- assert artifacts --------------------------------------------------
    expect = ["init.txt", "metrics.jsonl", "class_distribution.png",
              "training_curves.png", "confusion_matrix.png",
              "confusion_matrix_norm.png", "per_class_ap.png",
              "per_class_ap.csv", "results.txt"]
    miss = []
    for f in expect:
        p = logger.dir / f
        if not (p.exists() and p.stat().st_size > 0):
            miss.append(f)
    print(f"\nartifacts in {logger.dir}:")
    for f in expect:
        p = logger.dir / f
        flag = "OK " if (p.exists() and p.stat().st_size > 0) else "MISS"
        print(f"  [{flag}] {f}  ({p.stat().st_size if p.exists() else 0} B)")
    if miss:
        print(f"[FAIL] missing artifacts: {miss}")
    else:
        print("REPORT PIPELINE OK - all artifacts written")


if __name__ == "__main__":
    main()
