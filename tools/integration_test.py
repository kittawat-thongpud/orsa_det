"""Integration test on REAL IDD data.

Validates the full real-data pipeline before any training run:
  1. dataset load (XML parse, transforms) on actual train/val split
  2. DataLoader + collate_fn -> real batch
  3. train forward (aux on) -> SetCriterion -> backward (finite grads)
  4. optimizer step for BOTH Phase A (AdamW) and Phase B (MuSGD-hybrid)
  5. short evaluate() over a few val batches -> COCO metrics

Usage:
  python tools/integration_test.py [--root <IDD_Detection>] [--n 4] [--eval-batches 2]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.losses import SetCriterion
from orsa.data import IDDDetection, collate_fn
from orsa.engine.optim import build_optimizer
from orsa.engine.evaluator import evaluate


def _fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../IDD_Detection")
    ap.add_argument("--class-map", default="configs/idd_classes.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=4, help="batch size for the real batch")
    ap.add_argument("--eval-batches", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} size={args.size} batch={args.n}")

    # 1. real dataset --------------------------------------------------------
    root = Path(args.root)
    if not root.exists():
        _fail(f"IDD root not found: {root.resolve()}")
    train_ds = IDDDetection(root, "train.txt", args.class_map, args.size, train=True)
    val_ds = IDDDetection(root, "val.txt", args.class_map, args.size, train=False)
    nc = train_ds.num_classes
    print(f"[OK] dataset: train={len(train_ds)} val={len(val_ds)} classes={nc}")

    # probe a few real samples (XML parse + transform shape contract)
    n_with_boxes = 0
    for i in range(min(args.n, len(train_ds))):
        img, t = train_ds[i]
        if img.shape != (3, args.size, args.size):
            _fail(f"sample {i} bad img shape {tuple(img.shape)}")
        if t["boxes"].ndim != 2 or t["boxes"].shape[-1] != 4:
            _fail(f"sample {i} bad boxes shape {tuple(t['boxes'].shape)}")
        if t["boxes"].numel():
            b = t["boxes"]
            if not (b.min() >= 0 and b.max() <= 1.0001):
                _fail(f"sample {i} boxes not normalized [0,1]: min={b.min()} max={b.max()}")
            n_with_boxes += 1
    print(f"[OK] sample probe: {n_with_boxes}/{args.n} have GT boxes, shapes+norm valid")

    # 2. DataLoader + collate (num_workers=0 for deterministic test) ----------
    train_loader = DataLoader(Subset(train_ds, list(range(args.n))), batch_size=args.n,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)
    imgs, targets = next(iter(train_loader))
    assert imgs.shape[0] == args.n and len(targets) == args.n
    print(f"[OK] collate: imgs={tuple(imgs.shape)} targets={len(targets)}")

    # 3. train forward -> loss -> backward -----------------------------------
    model = build_model(num_classes=nc, scale="small", aux_query_groups=1,
                        use_aux_dense=True, use_ste=True).to(device)
    crit = SetCriterion(nc, lambda_surv=1.0, lambda_sparse=1e-3)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK] model built: {n_params:.2f}M params")

    model.train()
    imgs = imgs.to(device)
    targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
    out = model(imgs)
    loss, logs = crit(out, targets)
    if not torch.isfinite(loss):
        _fail(f"loss not finite: {loss}")
    loss.backward()
    n_p = n_g = n_bad = 0
    for p in model.parameters():
        if p.requires_grad:
            n_p += 1
            if p.grad is not None:
                n_g += 1
                if not torch.isfinite(p.grad).all():
                    n_bad += 1
    if n_bad:
        _fail(f"{n_bad} params have non-finite grads")
    log_str = " ".join(f"{k}={v:.3f}" for k, v in logs.items())
    print(f"[OK] train step: loss={loss.item():.3f} | grads {n_g}/{n_p} finite")
    print(f"      {log_str}")

    # 4. optimizer step: Phase A (AdamW) and Phase B (MuSGD-hybrid) -----------
    for phase in ("A", "B"):
        m2 = build_model(num_classes=nc, scale="small", aux_query_groups=1,
                         use_aux_dense=True, use_ste=True).to(device)
        opts = [o for o in build_optimizer(m2, phase=phase, lr=2e-4) if o is not None]
        m2.train()
        o2 = m2(imgs)
        l2, _ = crit(o2, targets)
        for o in opts:
            o.zero_grad()
        l2.backward()
        before = next(p for p in m2.parameters() if p.ndim >= 2).detach().clone().flatten()[:8]
        for o in opts:
            o.step()
        after = next(p for p in m2.parameters() if p.ndim >= 2).detach().flatten()[:8]
        moved = not torch.allclose(before, after)
        if not moved:
            _fail(f"phase {phase}: optimizer did not update weights")
        print(f"[OK] optimizer phase {phase}: {len(opts)} opt(s), weights updated")
        del m2

    # 5. short eval over a few val batches -----------------------------------
    val_loader = DataLoader(Subset(val_ds, list(range(args.n * args.eval_batches))),
                            batch_size=args.n, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
    amp = torch.bfloat16 if device == "cuda" else torch.float32
    t0 = time.time()
    metrics = evaluate(model, val_loader, device, nc, amp_dtype=amp,
                       max_batches=args.eval_batches)
    dt = time.time() - t0
    got = {k: round(float(v), 4) for k, v in metrics.items()}
    print(f"[OK] eval {args.eval_batches} batches in {dt:.1f}s -> {got}")

    print("\nINTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
