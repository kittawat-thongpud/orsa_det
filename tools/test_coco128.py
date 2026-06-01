"""End-to-end sanity on COCO128 (Ultralytics).

Proves the full pipeline LEARNS on real images:
  1. load YOLODetection (coco128, 80 cls)
  2. overfit a small subset for N steps (Phase A AdamW) -> loss must drop
  3. eval on that subset -> mAP must rise above 0

Usage:
  python tools/test_coco128.py [--root datasets/coco128] [--subset 16] [--steps 60]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.losses import SetCriterion
from orsa.data import YOLODetection, collate_fn
from orsa.engine.optim import build_optimizer
from orsa.engine.evaluator import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/coco128")
    ap.add_argument("--class-map", default="configs/coco128_classes.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--subset", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device={device} size={args.size} subset={args.subset} steps={args.steps}")

    root = Path(args.root)
    assert root.exists(), f"coco128 not found: {root.resolve()}"
    ds = YOLODetection(root, "train2017", args.class_map, args.size, train=False)
    nc = ds.num_classes
    print(f"[OK] dataset: {len(ds)} imgs, {nc} classes")

    sub = Subset(ds, list(range(min(args.subset, len(ds)))))
    loader = DataLoader(sub, batch_size=args.bs, shuffle=True, num_workers=0,
                        collate_fn=collate_fn, drop_last=True)

    model = build_model(num_classes=nc, scale="small", aux_query_groups=1,
                        use_aux_dense=True, use_ste=True).to(device)
    crit = SetCriterion(nc, lambda_surv=1.0, lambda_sparse=1e-3, lambda_dense=1.0)
    opt = build_optimizer(model, phase="A", lr=2e-4)[0]
    amp = torch.bfloat16 if device == "cuda" else torch.float32

    # ---- eval BEFORE training (baseline mAP, expect ~0) --------------------
    eval_loader = DataLoader(sub, batch_size=args.bs, shuffle=False, num_workers=0,
                             collate_fn=collate_fn)
    m0 = evaluate(model, eval_loader, device, nc, amp_dtype=amp)
    print(f"[before] mAP50={m0['mAP50']:.4f} mAP50-95={m0['mAP50-95']:.4f}")

    # ---- overfit -----------------------------------------------------------
    model.train()
    it = iter(loader)
    first = last = None
    for step in range(args.steps):
        try:
            imgs, targets = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, targets = next(it)
        imgs = imgs.to(device)
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                   for t in targets]
        with torch.autocast(device_type=device.split(":")[0], dtype=amp, enabled=device != "cpu"):
            out = model(imgs)
            loss, logs = crit(out, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss={loss.item():7.3f}  "
                  f"main={logs['loss_main']:.3f} dense={logs['loss_dense']:.3f}")

    # ---- eval AFTER training ----------------------------------------------
    m1 = evaluate(model, eval_loader, device, nc, amp_dtype=amp)
    print(f"[after ] mAP50={m1['mAP50']:.4f} mAP50-95={m1['mAP50-95']:.4f}")
    print(f"\nloss: {first:.3f} -> {last:.3f}  ({'DOWN' if last < first else 'UP'})")

    ok_loss = last < first * 0.7
    ok_map = m1["mAP50"] > m0["mAP50"]
    if ok_loss and ok_map:
        print("COCO128 LEARNS - PIPELINE OK")
    else:
        print(f"[WARN] ok_loss={ok_loss} ok_map={ok_map} "
              f"(loss {first:.2f}->{last:.2f}, mAP50 {m0['mAP50']:.3f}->{m1['mAP50']:.3f})")


if __name__ == "__main__":
    main()
