"""Eval a checkpoint. Usage: python scripts/eval.py --cfg configs/orsa_small_idd.yaml --ckpt runs/.../best.pth"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.data import IDDDetection, collate_fn
from orsa.engine import evaluate

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use-ema", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    d, m, tr = cfg["dataset"], cfg["model"], cfg["train"]

    val_ds = IDDDetection(d["root"], d["val_split"], d["class_map"], d["img_size"], train=False)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False,
                            num_workers=tr["workers"], collate_fn=collate_fn)
    model = build_model(num_classes=d["num_classes"], scale=m["scale"],
                        aux_query_groups=0, use_aux_dense=False, use_ste=m["use_ste"])
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["ema"] if (args.use_ema and ck.get("ema")) else ck["model"]
    model.load_state_dict(sd, strict=False)
    metrics = evaluate(model.to(tr["device"]), val_loader, tr["device"],
                       d["num_classes"], DTYPE[tr["amp_dtype"]])
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")


if __name__ == "__main__":
    main()
