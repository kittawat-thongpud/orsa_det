"""Export ORSA-Det to ONNX (static K, inference graph only).

Usage:
  python scripts/export_onnx.py --cfg configs/orsa_small_idd.yaml \
      --ckpt runs/orsa_small_idd/best.pth --out orsa_small.onnx --imgsz 512

Static top-K token bank -> fixed shapes -> TensorRT FP16/INT8 friendly.
Train-only aux branches are dropped via fuse_for_inference().
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model


class InferWrapper(nn.Module):
    """Return only (pred_logits, pred_boxes) -> clean ONNX I/O."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="orsa_det.onnx")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--use-ema", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    d, m = cfg["dataset"], cfg["model"]

    model = build_model(num_classes=d["num_classes"], scale=m["scale"],
                        aux_query_groups=0, use_aux_dense=False, use_ste=m["use_ste"])
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck["ema"] if (args.use_ema and ck.get("ema")) else ck["model"]
        model.load_state_dict(sd, strict=False)
        print(f"loaded ckpt: {args.ckpt}")
    model.fuse_for_inference()

    wrapper = InferWrapper(model).eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)

    # sanity forward
    with torch.no_grad():
        lg, bx = wrapper(dummy)
    print(f"pred_logits={tuple(lg.shape)} pred_boxes={tuple(bx.shape)}")

    torch.onnx.export(
        wrapper, dummy, args.out,
        input_names=["images"], output_names=["pred_logits", "pred_boxes"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch"},
                      "pred_logits": {0: "batch"}, "pred_boxes": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"exported -> {args.out}")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(args.out))
        print("onnx.checker: OK")
    except ImportError:
        print("onnx not installed; skipped checker")


if __name__ == "__main__":
    main()
