"""Probe: what YOLO model cfgs ship in this ultralytics, and how well does a
YOLOv8 backbone state_dict shape-match ORSA-Det's CSPBackbone (transfer feasibility).
Run: PYTHONPATH=. ./.venv/Scripts/python.exe tools/_probe_yolo_transfer.py
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from orsa.models.backbone import CSPBackbone

print("ultralytics", ultralytics.__version__)

# 1) which model cfgs exist (look for yolo26 / v8 / v11 ...) -------------------
cfg_root = Path(ultralytics.__file__).parent / "cfg" / "models"
yamls = sorted(p.name for p in cfg_root.rglob("*.yaml"))
fams = sorted({n.split(".")[0].rstrip("nslmx") for n in yamls})
print("model families:", fams)
print("has yolo26:", any("26" in n for n in yamls), "| has v8:", any("v8" in n for n in yamls))

# 2) build a YOLOv8n detection model (arch only, no download) -----------------
def backbone_convs(state):
    """conv weight tensors whose key sits in the backbone (model.0..model.9)."""
    out = []
    for k, v in state.items():
        if k.endswith(".conv.weight") and v.ndim == 4:
            out.append((k, tuple(v.shape)))
    return out

yolo = DetectionModel(cfg="yolov8n.yaml")  # random init, structure only
ystate = yolo.state_dict()
yconv = backbone_convs(ystate)
print(f"\nYOLOv8n conv layers (conv.weight, 4D): {len(yconv)}")
for k, s in yconv[:14]:
    print(f"  {k:38s} {s}")

# 3) ORSA CSPBackbone (nano + small) conv shapes ------------------------------
for scale, (w, d) in {"nano": ((16,32,64,128,256),(1,1,1,1)),
                      "small": ((32,64,128,256,512),(1,2,2,1))}.items():
    bb = CSPBackbone(w, d)
    bstate = bb.state_dict()
    bconv = [(k, tuple(v.shape)) for k, v in bstate.items()
             if k.endswith(".conv.weight") and v.ndim == 4]
    print(f"\nORSA CSPBackbone '{scale}': {len(bconv)} conv layers; first 14:")
    for k, s in bconv[:14]:
        print(f"  {k:46s} {s}")

    # shape-only greedy match against yolov8n backbone convs
    yshapes = [s for _, s in yconv]
    matched = sum(1 for _, s in bconv if s in yshapes)
    print(f"  -> shape-compatible (exists somewhere in v8n): {matched}/{len(bconv)}")
