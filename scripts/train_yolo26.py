"""Baseline: train YOLO26-N FROM SCRATCH on COCO128, official (paper) recipe.

Mirrors the ORSA-Det coco128 transfer test for apples-to-apples comparison:
  - same dataset, train==val (overfit / sanity demo, 128 fixed images)
  - imgsz 512, epochs 100
  - from scratch (yolo26n.yaml, NO pretrained .pt) -> pretrained=False
Otherwise uses Ultralytics' default YOLO training config (SGD, lr0, augment,
mosaic w/ close_mosaic) i.e. the recipe behind the paper baseline.

Run: PYTHONPATH=. ./.venv/Scripts/python.exe scripts/train_yolo26.py
"""
from __future__ import annotations
import json
from pathlib import Path
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "datasets" / "coco128"
NAMES = json.loads((ROOT / "configs" / "coco128_classes.json").read_text())["classes"]

# data yaml: train==val on coco128 (overfit sanity, matches ORSA test) ---------
DATA = ROOT / "configs" / "coco128_yolo.yaml"
DATA.write_text(_yaml.safe_dump({
    "path": str(COCO).replace("\\", "/"),
    "train": "images/train2017",
    "val": "images/train2017",
    "names": {i: n for i, n in enumerate(NAMES)},
}, sort_keys=False))

if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolo26n.yaml")          # from scratch (architecture only, random init)
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=512,
        batch=16,
        device=0,
        project=str(ROOT / "runs"),
        name="yolo26n_scratch_coco128",
        exist_ok=True,
        pretrained=False,                 # explicit: no COCO weights
        val=True,
        plots=True,
    )
