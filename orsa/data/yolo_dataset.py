"""YOLO-format detection dataset (e.g. COCO128).

Layout (Ultralytics convention):
  root/images/<split>/<stem>.jpg
  root/labels/<split>/<stem>.txt   # each line: "cls cx cy w h" (normalized [0,1])

Returns the SAME contract as IDDDetection:
  (image[3,S,S] float, target dict{boxes cxcywh norm, labels, image_id, orig_size}).
YOLO boxes are normalized cxcywh; we expand to xyxy-abs-px so the shared
transforms.Compose (which expects xyxy px) handles letterbox + augmentation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .transforms import Augmentor, AugConfig

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class YOLODetection(Dataset):
    def __init__(self, root, split="train2017", class_map="configs/coco128_classes.json",
                 size=640, train=True, aug: AugConfig | None = None):
        self.root = Path(root)
        cm = json.loads(Path(class_map).read_text())
        self.classes = cm["classes"]
        self.num_classes = cm["num_classes"]
        self.aug = Augmentor(size=size, cfg=aug, train=train)

        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        self.ids = sorted(p.stem for p in self.img_dir.iterdir()
                          if p.suffix.lower() in _IMG_EXT)

    def __len__(self):
        return len(self.ids)

    def _img_path(self, stem):
        for e in _IMG_EXT:
            p = self.img_dir / f"{stem}{e}"
            if p.exists():
                return p
        return self.img_dir / f"{stem}.jpg"

    def _parse(self, txt_path: Path, w0: int, h0: int):
        boxes, labels = [], []
        if txt_path.exists():
            for line in txt_path.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:5])
                x1 = (cx - bw / 2) * w0
                y1 = (cy - bh / 2) * h0
                x2 = (cx + bw / 2) * w0
                y2 = (cy + bh / 2) * h0
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(c)
        return (np.array(boxes, np.float32).reshape(-1, 4),
                np.array(labels, np.int64).reshape(-1))

    def _load_raw(self, i):
        """(img_bgr HxWx3 uint8, boxes [n,4] xyxy abs-px, labels [n])."""
        stem = self.ids[i]
        img = cv2.imread(str(self._img_path(stem)))
        if img is None:
            img = np.zeros((self.aug.size, self.aug.size, 3), np.uint8)
        h0, w0 = img.shape[:2]
        boxes, labels = self._parse(self.lbl_dir / f"{stem}.txt", w0, h0)
        return img, boxes, labels

    def __getitem__(self, i):
        img, boxes, labels, (h0, w0) = self.aug(self, i)
        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor(i), "orig_size": torch.tensor([h0, w0])}
        return img, target
