"""IDD Detection dataset (Pascal VOC XML).

Layout:
  root/JPEGImages/<entry>.jpg
  root/Annotations/<entry>.xml
  root/{train,val,test}.txt   # one <entry> stem per line

Returns (image[3,S,S] float, target dict{boxes cxcywh norm, labels, image_id, orig_size}).
"""
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .transforms import Augmentor, AugConfig


class IDDDetection(Dataset):
    def __init__(self, root, split="train.txt", class_map="configs/idd_classes.json",
                 size=640, train=True, aug: AugConfig | None = None):
        self.root = Path(root)
        cm = json.loads(Path(class_map).read_text())
        self.classes = cm["classes"]
        self.name_to_id = cm["name_to_id"]
        self.num_classes = cm["num_classes"]
        self.aug = Augmentor(size=size, cfg=aug, train=train)

        entries = (self.root / split).read_text().splitlines()
        self.ids = [e.strip() for e in entries if e.strip()]
        self.img_dir = self.root / "JPEGImages"
        self.ann_dir = self.root / "Annotations"

    def __len__(self):
        return len(self.ids)

    def _parse(self, xml_path: Path):
        boxes, labels = [], []
        if xml_path.exists():
            for obj in ET.parse(xml_path).getroot().findall("object"):
                name = (obj.findtext("name") or "").strip()
                if name not in self.name_to_id:
                    continue
                bb = obj.find("bndbox")
                x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
                x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(self.name_to_id[name])
        return (np.array(boxes, np.float32).reshape(-1, 4),
                np.array(labels, np.int64).reshape(-1))

    def _load_raw(self, i):
        """(img_bgr HxWx3 uint8, boxes [n,4] xyxy abs-px, labels [n])."""
        stem = self.ids[i]
        img = cv2.imread(str(self.img_dir / f"{stem}.jpg"))
        if img is None:  # some entries use .png fallback
            img = cv2.imread(str(self.img_dir / f"{stem}.png"))
        if img is None:
            img = np.zeros((self.aug.size, self.aug.size, 3), np.uint8)
        boxes, labels = self._parse(self.ann_dir / f"{stem}.xml")
        return img, boxes, labels

    def __getitem__(self, i):
        img, boxes, labels, (h0, w0) = self.aug(self, i)
        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor(i), "orig_size": torch.tensor([h0, w0])}
        return img, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    return imgs, targets
