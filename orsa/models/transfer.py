"""Transfer pretrained YOLO backbone weights into ORSA-Det's CSPBackbone.

Why this works: ORSA's CSPBackbone is built from the Ultralytics YOLOv8 block
lineage (ConvBNAct == Conv, C2f, SPPF) with identical submodule attribute names
(conv, bn, cv1, cv2, m). So a pretrained YOLOv8 backbone (model.0 .. model.9)
maps one-to-one onto ORSA stages by index.

Source selection (block-type + width must both match):
  - ORSA scale 'nano'  -> yolov8n  (width 16,32,64,128,256)
  - ORSA scale 'small' -> yolov8s  (width 32,64,128,256,512)
  - ORSA scale 'base'  -> yolov8m  (width 48,96,192,384,768)
YOLO26 is NOT usable: it replaces C2f with C3k2 (different structure).

Known non-transfer: ORSA's Bottleneck cv1 is 1x1 while YOLOv8's is 3x3, so the
inner bottleneck cv1 conv (+ its BN) is shape-mismatched and skipped. Everything
else (stem, downsample convs, C2f cv1/cv2, bottleneck cv2, SPPF) transfers.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn

# ORSA scale -> ultralytics pretrained checkpoint name (COCO weights).
SCALE_TO_YOLO = {"nano": "yolov8n.pt", "small": "yolov8s.pt", "base": "yolov8m.pt"}

# ORSA backbone submodule prefix -> YOLO sequential index (model.<idx>).
# stage4 holds (downsample Conv, C2f, SPPF) == model.7, model.8, model.9.
_PREFIX_MAP = {
    "stem.": "model.0.",
    "stage1.0.": "model.1.", "stage1.1.": "model.2.",
    "stage2.0.": "model.3.", "stage2.1.": "model.4.",
    "stage3.0.": "model.5.", "stage3.1.": "model.6.",
    "stage4.0.": "model.7.", "stage4.1.": "model.8.", "stage4.2.": "model.9.",
}


def _remap_key(orsa_key: str) -> str | None:
    """ORSA backbone param name -> matching YOLO DetectionModel key (or None)."""
    for pre, yolo_pre in _PREFIX_MAP.items():
        if orsa_key.startswith(pre):
            return yolo_pre + orsa_key[len(pre):]
    return None


def _load_yolo_state(scale: str) -> Dict[str, torch.Tensor]:
    """Fetch a pretrained YOLOv8 DetectionModel state_dict (downloads on first use)."""
    from ultralytics import YOLO
    ckpt = SCALE_TO_YOLO[scale]
    yolo = YOLO(ckpt)  # auto-downloads COCO-pretrained weights
    return yolo.model.state_dict()


def transfer_backbone(
    backbone: nn.Module, scale: str, *, verbose: bool = True
) -> Tuple[int, int, int, int]:
    """Copy matching pretrained YOLO backbone tensors into `backbone` in place.

    Returns (tensors_copied, tensors_total, params_copied, params_total).
    """
    if scale not in SCALE_TO_YOLO:
        raise ValueError(f"scale {scale!r} has no YOLO source; expected {list(SCALE_TO_YOLO)}")
    ystate = _load_yolo_state(scale)
    bstate = backbone.state_dict()

    copied: Dict[str, torch.Tensor] = {}
    n_t = n_pt = 0
    skipped_shape = []
    skipped_nokey = []
    for k, v in bstate.items():
        if k.endswith("num_batches_tracked"):
            continue
        n_t += 1
        n_pt += v.numel()
        yk = _remap_key(k)
        if yk is None or yk not in ystate:
            skipped_nokey.append(k)
            continue
        yv = ystate[yk]
        if tuple(yv.shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(yv.shape)))
            continue
        copied[k] = yv.clone()

    # apply (strict=False: we only fill the matched subset)
    backbone.load_state_dict(copied, strict=False)

    c_t = len(copied)
    c_pt = sum(t.numel() for t in copied.values())
    if verbose:
        print(f"[transfer] source={SCALE_TO_YOLO[scale]} scale={scale}")
        print(f"[transfer] tensors {c_t}/{n_t}  params {c_pt:,}/{n_pt:,} "
              f"({100.0*c_pt/max(n_pt,1):.1f}%)")
        if skipped_shape:
            print(f"[transfer] shape-mismatch skipped ({len(skipped_shape)}): "
                  f"{[s[0] for s in skipped_shape]}")
        if skipped_nokey:
            print(f"[transfer] no-source skipped ({len(skipped_nokey)}): {skipped_nokey}")
    return c_t, n_t, c_pt, n_pt
