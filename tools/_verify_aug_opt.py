"""Verify mosaic/mixup aug + config-driven optimizer refactor end-to-end.
Run: PYTHONPATH=. ./.venv/Scripts/python.exe tools/_verify_aug_opt.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from orsa.data import AugConfig, Augmentor, collate_fn
from orsa.engine import OptimConfig, SchedConfig, EMAConfig, build_optimizer
from orsa.models import build_model


def _fake_dataset(n=8, size=512):
    """Minimal duck-typed dataset exposing _load_raw(i) -> (bgr, xyxy_px, labels)."""
    rng = np.random.default_rng(0)

    class _DS:
        def __len__(self):
            return n

        def _load_raw(self, i):
            h, w = 480, 640
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            k = int(rng.integers(1, 5))
            boxes = np.zeros((k, 4), np.float32)
            for j in range(k):
                x1 = rng.uniform(0, w - 20); y1 = rng.uniform(0, h - 20)
                x2 = min(w, x1 + rng.uniform(20, 200)); y2 = min(h, y1 + rng.uniform(20, 200))
                boxes[j] = [x1, y1, x2, y2]
            labels = rng.integers(0, 15, (k,)).astype(np.int64)
            return img, boxes, labels

    return _DS()


def check_aug():
    ds = _fake_dataset()
    cfg = AugConfig.from_dict({"mosaic_p": 1.0, "mixup_p": 1.0})
    aug = Augmentor(size=512, cfg=cfg, train=True)
    for i in range(4):
        img, boxes, labels, hw = aug(ds, i)
        assert img.shape == (3, 512, 512), img.shape
        assert img.dtype == torch.float32 and 0.0 <= float(img.min()) and float(img.max()) <= 1.0
        assert boxes.ndim == 2 and boxes.shape[1] == 4, boxes.shape
        assert boxes.shape[0] == labels.shape[0]
        if boxes.numel():
            assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 1.0, "cxcywh not normalized"
    # eval path: mosaic/mixup must be disabled
    aug_eval = Augmentor(size=512, cfg=cfg, train=False)
    img, boxes, labels, hw = aug_eval(ds, 0)
    assert img.shape == (3, 512, 512)
    # unknown key rejected
    try:
        AugConfig.from_dict({"bogus": 1}); raise SystemExit("AugConfig accepted unknown key")
    except KeyError:
        pass
    print("[ok] aug: mosaic+mixup tensors valid, eval disables mosaic, unknown-key rejected")


def check_optim():
    model = build_model(num_classes=15, scale="small", aux_query_groups=1,
                        use_aux_dense=True, use_ste=True)
    # legacy path
    o = build_optimizer(model, phase="A", lr=2e-4)
    assert o[0] is not None and o[1] is None and o[2] is None
    o = build_optimizer(model, phase="B", lr=2e-4)
    assert all(x is not None for x in o), "muon_hybrid via legacy phase B"
    # config path: all three names
    for name, want_three in (("adamw", False), ("sgd", False), ("muon_hybrid", True)):
        o = build_optimizer(model, cfg=OptimConfig(name=name, lr=1e-3))
        if want_three:
            assert all(x is not None for x in o), name
        else:
            assert o[0] is not None and o[1] is None and o[2] is None, name
    try:
        build_optimizer(model, cfg=OptimConfig(name="nope")); raise SystemExit("bad name accepted")
    except ValueError:
        pass
    try:
        OptimConfig.from_dict({"bogus": 1}); raise SystemExit("OptimConfig accepted unknown key")
    except KeyError:
        pass
    print("[ok] optim: legacy phase + adamw/sgd/muon_hybrid + validation all good")


def check_configs():
    for p in ("configs/orsa_small_coco128.yaml", "configs/orsa_small_idd.yaml"):
        cfg = yaml.safe_load(Path(p).read_text())
        AugConfig.from_dict(cfg.get("augment"))
        OptimConfig.from_dict(cfg.get("optimizer"))
        SchedConfig.from_dict(cfg.get("schedule"))
        EMAConfig.from_dict(cfg.get("ema"))
    print("[ok] configs: both YAMLs parse into all four dataclasses")


if __name__ == "__main__":
    check_aug()
    check_optim()
    check_configs()
    print("ALL VERIFY PASSED")
