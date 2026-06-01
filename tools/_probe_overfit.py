"""Probe: does RAW model actually overfit? Inspect GT vs predictions on train images.

Loads checkpoint RAW weights, runs a few train images, prints for each:
  - GT boxes (cxcywh norm) + labels
  - top predictions by score (label, score, box cxcywh norm)
  - best IoU each GT gets from ANY prediction (regardless of score/label)
  - best IoU each GT gets from a prediction of the CORRECT label
This separates "boxes are garbage" from "scores/labels wrong".
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.data import YOLODetection, collate_fn
from orsa.losses.box_ops import cxcywh_to_xyxy

dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.bfloat16 if dev == "cuda" else torch.float32
N_IMG = 3


def iou_xyxy(a, b):
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


ck = torch.load("runs/orsa_small_coco128/ckpt/best.pth", map_location=dev, weights_only=False)
ds = YOLODetection("datasets/coco128", "train2017", "configs/coco128_classes.json", 512, train=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
names = ds.classes

m = build_model(num_classes=80, scale="small", aux_query_groups=1, use_aux_dense=True, use_ste=True).to(dev)
m.load_state_dict(ck["model"])
m.eval()

print(f"ckpt epoch={ck.get('epoch')} keys={list(ck.keys())}")
it = iter(loader)
for n in range(N_IMG):
    imgs, targets = next(it)
    imgs = imgs.to(dev)
    t = targets[0]
    gt_b = t["boxes"]            # cxcywh norm
    gt_l = t["labels"]
    with torch.no_grad(), torch.autocast(device_type=dev.split(":")[0], dtype=amp, enabled=dev != "cpu"):
        out = m(imgs)
    logits = out["pred_logits"][0].float().cpu()     # [Q,C]
    boxes = out["pred_boxes"][0].float().cpu()       # [Q,4] cxcywh
    prob = logits.sigmoid()
    sc_per_q, cls_per_q = prob.max(1)          # best class per query
    order = sc_per_q.argsort(descending=True)

    print(f"\n===== IMG {n}  GT={gt_l.numel()} objects  max_q_score={sc_per_q.max():.3f} =====")
    print("GT:")
    for i in range(gt_l.numel()):
        b = gt_b[i].tolist()
        print(f"  {names[gt_l[i]]:<14} cxcywh=[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}]")
    print("TOP-5 preds:")
    for qi in order[:5].tolist():
        b = boxes[qi].tolist()
        print(f"  {names[cls_per_q[qi]]:<14} s={sc_per_q[qi]:.3f} cxcywh=[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}]")

    if gt_l.numel():
        gt_xy = cxcywh_to_xyxy(gt_b)
        pr_xy = cxcywh_to_xyxy(boxes)
        iou = iou_xyxy(gt_xy, pr_xy)           # [G,Q]
        best_any, best_any_q = iou.max(1)
        # best IoU among preds whose top-class == gt label
        print("per-GT best-IoU:")
        for i in range(gt_l.numel()):
            same = (cls_per_q == gt_l[i])
            if same.any():
                bi = (iou[i] * same.float()).max()
            else:
                bi = torch.tensor(0.0)
            qi = best_any_q[i].item()
            print(f"  {names[gt_l[i]]:<14} bestIoU(any)={best_any[i]:.2f} "
                  f"(pred={names[cls_per_q[qi]]} s={sc_per_q[qi]:.2f}) "
                  f"bestIoU(sameclass)={bi:.2f}")
