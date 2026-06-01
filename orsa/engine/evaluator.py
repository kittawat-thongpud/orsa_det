"""Validation loop -> COCO metrics."""
from __future__ import annotations
import torch
from tqdm import tqdm
from ..utils.metrics import postprocess, COCOEvaluator
from ..losses.box_ops import cxcywh_to_xyxy


@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp_dtype=torch.bfloat16,
             max_batches=None, confusion=None, return_per_class=False):
    """confusion: optional ConfusionMatrix to accumulate. return_per_class: when
    True returns (metrics, per_class_ap, per_class_ap50)."""
    model.eval()
    ev = COCOEvaluator(num_classes)
    for bi, (imgs, targets) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches and bi >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=device != "cpu"):
            out = model(imgs)
        orig = [t["orig_size"] for t in targets]
        dets = postprocess(out, orig)
        for i, t in enumerate(targets):
            h, w = t["orig_size"].tolist()
            gt = t["boxes"]
            if gt.numel():
                gt = cxcywh_to_xyxy(gt) * torch.tensor([w, h, w, h])
            det_cpu = {k: v.cpu() for k, v in dets[i].items()}
            ev.update(int(t["image_id"]), gt, t["labels"], det_cpu)
            if confusion is not None:
                confusion.process(det_cpu, gt, t["labels"])
    metrics = ev.summarize()
    if return_per_class:
        return metrics, ev.per_class_ap, ev.per_class_ap50
    return metrics
