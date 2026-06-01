"""Postprocessing + COCO-style evaluation (mAP, AP-S/M/L, AR@100)."""
from __future__ import annotations
import contextlib, io
import numpy as np
import torch
from ..losses.box_ops import cxcywh_to_xyxy


@torch.no_grad()
def postprocess(outputs, orig_sizes, topk=300):
    """outputs: pred_logits[B,Q,C], pred_boxes[B,Q,4] cxcywh norm.
    Returns list per image: dict(boxes xyxy abs px, scores, labels)."""
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    B, Q, C = logits.shape
    prob = logits.sigmoid()
    results = []
    for b in range(B):
        sz = orig_sizes[b]
        h, w = sz.tolist() if torch.is_tensor(sz) else tuple(sz)
        scores_all = prob[b].flatten()                       # [Q*C]
        k = min(topk, scores_all.numel())
        topv, topi = scores_all.topk(k)
        q_idx = topi // C
        cls = topi % C
        bb = cxcywh_to_xyxy(boxes[b][q_idx])                 # normalized
        bb = bb * torch.tensor([w, h, w, h], device=bb.device)
        results.append({"boxes": bb, "scores": topv, "labels": cls})
    return results


class COCOEvaluator:
    """Accumulate preds + GT in COCO format, compute metrics via pycocotools."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.images, self.anns, self.dets = [], [], []
        self._ann_id = 1

    def update(self, image_id, gt_boxes_xyxy, gt_labels, det):
        self.images.append({"id": int(image_id)})
        for box, lab in zip(gt_boxes_xyxy.tolist(), gt_labels.tolist()):
            x1, y1, x2, y2 = box
            self.anns.append({"id": self._ann_id, "image_id": int(image_id),
                              "category_id": int(lab), "bbox": [x1, y1, x2 - x1, y2 - y1],
                              "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
            self._ann_id += 1
        for box, sc, lab in zip(det["boxes"].tolist(), det["scores"].tolist(), det["labels"].tolist()):
            x1, y1, x2, y2 = box
            self.dets.append({"image_id": int(image_id), "category_id": int(lab),
                              "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)})

    def summarize(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        # per_class_ap / per_class_ap50: AP per category id (NaN if no GT for that class)
        self.per_class_ap = [float("nan")] * self.num_classes
        self.per_class_ap50 = [float("nan")] * self.num_classes
        if not self.dets:
            return {"mAP50-95": 0.0, "mAP50": 0.0, "AP_s": 0.0, "AP_m": 0.0, "AP_l": 0.0, "AR100": 0.0}
        gt = {"images": self.images,
              "annotations": self.anns,
              "categories": [{"id": i} for i in range(self.num_classes)]}
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO()
            coco.dataset = gt
            coco.createIndex()
            dt = coco.loadRes(self.dets)
            ev = COCOeval(coco, dt, "bbox")
            ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
        # per-class AP from precision array [T(iou),R(recall),K(cls),A(area),M(maxdet)]
        prec = ev.eval["precision"]                          # [10,101,K,4,3]
        cat_ids = ev.params.catIds
        for k, cid in enumerate(cat_ids):
            if cid >= self.num_classes:
                continue
            p = prec[:, :, k, 0, -1]                          # area=all, maxdet=100
            p_all = p[p > -1]
            p50 = prec[0, :, k, 0, -1]
            p50 = p50[p50 > -1]
            self.per_class_ap[cid] = float(p_all.mean()) if p_all.size else float("nan")
            self.per_class_ap50[cid] = float(p50.mean()) if p50.size else float("nan")
        return {"mAP50-95": s[0], "mAP50": s[1], "AP_s": s[3], "AP_m": s[4],
                "AP_l": s[5], "AR100": s[8]}


def _box_iou(a, b):
    """a[N,4], b[M,4] xyxy -> IoU[N,M]."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class ConfusionMatrix:
    """Ultralytics-style confusion matrix. Rows = predicted class, cols = GT class.
    Index `num_classes` is the background row/col (FP / FN)."""

    def __init__(self, num_classes, conf=0.25, iou_thres=0.45):
        self.nc = num_classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    @torch.no_grad()
    def process(self, det, gt_boxes_xyxy, gt_labels):
        """det: dict(boxes xyxy, scores, labels) all CPU tensors. gt_* tensors."""
        nc = self.nc
        gt_n = int(gt_labels.numel())
        keep = det["scores"] >= self.conf
        d_boxes = det["boxes"][keep]
        d_labels = det["labels"][keep].long()
        if gt_n == 0:                                        # all dets are FP
            for c in d_labels.tolist():
                self.matrix[c, nc] += 1
            return
        if d_boxes.shape[0] == 0:                            # all GT missed (FN)
            for g in gt_labels.long().tolist():
                self.matrix[nc, g] += 1
            return
        iou = _box_iou(gt_boxes_xyxy, d_boxes)               # [G,D]
        idx = torch.where(iou > self.iou_thres)
        if idx[0].shape[0]:
            m = torch.stack(idx, 1).cpu().numpy()            # [K,2] (gt,det)
            v = iou[idx[0], idx[1]].cpu().numpy()
            order = v.argsort()[::-1]
            m, v = m[order], v[order]
            m = m[np.unique(m[:, 1], return_index=True)[1]]  # one gt per det
            m = m[np.unique(m[:, 0], return_index=True)[1]]  # one det per gt
        else:
            m = np.zeros((0, 2), dtype=np.int64)
        matched_gt = set(m[:, 0].tolist())
        matched_det = set(m[:, 1].tolist())
        gl = gt_labels.long().tolist()
        dl = d_labels.tolist()
        for gi, di in m:
            self.matrix[dl[di], gl[gi]] += 1                 # TP / misclass
        for gi in range(gt_n):
            if gi not in matched_gt:
                self.matrix[nc, gl[gi]] += 1                 # FN (predicted bg)
        for di in range(len(dl)):
            if di not in matched_det:
                self.matrix[dl[di], nc] += 1                 # FP (gt is bg)
