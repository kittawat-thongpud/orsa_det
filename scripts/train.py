"""Train ORSA-Det. Usage: python scripts/train.py --cfg configs/orsa_small_idd.yaml

Toggles (Ultralytics-style reporting):
  --no-train     skip training (eval-only / report-only on init weights)
  --no-val       skip validation + final report
  --no-report    train + val but skip PNG report generation
  --no-density   skip the (slow) dataset class-density scan
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.losses import SetCriterion
from orsa.data import IDDDetection, YOLODetection, AugConfig, collate_fn
from orsa.engine import train, OptimConfig, SchedConfig, EMAConfig
from orsa.utils import RunLogger, log_init, class_density

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_dataset(d, split, train_flag, aug_cfg):
    fmt = d.get("format", "voc")
    if fmt == "yolo":
        return YOLODetection(d["root"], split, d["class_map"], d["img_size"],
                             train=train_flag, aug=aug_cfg)
    return IDDDetection(d["root"], split, d["class_map"], d["img_size"],
                        train=train_flag, aug=aug_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-density", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text())

    d, m, l, tr, rn = cfg["dataset"], cfg["model"], cfg["loss"], cfg["train"], cfg["run"]
    if args.epochs: tr["epochs"] = args.epochs

    # ---- config-driven sub-configs (no magic numbers; YAML overrides) ------
    aug_cfg = AugConfig.from_dict(cfg.get("augment"))
    sched_cfg = SchedConfig.from_dict(cfg.get("schedule"))
    if cfg.get("ema") is not None:
        ema_cfg = EMAConfig.from_dict(cfg["ema"])
    else:  # legacy: bool flag under train:
        ema_cfg = EMAConfig(enabled=bool(tr.get("ema", True)))
    if cfg.get("optimizer") is not None:
        opt_cfg = OptimConfig.from_dict(cfg["optimizer"])
    else:  # legacy: phase + lr under train:
        opt_cfg = OptimConfig.from_phase(tr.get("phase", "A"), tr["lr"],
                                         tr.get("weight_decay", OptimConfig.weight_decay))
    if args.phase:  # CLI override keeps lr/wd, swaps optimizer family
        opt_cfg = OptimConfig.from_phase(args.phase, opt_cfg.lr, opt_cfg.weight_decay)

    train_ds = build_dataset(d, d["train_split"], True, aug_cfg)
    val_ds = build_dataset(d, d["val_split"], False, aug_cfg)
    names = getattr(train_ds, "classes", [str(i) for i in range(d["num_classes"])])
    print(f"train={len(train_ds)} val={len(val_ds)} classes={d['num_classes']}")

    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                              num_workers=tr["workers"], collate_fn=collate_fn,
                              pin_memory=True, drop_last=True, persistent_workers=tr["workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False,
                            num_workers=tr["workers"], collate_fn=collate_fn, pin_memory=True)

    model = build_model(num_classes=d["num_classes"], scale=m["scale"],
                        aux_query_groups=m["aux_query_groups"],
                        use_aux_dense=m["use_aux_dense"], use_ste=m["use_ste"])
    # optional: warm-start CSPBackbone from pretrained YOLOv8 COCO weights.
    # model.pretrained_backbone: yolo (auto-pick by scale) | none (default).
    if str(m.get("pretrained_backbone", "none")).lower() not in ("none", "", "false"):
        from orsa.models import transfer_backbone
        transfer_backbone(model.backbone, m["scale"])
    criterion = SetCriterion(d["num_classes"], lambda_surv=l["lambda_surv"],
                             lambda_sparse=l["lambda_sparse"],
                             lambda_dense=l.get("lambda_dense", 1.0))
    logger = RunLogger(rn["dir"], rn["name"])

    # ---- init banner: config + model components + param counts -------------
    log_init(model, cfg, num_classes=d["num_classes"], names=names,
             train_n=len(train_ds), val_n=len(val_ds), out_dir=logger.dir,
             phase=opt_cfg.name, img_size=d["img_size"])

    # ---- dataset class density (for report histogram) ----------------------
    class_counts = None
    if not args.no_density and not args.no_report:
        print("scanning class density...")
        class_counts = class_density(train_ds, d["num_classes"])

    best = train(model, criterion, train_loader, val_loader, logger,
                 device=tr["device"], epochs=tr["epochs"],
                 accum=tr["accum"], grad_clip=tr["grad_clip"],
                 amp_dtype=DTYPE[tr["amp_dtype"]], eval_every=tr["eval_every"],
                 opt_cfg=opt_cfg, sched_cfg=sched_cfg, ema_cfg=ema_cfg,
                 num_classes=d["num_classes"],
                 do_train=not args.no_train, do_val=not args.no_val,
                 make_report=not args.no_report, names=names, class_counts=class_counts)
    logger.close()
    print(f"best mAP50-95={best:.4f}")


if __name__ == "__main__":
    main()
