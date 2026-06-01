"""Compare raw-model vs EMA-model val mAP from a checkpoint (diagnose EMA warmup)."""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.data import YOLODetection, collate_fn
from orsa.engine.evaluator import evaluate

dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.bfloat16 if dev == "cuda" else torch.float32
ck = torch.load("runs/orsa_small_coco128/ckpt/last.pth", map_location=dev, weights_only=False)
ds = YOLODetection("datasets/coco128", "train2017", "configs/coco128_classes.json", 512, train=False)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

m = build_model(num_classes=80, scale="small", aux_query_groups=1, use_aux_dense=True, use_ste=True).to(dev)
m.load_state_dict(ck["model"])
mr = evaluate(m, loader, dev, 80, amp_dtype=amp)
print(f"RAW  model: mAP50={mr['mAP50']:.4f} mAP50-95={mr['mAP50-95']:.4f}")
if ck.get("ema"):
    m.load_state_dict(ck["ema"])
    me = evaluate(m, loader, dev, 80, amp_dtype=amp)
    print(f"EMA  model: mAP50={me['mAP50']:.4f} mAP50-95={me['mAP50-95']:.4f}")
