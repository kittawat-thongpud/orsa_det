"""Isolate cls learnability: overfit a SINGLE batch with accum=1, many updates.

If max class score climbs toward ~0.9 -> model CAN learn cls; prior near-zero mAP
was undertraining (grad-accum starved optimizer steps). If it stays stuck ~0.15
-> deeper classifier bug.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.losses import SetCriterion
from orsa.data import YOLODetection, collate_fn
from orsa.engine.optim import build_optimizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.bfloat16 if dev == "cuda" else torch.float32
STEPS = 400
LR = 1e-3

ds = YOLODetection("datasets/coco128", "train2017", "configs/coco128_classes.json", 512, train=False)
sub = Subset(ds, [0, 1, 2, 3])
loader = DataLoader(sub, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
imgs, targets = next(iter(loader))
imgs = imgs.to(dev)
targets = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

m = build_model(num_classes=80, scale="small", aux_query_groups=1, use_aux_dense=True, use_ste=True).to(dev)
crit = SetCriterion(80, lambda_surv=1.0, lambda_sparse=1e-3, lambda_dense=1.0)
opt = build_optimizer(m, phase="A", lr=LR)[0]
m.train()
for step in range(STEPS):
    with torch.autocast(device_type=dev.split(":")[0], dtype=amp, enabled=dev != "cpu"):
        out = m(imgs)
        loss, logs = crit(out, targets)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 50 == 0 or step == STEPS - 1:
        with torch.no_grad():
            sc = out["pred_logits"][0].float().sigmoid().max().item()
        print(f"step {step:3d} total={loss.item():6.2f} main={logs['loss_main']:.2f} "
              f"max_score(img0)={sc:.3f}")

# final eval on the 4 imgs
m.eval()
with torch.no_grad(), torch.autocast(device_type=dev.split(":")[0], dtype=amp, enabled=dev != "cpu"):
    out = m(imgs)
for b in range(4):
    prob = out["pred_logits"][b].float().sigmoid()
    s, c = prob.max(1)
    top = s.argmax()
    print(f"img{b}: max_score={s.max():.3f} as class={ds.classes[c[top]]}  "
          f"GT={[ds.classes[l] for l in targets[b]['labels'].tolist()][:5]}")
