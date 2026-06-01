"""End-to-end smoke test: forward + criterion backward on a tiny dummy batch."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orsa.models import build_model
from orsa.losses import SetCriterion
from orsa.engine import evaluate
from orsa.utils import postprocess


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nc = 15
    model = build_model(num_classes=nc, scale="small", aux_query_groups=1,
                        use_aux_dense=True, use_ste=True).to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={dev} params={n_params:.2f}M")

    x = torch.randn(2, 3, 512, 512, device=dev)
    targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]], device=dev),
         "labels": torch.tensor([1, 4], device=dev),
         "image_id": torch.tensor([0]), "orig_size": torch.tensor([512, 512])},
        {"boxes": torch.tensor([[0.6, 0.4, 0.25, 0.25]], device=dev),
         "labels": torch.tensor([7], device=dev),
         "image_id": torch.tensor([1]), "orig_size": torch.tensor([512, 512])},
    ]

    # ---- train forward + loss + backward ----
    model.train()
    out = model(x)
    print("forward keys:", sorted(out.keys()))
    print("pred_logits:", tuple(out["pred_logits"].shape), "pred_boxes:", tuple(out["pred_boxes"].shape))
    crit = SetCriterion(nc, lambda_surv=1.0, lambda_sparse=1e-3).to(dev)
    total, logs = crit(out, targets)
    print("loss total:", float(total))
    for k, v in logs.items():
        print(f"  {k}: {float(v):.4f}")
    total.backward()
    grad_ok = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"params_with_grad={n_with_grad} finite_grad={grad_ok}")
    assert torch.isfinite(total), "loss is not finite"

    # ---- eval / inference path ----
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    dets = postprocess(out_e, [(512, 512), (512, 512)], topk=100)
    print("postprocess -> dets:", len(dets), "sample keys:", sorted(dets[0].keys()))
    print("sample boxes:", tuple(dets[0]["boxes"].shape), "scores:", tuple(dets[0]["scores"].shape))

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
