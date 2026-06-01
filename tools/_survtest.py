"""Sanity: survival_loss now live (nonzero + grad) after sum->mean fix."""
import torch
from orsa.losses.survival import survival_loss

torch.manual_seed(0)
B = 2
scales = [(64, 64), (32, 32), (16, 16)]
dense_scores, leaves, grids = [], [], []
for H, W in scales:
    s = (torch.rand(B, H, W) * 0.3).requires_grad_(True)   # low scores -> deficit
    leaves.append(s)
    dense_scores.append(s)
    ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    grids.append(torch.stack([xs.flatten(), ys.flatten()], 1))

targets = [
    {"boxes": torch.tensor([[0.5, 0.5, 0.05, 0.05], [0.3, 0.3, 0.4, 0.4]])},  # small + large
    {"boxes": torch.tensor([[0.7, 0.7, 0.02, 0.02]])},                         # tiny
]
loss = survival_loss(dense_scores, grids, targets)
loss.backward()
gmax = max(s.grad.abs().max().item() for s in leaves)
print(f"loss_surv = {loss.item():.6f}  (was 0.0 before)")
print(f"max |grad| on dense_scores = {gmax:.6e}  (live gradient: {gmax > 0})")
