"""
Lightweight MLP student for P4 EHGNN-to-MLP distillation.

The student uses **node features only** — no graph structure, no DGL.
Input: ``x`` tensor ``[batch, in_dim]``; output: raw logits ``[batch, num_classes]``.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPStudent(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        n_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = in_dim
        for i in range(max(n_layers - 1, 0)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, num_classes))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    x: torch.Tensor,
    repeat: int = 20,
    device: torch.device | None = None,
    warmup: int = 5,
) -> float:
    """
    Return mean forward time in **milliseconds** per full-batch forward.

    Uses ``model.eval()`` and synchronizes CUDA if available.
    Does **not** include data loading or graph preprocessing.
    """
    if device is None:
        device = x.device
    model.eval()
    x = x.to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        model(x)
        sync()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(times) / len(times))
