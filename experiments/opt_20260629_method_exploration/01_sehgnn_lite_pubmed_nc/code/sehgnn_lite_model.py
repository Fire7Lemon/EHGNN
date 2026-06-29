"""
SeHGNN-lite lightweight classification heads.

Input format
------------
All models accept ``views``: a list of 2-D float tensors with shape ``[batch, in_dim]``.
Each tensor is one precomputed feature view (e.g. raw self feature or one meta-path
aggregation). Views may share the same ``in_dim`` but batch size must match.

Output: logits ``[batch, num_classes]`` (no softmax; use NLLLoss on log_softmax output).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseMLPHead(nn.Module):
    """Shared MLP trunk: Linear -> ReLU -> Dropout (x n_hidden_layers-1) -> Linear."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(max(n_hidden_layers - 1, 0)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ConcatMLP(_BaseMLPHead):
    """
    Version A: concat multiple views then MLP.

    ``views = [h_1, h_2, ..., h_m]`` each ``[B, in_dim]`` -> concat -> MLP -> logits.
    Single-view input (``m=1``) is supported.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        n_views: int = 1,
        n_hidden_layers: int = 2,
    ):
        self.in_dim = in_dim
        self.n_views = n_views
        super().__init__(
            in_dim=in_dim * n_views,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        if len(views) == 1:
            x = views[0]
        else:
            x = torch.cat(views, dim=1)
        return super().forward(x)


class MeanFusionMLP(nn.Module):
    """
    Version B: project each view, mean-fuse, then MLP.

    ``views = [h_1, ..., h_m]`` -> Linear(in_dim, hidden) + ReLU per view
    -> stack -> mean(dim=0) -> MLP -> logits.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        n_views: int = 1,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_views = n_views
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim) for _ in range(n_views)]
        )
        self.head = _BaseMLPHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        projected = [F.relu(proj(v)) for proj, v in zip(self.projections, views)]
        fused = torch.stack(projected, dim=0).mean(dim=0)
        return self.head(fused)


def build_sehgnn_lite_head(
    fusion: str,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    n_views: int,
    n_hidden_layers: int = 2,
) -> nn.Module:
    fusion = fusion.lower()
    if fusion == "concat":
        return ConcatMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            n_views=n_views,
            n_hidden_layers=n_hidden_layers,
        )
    if fusion == "mean":
        return MeanFusionMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            n_views=n_views,
            n_hidden_layers=n_hidden_layers,
        )
    raise ValueError("fusion must be 'concat' or 'mean', got {!r}".format(fusion))
