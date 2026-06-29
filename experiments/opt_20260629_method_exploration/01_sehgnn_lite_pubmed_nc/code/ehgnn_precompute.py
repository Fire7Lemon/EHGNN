"""
Precompute SeHGNN-lite feature views by reusing EHGNN random-walk CSR matrices.

Aggregation follows EHGNN neighbor scatter logic with **uniform** meta-path and
node-type weights (no learnable MWeight / TWeight). Raw 200-d features are used
(no inner trainable MLP during aggregation), matching SIGN/SeHGNN-style
precomputation before a lightweight classifier.
"""
from __future__ import annotations

import numpy as np
import torch


def sparse_aggregate(features_t: torch.Tensor, sim_csr) -> torch.Tensor:
    """
    Weighted neighbor sum: ``sim_csr @ features_t``.

    Parameters
    ----------
    features_t : tensor [num_nodes_type, feat_dim]
    sim_csr : scipy CSR, shape [num_source, num_target]
    """
    feat_np = features_t.detach().cpu().numpy().astype(np.float32)
    agg = sim_csr.dot(feat_np)
    return torch.from_numpy(np.asarray(agg, dtype=np.float32))


def metapath_view(
    features: dict,
    sim_matrix: dict,
    t_types: list,
) -> torch.Tensor:
    """
    One meta-path aggregated view for all source nodes in sim_matrix rows.

    Returns tensor [num_source, feat_dim] — uniform sum over target types.
    """
    feat_dim = next(iter(features.values())).shape[1]
    out = None
    n_types = len(t_types)
    for t_type in t_types:
        part = sparse_aggregate(features[t_type], sim_matrix[t_type])
        if out is None:
            out = part / float(n_types)
        else:
            out = out + part / float(n_types)
    if out is None:
        raise RuntimeError("metapath_view: empty t_types")
    return out


def build_node_views(
    features: dict,
    s_type: int,
    sim_matrixs: list,
    t_typess: list,
    node_indices: torch.Tensor,
    include_self: bool = True,
) -> list[torch.Tensor]:
    """
    Build per-view tensors indexed by ``node_indices`` (local disease ids).

    Returns list of tensors each [len(node_indices), feat_dim]:
      - optional self raw feature
      - one tensor per meta-path aggregation
    """
    views_full: list[torch.Tensor] = []

    if include_self:
        self_full = features[s_type].float()
        views_full.append(self_full)

    for sim_matrix, t_types in zip(sim_matrixs, t_typess):
        views_full.append(metapath_view(features, sim_matrix, t_types))

    views = []
    idx = node_indices.long()
    for v in views_full:
        views.append(v.index_select(0, idx))
    return views


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
