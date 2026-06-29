"""
PPR-TopK-lite: deterministic PPR approximation for Top-K neighbor selection.

This is a **design / demo** module — power-iteration PPR-lite, not full push-based PPRGo.
Intended to compare against EHGNN's random-walk landing-frequency Top-K.

Parameters
----------
alpha : restart probability (teleport back to seed), typical 0.15
num_iters : power iteration steps; higher = closer to converged PPR
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def row_normalize_csr(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Return row-normalized sparse transition matrix (zero rows stay zero)."""
    adj = adj.tocsr()
    row_sum = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=np.float64)
    nz = row_sum > 0
    inv[nz] = 1.0 / row_sum[nz]
    d_inv = sp.diags(inv)
    return d_inv @ adj


def ppr_power_iteration(
    transition: sp.csr_matrix,
    seed_indices: np.ndarray | list,
    alpha: float = 0.15,
    num_iters: int = 10,
) -> np.ndarray:
    """
    Approximate personalized PageRank scores for selected seed nodes.

    Parameters
    ----------
    transition : scipy CSR row-normalized (n x n)
    seed_indices : seed node ids in [0, n)
    alpha : restart probability
    num_iters : power iteration count

    Returns
    -------
    scores : dense ndarray shape (len(seed_indices), n)
    """
    n = transition.shape[0]
    seeds = np.asarray(seed_indices, dtype=np.int64).ravel()
    out = np.zeros((len(seeds), n), dtype=np.float64)
    t_t = transition.T.tocsr()

    for i, s in enumerate(seeds):
        reset = np.zeros(n, dtype=np.float64)
        reset[s] = 1.0
        r = reset.copy()
        for _ in range(num_iters):
            r = alpha * reset + (1.0 - alpha) * t_t.dot(r)
        out[i] = r
    return out


def topk_from_scores(
    scores: np.ndarray,
    k: int = 20,
    exclude_self: bool = True,
    seed_indices: np.ndarray | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Top-k neighbor indices and normalized weights per score row.

    scores : (batch, n) or (n,)
    seed_indices : optional seed id per row for self-exclusion
    """
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]
    batch, n = scores.shape
    all_idx, all_w = [], []
    for i in range(batch):
        row = scores[i].copy()
        if exclude_self and seed_indices is not None:
            row[int(seed_indices[i])] = -np.inf
        k_eff = min(k, n - (1 if exclude_self else 0))
        if k_eff <= 0:
            all_idx.append([])
            all_w.append([])
            continue
        idx = np.argpartition(-row, kth=min(k_eff - 1, n - 1))[:k_eff]
        idx = idx[np.argsort(-row[idx])]
        w = row[idx]
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            w = np.ones(len(idx)) / len(idx)
        else:
            w = w / s
        all_idx.append(idx.tolist())
        all_w.append(w.tolist())
    return all_idx, all_w


def build_csr_from_topk(
    row_indices: list[int],
    topk_indices: list[list[int]],
    topk_weights: list[list[float]],
    num_cols: int,
) -> sp.csr_matrix:
    """Build CSR matrix (num_rows x num_cols) compatible with EHGNN-style aggregation."""
    rows, cols, data = [], [], []
    for r, nbrs, ws in zip(row_indices, topk_indices, topk_weights):
        for c, w in zip(nbrs, ws):
            rows.append(r)
            cols.append(c)
            data.append(w)
    num_rows = max(row_indices) + 1 if row_indices else 0
    return sp.csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))


def jaccard_topk(a: list | set, b: list | set) -> float:
    """Jaccard similarity between two neighbor index sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def overlap_at_k(a: list | set, b: list | set, k: int) -> float:
    """|A cap B| / k for top-k lists (pad with empty if shorter)."""
    sa = set(list(a)[:k])
    sb = set(list(b)[:k])
    if k <= 0:
        return 0.0
    return len(sa & sb) / float(k)


def dgl_graph_to_scipy_csr(graph, transpose: bool = False) -> tuple[sp.csr_matrix, str]:
    """Convert a DGL homogeneous graph to scipy CSR across DGL versions."""
    # Preferred DGL 2.x external adjacency API
    if hasattr(graph, "adj_external"):
        try:
            mat = graph.adj_external(transpose=transpose, scipy_fmt="csr")
            return sp.csr_matrix(mat).tocsr(), "adj_external"
        except TypeError:
            pass

    # Old DGL API
    try:
        mat = graph.adj(scipy_fmt="csr", transpose=transpose)
        return sp.csr_matrix(mat).tocsr(), "old_adj"
    except TypeError:
        pass

    # Fallback: build from edges manually
    src, dst = graph.edges()
    src_np = src.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()

    if transpose:
        row, col = dst_np, src_np
    else:
        row, col = src_np, dst_np

    data = np.ones(len(row), dtype=np.float32)
    n = graph.num_nodes()
    return sp.csr_matrix((data, (row, col)), shape=(n, n)), "edges_fallback"
