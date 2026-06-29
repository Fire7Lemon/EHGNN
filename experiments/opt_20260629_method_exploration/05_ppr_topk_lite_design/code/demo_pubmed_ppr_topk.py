#!/usr/bin/env python3
"""
P5 Demo: compare PPR-TopK vs RW-TopK on PubMed meta-path (small node sample).

Server-only (requires DGL + PubMed data). Does NOT train any model.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

_FILE = Path(__file__).resolve()
EXP_ROOT = _FILE.parents[2]
COMMON_DIR = EXP_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from path_utils import bootstrap_paths, resolve_project_data_path  # noqa: E402

PROJECT_ROOT, EXP_ROOT, CODE_DIR = bootstrap_paths(__file__, task="nc")

from ppr_topk_lite import (  # noqa: E402
    build_csr_from_topk,
    dgl_graph_to_scipy_csr,
    jaccard_topk,
    overlap_at_k,
    ppr_power_iteration,
    row_normalize_csr,
    topk_from_scores,
)
from utils import load_PubMed, random_walk_sim  # noqa: E402

METAPATHS_PUBMED = [
    ["dad_r", "dad"],
    ["gcd_r", "gcd"],
    ["gcd_r", "gag", "gcd"],
    ["cid_r", "cid"],
    ["cid_r", "cig", "cig_r", "cac", "cis", "cis_r", "cid"],
    ["swd_r", "sas", "swd"],
]
WALK_NUM = 100


def parse_args():
    p = argparse.ArgumentParser(description="P5 PPR-TopK PubMed demo")
    p.add_argument("--dataset", type=str, default="PubMed")
    p.add_argument("--path", type=str, default="data",
                   help="Optional absolute data root; default PROJECT_ROOT/data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_target_nodes", type=int, default=500)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--num_iters", type=int, default=10)
    p.add_argument("--metapath_index", type=int, default=0)
    p.add_argument(
        "--root_out",
        type=str,
        default="experiments/opt_20260629_method_exploration",
    )
    return p.parse_args()


def build_metapath_adjacency(
    g, metapath: list[str]
) -> tuple[sp.csr_matrix, int, int, str, str]:
    """
    Build meta-path adjacency on source-type local id space via DGL metapath_reachable_graph.
    Returns (adj_csr, num_src, num_dst, adj_conversion_mode, dgl_version).
    """
    import dgl

    dgl_ver = dgl.__version__
    print("dgl_version:", dgl_ver)
    mg = dgl.metapath_reachable_graph(g, metapath)
    adj, mode = dgl_graph_to_scipy_csr(mg, transpose=False)
    print("dgl_adj_conversion_mode:", mode)
    n = mg.num_nodes()
    return adj, n, n, mode, dgl_ver


def rw_topk_for_seeds(g, metapath, seed_indices, k, walk_num, random_flag=False):
    """Run original random_walk_sim on seed batch; return per-type CSR dict."""
    batch = torch.as_tensor(seed_indices, dtype=torch.long)
    sim_matrix, t_types, s_type = random_walk_sim(
        batch, g, metapath, walk_num, k, random_flag
    )
    return sim_matrix, t_types, s_type


def neighbors_from_csr_row(csr: sp.csr_matrix, row: int, k: int) -> list[int]:
    start, end = csr.indptr[row], csr.indptr[row + 1]
    cols = csr.indices[start:end]
    data = csr.data[start:end]
    if len(cols) == 0:
        return []
    order = np.argsort(-data)[:k]
    return cols[order].tolist()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)

    data_path = resolve_project_data_path(PROJECT_ROOT, args.path)
    print("data_path:", data_path)
    results_dir = Path(args.root_out) / "05_ppr_topk_lite_design" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    g, features, labels, idx_train, idx_test = load_PubMed(data_path, args.dataset, False)
    load_sec = time.perf_counter() - t0
    print("Done Load Data, Running time: {:.4f} Seconds".format(load_sec))

    metapath = METAPATHS_PUBMED[args.metapath_index]
    print("Metapath:", metapath)

    t1 = time.perf_counter()
    mp_adj, num_src, _, adj_mode, dgl_ver = build_metapath_adjacency(g, metapath)
    build_sec = time.perf_counter() - t1
    print("Metapath adjacency shape: {} (build {:.4f}s)".format(mp_adj.shape, build_sec))

    rng = np.random.default_rng(args.seed)
    n_pick = min(args.num_target_nodes, num_src)
    target_nodes = np.sort(rng.choice(num_src, size=n_pick, replace=False))

    transition = row_normalize_csr(mp_adj)

    t_ppr = time.perf_counter()
    scores = ppr_power_iteration(transition, target_nodes, args.alpha, args.num_iters)
    ppr_idx, ppr_w = topk_from_scores(scores, k=args.k, exclude_self=True, seed_indices=target_nodes)
    ppr_sec = time.perf_counter() - t_ppr

    ppr_csr = build_csr_from_topk(
        target_nodes.tolist(), ppr_idx, ppr_w, num_cols=mp_adj.shape[1]
    )

    t_rw = time.perf_counter()
    sim_matrix, t_types, s_type = rw_topk_for_seeds(
        g, metapath, target_nodes, args.k, WALK_NUM, False
    )
    rw_sec = time.perf_counter() - t_rw

    # Use first target type CSR from RW (same-type metapath dad->dad)
    rw_csr = sim_matrix[t_types[0]] if t_types else sp.csr_matrix((num_src, num_src))

    jaccards, overlaps = [], []
    per_node_rows = []
    for i, node in enumerate(target_nodes):
        ppr_n = ppr_idx[i]
        rw_n = neighbors_from_csr_row(rw_csr, int(node), args.k)
        j = jaccard_topk(ppr_n, rw_n)
        o = overlap_at_k(ppr_n, rw_n, args.k)
        jaccards.append(j)
        overlaps.append(o)
        per_node_rows.append(
            {
                "node_id": int(node),
                "jaccard": j,
                "overlap_at_k": o,
                "ppr_neighbors": len(ppr_n),
                "rw_neighbors": len(rw_n),
            }
        )

    mean_j = float(np.mean(jaccards)) if jaccards else 0.0
    median_j = float(np.median(jaccards)) if jaccards else 0.0
    mean_ov = float(np.mean(overlaps)) if overlaps else 0.0

    print(
        "PPR runtime {:.4f}s | RW runtime {:.4f}s | mean Jaccard {:.4f} | mean overlap@{:.0f} {:.4f}".format(
            ppr_sec, rw_sec, mean_j, args.k, mean_ov
        )
    )

    demo_csv = results_dir / "pubmed_ppr_topk_demo.csv"
    import csv

    with open(demo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "node_id", "jaccard", "overlap_at_k", "ppr_neighbors", "rw_neighbors",
            ],
        )
        w.writeheader()
        for row in per_node_rows:
            w.writerow(row)

    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "metapath": metapath,
        "metapath_index": args.metapath_index,
        "num_target_nodes": int(n_pick),
        "k": args.k,
        "alpha": args.alpha,
        "num_iters": args.num_iters,
        "walk_num": WALK_NUM,
        "mean_jaccard": mean_j,
        "median_jaccard": median_j,
        "mean_overlap_at_k": mean_ov,
        "ppr_runtime_sec": round(ppr_sec, 6),
        "rw_runtime_sec": round(rw_sec, 6),
        "load_data_sec": round(load_sec, 6),
        "metapath_adj_build_sec": round(build_sec, 6),
        "dgl_version": dgl_ver,
        "dgl_adj_conversion_mode": adj_mode,
        "ppr_csr_nnz": int(ppr_csr.nnz),
        "rw_csr_nnz": int(rw_csr.nnz),
        "evidence_level": "Prototype",
        "claim_boundary": "Design/demo only; not full EHGNN RW replacement",
    }
    meta_path = results_dir / "pubmed_ppr_topk_demo_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    overlap_csv = results_dir / "rw_vs_ppr_neighbor_overlap.csv"
    with open(overlap_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta.keys()))
        w.writeheader()
        w.writerow(meta)

    print("Wrote {} and {}".format(demo_csv, meta_path))


if __name__ == "__main__":
    main()
