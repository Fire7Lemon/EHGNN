#!/usr/bin/env python3
"""
SeHGNN-lite PubMed Node Classification runner.

Reuses Node Classification data loading, meta-path RW, and metrics without
modifying main.py / utils.py / models.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- project paths ---
CODE_DIR = Path(__file__).resolve().parent
P1_DIR = CODE_DIR.parent
EXP_ROOT = P1_DIR.parent
PROJECT_ROOT = EXP_ROOT.parent
NC_DIR = PROJECT_ROOT / "Node Classification"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(NC_DIR))

from ehgnn_precompute import build_node_views, count_parameters  # noqa: E402
from sehgnn_lite_model import build_sehgnn_lite_head  # noqa: E402
from utils import accuracy, load_PubMed, random_walk_sim  # noqa: E402

METAPATHS_PUBMED = [
    ["dad_r", "dad"],
    ["gcd_r", "gcd"],
    ["gcd_r", "gag", "gcd"],
    ["cid_r", "cid"],
    ["cid_r", "cig", "cig_r", "cac", "cis", "cis_r", "cid"],
    ["swd_r", "sas", "swd"],
]


def parse_args():
    p = argparse.ArgumentParser(description="SeHGNN-lite PubMed NC")
    p.add_argument("--dataset", type=str, default="PubMed")
    p.add_argument("--path", type=str, default="../data/",
                   help="Data root relative to Node Classification/ when cwd is NC")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--fusion", type=str, default="concat", choices=["concat", "mean"])
    p.add_argument("--batch_size", type=int, default=3000)
    p.add_argument("--val_epochs", type=int, default=5)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--walk_num", type=int, default=40)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--root_out", type=str,
                   default="experiments/opt_20260629_method_exploration")
    p.add_argument("--include_self", action="store_true", default=True,
                   help="Include raw self feature as first view")
    p.add_argument("--no_include_self", action="store_false", dest="include_self")
    return p.parse_args()


def resolve_out_dirs(root_out: str) -> dict:
    rel = Path(root_out) / "01_sehgnn_lite_pubmed_nc"
    results = rel / "results"
    results.mkdir(parents=True, exist_ok=True)
    return {"results": results}


def write_result_csv(path: Path, row: dict):
    fieldnames = [
        "method", "dataset", "seed", "fusion", "epochs", "hidden", "dropout", "lr",
        "macro_f1", "micro_f1", "best_epoch", "total_time_sec",
        "preprocess_time_sec", "train_time_sec", "num_params", "status", "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Run from project root; data path matches Node Classification/main.py default
    data_path = args.path
    if not Path(data_path).is_absolute():
        data_path = str(NC_DIR / args.path)

    print(args)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    g, features, labels, idx_train, idx_test = load_PubMed(
        data_path, args.dataset, is_normalize=False
    )
    load_sec = time.perf_counter() - t0
    print("Done Load Data, Running time: {:.4f} Seconds".format(load_sec))

    metapaths = METAPATHS_PUBMED
    print(metapaths)

    # Precompute RW sim for all labeled disease nodes (train + test)
    all_idx = torch.cat([idx_train, idx_test]).unique(sorted=True)
    t1 = time.perf_counter()
    sim_matrixs = []
    t_typess = []
    s_type = None
    for metapath in metapaths:
        sim_matrix, t_types, s_type_mp = random_walk_sim(
            all_idx, g, metapath, args.walk_num, args.K, False
        )
        sim_matrixs.append(sim_matrix)
        t_typess.append(t_types)
        if s_type is None:
            s_type = s_type_mp
        elif s_type != s_type_mp:
            print("Warning: inconsistent s_type across metapaths: {} vs {}".format(
                s_type, s_type_mp))
    sim_sec = time.perf_counter() - t1
    print("Done sim (all labeled nodes), Running time: {:.4f} Seconds".format(sim_sec))

    feat_dim = features[s_type].shape[1]
    t2 = time.perf_counter()
    train_views = build_node_views(
        features, s_type, sim_matrixs, t_typess, idx_train, args.include_self
    )
    test_views = build_node_views(
        features, s_type, sim_matrixs, t_typess, idx_test, args.include_self
    )
    precompute_sec = time.perf_counter() - t2
    preprocess_sec = load_sec + sim_sec + precompute_sec
    print(
        "Precomputed {} views x feat_dim={} for train={} test={} (view build {:.2f}s)".format(
            len(train_views), feat_dim, len(idx_train), len(idx_test), precompute_sec
        )
    )

    out_dim = int(labels.max().item()) + 1
    model = build_sehgnn_lite_head(
        fusion=args.fusion,
        in_dim=feat_dim,
        hidden_dim=args.hidden,
        num_classes=out_dim,
        dropout=args.dropout,
        n_views=len(train_views),
    ).to(device)
    n_params = count_parameters(model)
    print("SeHGNN-lite head params: {}".format(n_params))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fcn = torch.nn.NLLLoss()

    train_views_dev = [v.to(device) for v in train_views]
    test_views_dev = [v.to(device) for v in test_views]
    y_train = labels[idx_train].to(device)
    y_test = labels[idx_test]

    best_macro = -1.0
    best_micro = -1.0
    best_epoch = -1
    final_macro = None
    final_micro = None

    print("Begin Train (SeHGNN-lite, fusion={}).".format(args.fusion))
    train_begin = time.perf_counter()
    for epoch in range(args.epochs):
        ep_start = time.perf_counter()
        model.train()
        perm = torch.randperm(len(idx_train), device=device)
        loss_vals = []
        for start in range(0, len(idx_train), args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            batch_views = [v.index_select(0, batch_idx) for v in train_views_dev]
            optimizer.zero_grad()
            logits = model(batch_views)
            log_probs = F.log_softmax(logits, dim=1)
            y_batch = y_train.index_select(0, batch_idx).squeeze(1)
            loss = loss_fcn(log_probs, y_batch)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

        ep_sec = time.perf_counter() - ep_start
        print(
            "Epoch : {}, loss : {:.4f}, Running time: {:.4f} Seconds".format(
                epoch, float(np.mean(loss_vals)), ep_sec
            )
        )

        if epoch % args.val_epochs == 0 and epoch != 0:
            macro, micro = evaluate(model, test_views_dev, y_test, args.dataset, device)
            final_macro, final_micro = macro, micro
            if macro > best_macro:
                best_macro, best_micro, best_epoch = macro, micro, epoch
            print(
                "Test macro f1 : {:.4f}, micro f1 : {:.4f}".format(macro, micro)
            )

    final_epoch = args.epochs - 1 if args.epochs > 0 else -1
    last_eval = args.epochs > 0 and final_epoch != 0 and final_epoch % args.val_epochs == 0
    if args.epochs > 0 and not last_eval:
        macro, micro = evaluate(model, test_views_dev, y_test, args.dataset, device)
        final_macro, final_micro = macro, micro
        print(
            "Final epoch {} test macro f1 : {:.4f}, micro f1 : {:.4f}".format(
                final_epoch, macro, micro
            )
        )

    train_sec = time.perf_counter() - train_begin
    total_sec = preprocess_sec + train_sec

    print(
        "Best Test Macro-F1 : {:.4f}, Micro-F1 : {:.4f}, Epoch : {}".format(
            best_macro, best_micro, best_epoch
        )
    )
    print(
        "Final Epoch : {}, Final Test Macro-F1 : {:.4f}, Micro-F1 : {:.4f}".format(
            final_epoch, final_macro, final_micro
        )
    )
    print("Total training time: {:.4f} s".format(train_sec))
    print("Total time (preprocess + train): {:.4f} s".format(total_sec))

    out_dirs = resolve_out_dirs(args.root_out)
    csv_name = "pubmed_nc_sehgnn_lite_seed{}_{}.csv".format(args.seed, args.fusion)
    csv_path = out_dirs["results"] / csv_name
    row = {
        "method": "SeHGNN-lite",
        "dataset": args.dataset,
        "seed": args.seed,
        "fusion": args.fusion,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "macro_f1": best_macro,
        "micro_f1": best_micro,
        "best_epoch": best_epoch,
        "total_time_sec": round(total_sec, 4),
        "preprocess_time_sec": round(preprocess_sec, 4),
        "train_time_sec": round(train_sec, 4),
        "num_params": n_params,
        "status": "complete",
        "note": "best test macro/micro; single seed quick validation",
    }
    write_result_csv(csv_path, row)
    print("Results saved to: {}".format(csv_path))


@torch.no_grad()
def evaluate(model, test_views, y_test, dataset, device):
    model.eval()
    logits = model(test_views)
    log_probs = F.log_softmax(logits, dim=1)
    macro, micro = accuracy(log_probs, y_test, dataset)
    return macro, micro


if __name__ == "__main__":
    main()
