#!/usr/bin/env python3
"""
P3: PubMed Link Prediction with sampled training protocol.

Copies Link Prediction/main.py training/eval logic; only changes how many
train indices are sampled per epoch (see sampled_lp_loader.py).
Uses original dot decoder + BCELoss(sigmoid(dot)) — no PairMLPDecoder.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
LP_DIR = PROJECT_ROOT / "Link Prediction"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(LP_DIR) not in sys.path:
    sys.path.insert(0, str(LP_DIR))

from models import EHGNN  # noqa: E402
from sampled_lp_loader import (  # noqa: E402
    compute_steps_per_epoch,
    describe_protocol,
    init_fixed_indices,
    ratio_tag,
    sample_train_indices,
)
from utils import (  # noqa: E402
    accuracy,
    get_model_need,
    load_PubMed,
    lp_rw_seeds_all_nodes,
    neg_sample,
    random_walk_sim,
)

METAPATHS_PUBMED = [
    ["dad_r", "dad"],
    ["gcd_r", "gcd"],
    ["gcd_r", "gag", "gcd"],
    ["cid_r", "cid"],
    ["cid_r", "cig", "cig_r", "cac", "cis", "cis_r", "cid"],
    ["swd_r", "sas", "swd"],
]


def parse_args():
    p = argparse.ArgumentParser(description="P3 Sampled-LP PubMed LP")
    p.add_argument("--dataset", type=str, default="PubMed")
    p.add_argument("--path", type=str, default="../data/")
    p.add_argument("--is_normalize", action="store_true")
    p.add_argument("--wo_l2", action="store_true")
    p.add_argument("--wo_mweight", action="store_true")
    p.add_argument("--wo_tweight", action="store_true")
    p.add_argument("--r_neighbor", action="store_true")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--walk_num", type=int, default=100)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--val_epochs", type=int, default=5)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--skip_batch_metrics", action="store_true")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_ratio", type=float, default=0.5)
    p.add_argument("--resample_each_epoch", action="store_true", default=True)
    p.add_argument("--no_resample_each_epoch", action="store_false", dest="resample_each_epoch")
    p.add_argument(
        "--root_out",
        type=str,
        default="experiments/opt_20260629_method_exploration",
    )
    return p.parse_args()


def write_result_csv(path: Path, row: dict):
    fields = [
        "method", "dataset", "seed", "sample_ratio", "resample_each_epoch",
        "epochs", "hidden", "dropout", "lr", "best_auc", "best_ap", "best_epoch",
        "final_auc", "final_ap", "total_time_sec", "preprocess_time_sec",
        "train_time_sec", "steps_per_epoch", "base_steps_per_epoch", "status", "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    if args.log_interval < 1:
        raise ValueError("--log_interval must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(args)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    data_path = args.path if Path(args.path).is_absolute() else str(LP_DIR / args.path)

    t0 = time.perf_counter()
    g, features, train_links, test_links, labels = load_PubMed(
        data_path, args.dataset, args.is_normalize
    )
    metapaths = METAPATHS_PUBMED
    num_sample = int(g.num_nodes("disease"))
    load_sec = time.perf_counter() - t0
    print("Done Load Data, Running time: {:.4f} Seconds".format(load_sec))
    print(metapaths)

    t1 = time.perf_counter()
    sim_matrixs = []
    t_typess = []
    s_type = None
    for metapath in metapaths:
        rw_seeds = lp_rw_seeds_all_nodes(g, metapath)
        sim_matrix, t_types, s_type_mp = random_walk_sim(
            rw_seeds, g, metapath, args.walk_num, args.K, args.r_neighbor
        )
        sim_matrixs.append(sim_matrix)
        t_typess.append(t_types)
        if s_type is None:
            s_type = s_type_mp
    sim_sec = time.perf_counter() - t1
    print("Done my sim, Running time: {:.4f} Seconds".format(sim_sec))
    preprocess_sec = load_sec + sim_sec

    index_upper = train_links.shape[0]
    protocol = describe_protocol(
        num_sample, args.sample_ratio, args.resample_each_epoch, index_upper
    )
    print(
        "Sampling protocol: object={} ratio={} resample_each_epoch={} "
        "steps/epoch={}/{} index_upper={}".format(
            protocol.sampling_object,
            protocol.sample_ratio,
            protocol.resample_each_epoch,
            protocol.steps_per_epoch,
            protocol.base_steps,
            protocol.index_upper,
        )
    )

    model = EHGNN(
        in_feat=features[0].shape[1],
        hidden=args.hidden,
        out_feat=args.hidden,
        n_layer=args.n_layers,
        alpha=args.alpha,
        n_metapath=len(metapaths),
        n_types=len(g.ntypes),
        wo_l2=args.wo_l2,
        wo_mweight=args.wo_mweight,
        wo_tweight=args.wo_tweight,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fcn = torch.nn.BCELoss()

    def evaluate_lp():
        with torch.no_grad():
            model.eval()
            test_loader = torch.utils.data.DataLoader(
                torch.arange(num_sample, dtype=torch.long),
                batch_size=4000,
                shuffle=False,
                drop_last=False,
            )
            emb = torch.FloatTensor([])
            for batch_test in test_loader:
                s_idxs, t_idxs, weightss = get_model_need(
                    len(metapaths), sim_matrixs, t_typess, batch_test
                )
                batch_out = model(
                    features,
                    features[s_type][batch_test],
                    s_idxs,
                    t_idxs,
                    weightss,
                    t_typess,
                    batch_test.shape[0],
                    device,
                )
                emb = torch.cat((emb, batch_out.cpu()), dim=0)

            pos_s_out = emb[test_links[0]]
            pos_t_out = emb[test_links[1]]
            pos = (pos_s_out * pos_t_out).sum(dim=-1)
            test_out = pos.sigmoid().cpu()
            auc, ap = accuracy(test_out, labels)
            return auc, ap

    fixed_indices = None
    if not args.resample_each_epoch:
        fixed_indices = init_fixed_indices(
            protocol.steps_per_epoch, index_upper, args.seed
        )

    best_test_auc = float("-inf")
    best_test_ap = float("-inf")
    best_epoch = -1

    print("Begin Train (Sampled-LP, dot decoder).")
    train_begin = time.perf_counter()
    for run in range(args.epochs):
        train_idx, fixed_indices = sample_train_indices(
            protocol.steps_per_epoch,
            index_upper,
            args.resample_each_epoch,
            run,
            args.seed,
            fixed_indices,
        )
        train_loader = torch.utils.data.DataLoader(
            train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
        step = 0
        for batch_train in train_loader:
            model.train()
            step += 1
            step_start = time.perf_counter()
            optimizer.zero_grad()

            pos_s = train_links[0][batch_train]
            pos_t = train_links[1][batch_train]
            neg_s, neg_t = neg_sample(pos_s, 0, num_sample)

            s_idxs, t_idxs, weightss = get_model_need(
                len(metapaths), sim_matrixs, t_typess, pos_s
            )
            pos_s_out = model(
                features, features[s_type][pos_s], s_idxs, t_idxs, weightss,
                t_typess, pos_s.shape[0], device,
            )
            s_idxs, t_idxs, weightss = get_model_need(
                len(metapaths), sim_matrixs, t_typess, pos_t
            )
            pos_t_out = model(
                features, features[s_type][pos_t], s_idxs, t_idxs, weightss,
                t_typess, pos_t.shape[0], device,
            )
            s_idxs, t_idxs, weightss = get_model_need(
                len(metapaths), sim_matrixs, t_typess, neg_s
            )
            neg_s_out = model(
                features, features[s_type][neg_s], s_idxs, t_idxs, weightss,
                t_typess, neg_s.shape[0], device,
            )
            s_idxs, t_idxs, weightss = get_model_need(
                len(metapaths), sim_matrixs, t_typess, neg_t
            )
            neg_t_out = model(
                features, features[s_type][neg_t], s_idxs, t_idxs, weightss,
                t_typess, neg_t.shape[0], device,
            )

            pos = (pos_s_out * pos_t_out).sum(dim=-1)
            neg = (neg_s_out * neg_t_out).sum(dim=-1)
            batch_out = torch.cat((pos, neg)).sigmoid()
            y_true = torch.cat(
                (
                    torch.ones(batch_train.shape[0], dtype=int),
                    torch.zeros(batch_train.shape[0], dtype=int),
                )
            )
            loss = loss_fcn(batch_out, y_true.float().to(device))
            loss_val = loss.item()

            if not args.skip_batch_metrics:
                auc, precision = accuracy(batch_out.cpu(), y_true)
            else:
                auc = precision = 0.0

            loss.backward()
            optimizer.step()
            step_end = time.perf_counter()

            if step % args.log_interval == 0:
                if args.skip_batch_metrics:
                    print(
                        "Epoch : {}, Step : {}, loss : {:.4f}, Running time: {:.4f} Seconds".format(
                            run, step, loss_val, step_end - step_start
                        )
                    )
                else:
                    print(
                        "Epoch : {}, Step : {}, loss : {:.4f}, auc : {:.4f}, precision : {:.4f}, "
                        "Running time: {:.4f} Seconds".format(
                            run, step, loss_val, auc, precision, step_end - step_start
                        )
                    )

            if step % args.val_epochs == 0 and step != 0:
                ev_start = time.perf_counter()
                test_auc, test_ap = evaluate_lp()
                ev_end = time.perf_counter()
                print(
                    "Test auc : {:.4f}, precision : {:.4f}, Time : {:.4f}".format(
                        test_auc, test_ap, ev_end - ev_start
                    )
                )
                if test_auc > best_test_auc or (
                    test_auc == best_test_auc and test_ap > best_test_ap
                ):
                    best_test_auc = test_auc
                    best_test_ap = test_ap
                    best_epoch = run

    final_epoch = args.epochs - 1 if args.epochs > 0 else -1
    final_test_auc, final_test_ap = evaluate_lp()
    train_sec = time.perf_counter() - train_begin
    total_sec = preprocess_sec + train_sec

    if best_epoch < 0:
        best_test_auc = final_test_auc
        best_test_ap = final_test_ap
        best_epoch = final_epoch

    print(
        "Best Test AUC : {:.4f}, AP : {:.4f}, Epoch : {}".format(
            best_test_auc, best_test_ap, best_epoch
        )
    )
    print(
        "Final Epoch : {}, Final Test AUC : {:.4f}, AP : {:.4f}".format(
            final_epoch, final_test_auc, final_test_ap
        )
    )
    print("Total training time: {:.4f} s".format(train_sec))
    print("Total time (preprocess + train): {:.4f} s".format(total_sec))

    tag = ratio_tag(args.sample_ratio)
    csv_path = (
        Path(args.root_out)
        / "03_sampled_lp_training_pubmed_lp"
        / "results"
        / "pubmed_lp_sampled_training_seed{}_{}.csv".format(args.seed, tag)
    )
    write_result_csv(
        csv_path,
        {
            "method": "Sampled-LP Training",
            "dataset": args.dataset,
            "seed": args.seed,
            "sample_ratio": args.sample_ratio,
            "resample_each_epoch": args.resample_each_epoch,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "best_auc": best_test_auc,
            "best_ap": best_test_ap,
            "best_epoch": best_epoch,
            "final_auc": final_test_auc,
            "final_ap": final_test_ap,
            "total_time_sec": round(total_sec, 4),
            "preprocess_time_sec": round(preprocess_sec, 4),
            "train_time_sec": round(train_sec, 4),
            "steps_per_epoch": protocol.steps_per_epoch,
            "base_steps_per_epoch": protocol.base_steps,
            "status": "complete",
            "note": (
                "sampled train_index_tensor; dot decoder; full evaluate_lp; "
                "ratio={} resample={}".format(args.sample_ratio, args.resample_each_epoch)
            ),
        },
    )
    print("Results saved to: {}".format(csv_path))


if __name__ == "__main__":
    main()
