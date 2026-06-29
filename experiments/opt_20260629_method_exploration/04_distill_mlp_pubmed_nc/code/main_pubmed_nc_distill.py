#!/usr/bin/env python3
"""
P4: EHGNN-to-MLP distillation on PubMed Node Classification.

Teacher: full EHGNN (train or load checkpoint/logits).
Student: feature-only MLP distilled from teacher soft labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_FILE = Path(__file__).resolve()
EXP_ROOT = _FILE.parents[2]
COMMON_DIR = EXP_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from path_utils import add_code_dir, bootstrap_paths, resolve_project_data_path  # noqa: E402

PROJECT_ROOT, EXP_ROOT, CODE_DIR = bootstrap_paths(__file__, task="nc")
add_code_dir(EXP_ROOT / "01_sehgnn_lite_pubmed_nc" / "code")

from distillation_losses import mixed_distill_loss  # noqa: E402
from mlp_student_model import MLPStudent, measure_inference_time  # noqa: E402
from models import EHGNN  # noqa: E402
from utils import accuracy, get_model_need, load_PubMed, random_walk_sim  # noqa: E402

try:
    from ehgnn_precompute import build_node_views  # noqa: E402
    HAS_PRECOMPUTE = True
except ImportError:
    HAS_PRECOMPUTE = False

METAPATHS_PUBMED = [
    ["dad_r", "dad"],
    ["gcd_r", "gcd"],
    ["gcd_r", "gag", "gcd"],
    ["cid_r", "cid"],
    ["cid_r", "cig", "cig_r", "cac", "cis", "cis_r", "cid"],
    ["swd_r", "sas", "swd"],
]


def parse_args():
    p = argparse.ArgumentParser(description="P4 EHGNN-to-MLP Distillation PubMed NC")
    p.add_argument("--dataset", type=str, default="PubMed")
    p.add_argument("--path", type=str, default="data",
                   help="Optional absolute data root; default PROJECT_ROOT/data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher_epochs", type=int, default=100)
    p.add_argument("--student_epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--student_hidden", type=int, default=256)
    p.add_argument("--student_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--student_lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=3000)
    p.add_argument("--val_epochs", type=int, default=5)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--walk_num", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--teacher_mode", type=str, default="train", choices=["train", "load"])
    p.add_argument(
        "--student_input",
        type=str,
        default="raw",
        choices=["raw", "precomputed", "raw_plus_precomputed"],
    )
    p.add_argument("--root_out", type=str, default="experiments/opt_20260629_method_exploration")
    p.add_argument(
        "--dry_run_runtime_check",
        action="store_true",
        help="Load PubMed, build RW matrices, run one teacher/student forward; no training or result writes",
    )
    return p.parse_args()


def build_rw_similarity_matrices(
    g,
    idx_train,
    idx_test,
    metapaths,
    walk_num,
    k,
    r_neighbor,
):
    """
    Mirror ``Node Classification/main.py`` PubMed NC RW preprocessing (lines 88–100).

    Returns train/test CSR dicts per meta-path, ``t_typess`` (list of target-type lists),
    and source node type index ``s_type``.
    """
    train_matrixs = []
    test_matrixs = []
    t_typess = []
    s_type = None
    for metapath in metapaths:
        train_matrix, t_types, s_type = random_walk_sim(
            idx_train, g, metapath, walk_num, k, r_neighbor
        )
        test_matrix, _, _ = random_walk_sim(
            idx_test, g, metapath, walk_num, k, r_neighbor
        )
        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)
        t_typess.append(t_types)
    return train_matrixs, test_matrixs, t_typess, s_type


def create_teacher_ehgnn(g, features, labels, metapaths, args, device):
    """EHGNN init aligned with ``Node Classification/main.py`` PubMed defaults."""
    out_dim = int(labels.max().item()) + 1
    return EHGNN(
        in_feat=features[0].shape[1],
        hidden=args.hidden,
        out_feat=out_dim,
        n_layer=args.n_layers,
        alpha=args.alpha,
        n_metapath=len(metapaths),
        n_types=len(g.ntypes),
        wo_l2=False,
        wo_mweight=False,
        wo_tweight=False,
        dropout=args.dropout,
    ).to(device)


TEACHER_SIGNAL_TYPE = "raw_logits"


def run_dry_runtime_check(args, device):
    """Tiny runtime path: data → RW → get_model_need → teacher/student forward; no I/O."""
    data_path = resolve_project_data_path(PROJECT_ROOT, args.path)
    g, features, labels, idx_train, idx_test = load_PubMed(
        data_path, args.dataset, False
    )
    print("[P4-DRY-RUN] load data OK")

    n_train = min(32, int(idx_train.shape[0]))
    n_test = min(32, int(idx_test.shape[0]))
    idx_train_sub = idx_train[:n_train]
    idx_test_sub = idx_test[:n_test]
    walk_num = min(args.walk_num, 5)

    train_matrixs, test_matrixs, t_typess, s_type = build_rw_similarity_matrices(
        g,
        idx_train_sub,
        idx_test_sub,
        METAPATHS_PUBMED,
        walk_num,
        args.K,
        False,
    )
    if not isinstance(t_typess, list) or len(t_typess) != len(METAPATHS_PUBMED):
        raise RuntimeError("t_typess must be a list with one entry per meta-path")
    print("[P4-DRY-RUN] teacher graph preprocessing OK")

    teacher = create_teacher_ehgnn(g, features, labels, METAPATHS_PUBMED, args, device)
    batch_size = min(8, n_train)
    batch = idx_train_sub[:batch_size]
    s_idxs, t_idxs, weightss = get_model_need(
        len(METAPATHS_PUBMED), train_matrixs, t_typess, batch
    )
    logits = teacher(
        features,
        features[s_type][batch],
        s_idxs,
        t_idxs,
        weightss,
        t_typess,
        batch.shape[0],
        device,
    )
    print(
        "[P4-DRY-RUN] teacher forward OK ({} shape={})".format(
            TEACHER_SIGNAL_TYPE, tuple(logits.shape)
        )
    )

    in_dim = int(features[s_type][idx_train_sub[:1]].shape[1])
    out_dim = int(labels.max().item()) + 1
    student = MLPStudent(
        in_dim=in_dim,
        hidden_dim=args.student_hidden,
        num_classes=out_dim,
        n_layers=args.student_layers,
        dropout=args.dropout,
    ).to(device)
    x_probe = features[s_type][idx_train_sub[:batch_size]].float()
    if device.type == "cuda":
        x_probe = x_probe.to(device)
    s_logits = student(x_probe)
    print("[P4-DRY-RUN] student forward OK (shape={})".format(tuple(s_logits.shape)))
    print("[P4-DRY-RUN] PASS")
    return 0


def out_dir(root_out: str) -> Path:
    d = Path(root_out) / "04_distill_mlp_pubmed_nc" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def teacher_paths(results: Path, seed: int) -> dict:
    return {
        "checkpoint": results / "pubmed_nc_teacher_seed{}_checkpoint.pt".format(seed),
        "logits": results / "pubmed_nc_teacher_seed{}_logits.pt".format(seed),
        "metrics": results / "pubmed_nc_teacher_seed{}_metrics.json".format(seed),
    }


def train_teacher(
    model,
    features,
    labels,
    idx_train,
    idx_test,
    train_matrixs,
    test_matrixs,
    t_typess,
    s_type,
    metapaths,
    args,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = torch.nn.NLLLoss()
    best_macro, best_micro, best_epoch = -1.0, -1.0, -1
    final_macro = final_micro = None

    for epoch in range(args.teacher_epochs):
        model.train()
        loader = torch.utils.data.DataLoader(
            idx_train, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
        for batch in loader:
            optimizer.zero_grad()
            s_idxs, t_idxs, weightss = get_model_need(
                len(metapaths), train_matrixs, t_typess, batch
            )
            logits = model(
                features,
                features[s_type][batch],
                s_idxs,
                t_idxs,
                weightss,
                t_typess,
                batch.shape[0],
                device,
            )
            log_probs = F.log_softmax(logits, dim=1)
            y = labels[batch].to(device).squeeze(1)
            loss = loss_fcn(log_probs, y)
            loss.backward()
            optimizer.step()

        if epoch % args.val_epochs == 0 and epoch != 0:
            macro, micro = eval_teacher(
                model, features, labels, idx_test, test_matrixs, t_typess, s_type, metapaths, device, args
            )
            final_macro, final_micro = macro, micro
            if macro > best_macro:
                best_macro, best_micro, best_epoch = macro, micro, epoch

    macro, micro = eval_teacher(
        model, features, labels, idx_test, test_matrixs, t_typess, s_type, metapaths, device, args
    )
    final_macro, final_micro = macro, micro
    if best_epoch < 0:
        best_macro, best_micro, best_epoch = macro, micro, args.teacher_epochs - 1

    return {
        "best_test_macro": best_macro,
        "best_test_micro": best_micro,
        "best_epoch": best_epoch,
        "final_test_macro": final_macro,
        "final_test_micro": final_micro,
    }


@torch.no_grad()
def eval_teacher(model, features, labels, idx_test, test_matrixs, t_typess, s_type, metapaths, device, args):
    model.eval()
    test_out = []
    loader = torch.utils.data.DataLoader(
        idx_test, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    for batch in loader:
        s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch)
        logits = model(
            features,
            features[s_type][batch],
            s_idxs,
            t_idxs,
            weightss,
            t_typess,
            batch.shape[0],
            device,
        )
        test_out.append(F.log_softmax(logits, dim=1).cpu())
    test_out = torch.cat(test_out, dim=0)
    y_true = labels[idx_test]
    return accuracy(test_out, y_true, args.dataset)


@torch.no_grad()
def collect_teacher_logits(
    model, features, indices, sim_matrixs, t_typess, s_type, metapaths, device, batch_size
):
    model.eval()
    parts = []
    loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=False)
    for batch in loader:
        s_idxs, t_idxs, weightss = get_model_need(len(metapaths), sim_matrixs, t_typess, batch)
        logits = model(
            features,
            features[s_type][batch],
            s_idxs,
            t_idxs,
            weightss,
            t_typess,
            batch.shape[0],
            device,
        )
        parts.append(logits.cpu())
    return torch.cat(parts, dim=0)


def build_student_features(
    student_input: str,
    features: dict,
    s_type: int,
    sim_matrixs,
    t_typess,
    node_indices: torch.Tensor,
) -> torch.Tensor:
    """Return ``[len(node_indices), in_dim]`` feature matrix for MLP student."""
    if student_input == "raw":
        return features[s_type][node_indices].float()

    if not HAS_PRECOMPUTE:
        raise RuntimeError(
            "student_input={} requires P1 ehgnn_precompute module".format(student_input)
        )

    views = build_node_views(
        features, s_type, sim_matrixs, t_typess, node_indices, include_self=False
    )
    if student_input == "precomputed":
        return torch.cat(views, dim=1)
    raw = features[s_type][node_indices].float()
    return torch.cat([raw] + views, dim=1)


def student_in_dim(student_input, features, s_type, sim_matrixs, t_typess, probe_idx):
    x = build_student_features(student_input, features, s_type, sim_matrixs, t_typess, probe_idx)
    return x.shape[1]


def train_student(student, x_train, y_train, t_logits_train, args, device):
    optimizer = torch.optim.Adam(student.parameters(), lr=args.student_lr)
    best_macro, best_micro = -1.0, -1.0
    n = x_train.shape[0]
    for _ in range(args.student_epochs):
        student.train()
        perm = torch.randperm(n)
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            x_b = x_train[idx].to(device)
            y_b = y_train[idx].to(device)
            t_b = t_logits_train[idx].to(device)
            optimizer.zero_grad()
            s_logits = student(x_b)
            loss = mixed_distill_loss(
                s_logits, t_b, y_b, args.temperature, args.kd_alpha
            )
            loss.backward()
            optimizer.step()

        student.eval()
        with torch.no_grad():
            s_logits = student(x_train.to(device))
            log_p = F.log_softmax(s_logits, dim=1)
            macro, micro = accuracy(log_p.cpu(), y_train.unsqueeze(1), args.dataset)
            if macro > best_macro:
                best_macro, best_micro = macro, micro
    return best_macro, best_micro


@torch.no_grad()
def eval_student(student, x, labels_idx, labels, dataset, device):
    student.eval()
    logits = student(x.to(device))
    log_p = F.log_softmax(logits, dim=1).cpu()
    return accuracy(log_p, labels[labels_idx].unsqueeze(1), dataset)


@torch.no_grad()
def measure_teacher_inference_ms(model, features, idx_test, test_matrixs, t_typess, s_type, metapaths, device, batch_size):
    model.eval()
    batches = list(torch.utils.data.DataLoader(idx_test, batch_size=batch_size, shuffle=False))
    if not batches:
        return 0.0
    batch = batches[0].to(device)
    s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):
        model(features, features[s_type][batch], s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device)
        sync()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model(features, features[s_type][batch], s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(times) / len(times))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(args)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.dry_run_runtime_check:
        raise SystemExit(run_dry_runtime_check(args, device))

    results = out_dir(args.root_out)
    paths = teacher_paths(results, args.seed)
    data_path = resolve_project_data_path(PROJECT_ROOT, args.path)
    print("data_path:", data_path)

    t0 = time.perf_counter()
    g, features, labels, idx_train, idx_test = load_PubMed(data_path, args.dataset, False)
    metapaths = METAPATHS_PUBMED
    load_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    train_matrixs, test_matrixs, t_typess, s_type = build_rw_similarity_matrices(
        g, idx_train, idx_test, metapaths, args.walk_num, args.K, False
    )
    sim_sec = time.perf_counter() - t1
    preprocess_sec = load_sec + sim_sec

    teacher = create_teacher_ehgnn(g, features, labels, metapaths, args, device)

    if args.teacher_mode == "load":
        if not paths["checkpoint"].is_file() or not paths["logits"].is_file():
            raise FileNotFoundError(
                "teacher_mode=load but missing checkpoint/logits under {}. "
                "Run with --teacher_mode train first.".format(results)
            )
        ckpt = torch.load(paths["checkpoint"], map_location=device)
        teacher.load_state_dict(ckpt["state_dict"])
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        logits_pack = torch.load(paths["logits"], map_location="cpu")
        teacher_train_logits = logits_pack["train_logits"]
        teacher_test_logits = logits_pack["test_logits"]
        print("Loaded teacher from {}".format(results))
    else:
        t_train_start = time.perf_counter()
        metrics = train_teacher(
            teacher, features, labels, idx_train, idx_test,
            train_matrixs, test_matrixs, t_typess, s_type, metapaths, args, device,
        )
        teacher_train_sec = time.perf_counter() - t_train_start
        metrics["teacher_train_time_sec"] = teacher_train_sec

        teacher_train_logits = collect_teacher_logits(
            teacher, features, idx_train, train_matrixs, t_typess, s_type, metapaths, device, args.batch_size
        )
        teacher_test_logits = collect_teacher_logits(
            teacher, features, idx_test, test_matrixs, t_typess, s_type, metapaths, device, args.batch_size
        )

        torch.save({"state_dict": teacher.state_dict(), "args": vars(args)}, paths["checkpoint"])
        torch.save(
            {
                "train_indices": idx_train,
                "test_indices": idx_test,
                "train_logits": teacher_train_logits,
                "test_logits": teacher_test_logits,
                "teacher_signal_type": TEACHER_SIGNAL_TYPE,
                "note": "teacher raw logits before log_softmax",
            },
            paths["logits"],
        )
        paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Saved teacher checkpoint and logits to {}".format(results))

    teacher_macro = metrics.get("best_test_macro", metrics.get("final_test_macro"))
    teacher_micro = metrics.get("best_test_micro", metrics.get("final_test_micro"))

    x_train = build_student_features(
        args.student_input, features, s_type, train_matrixs, t_typess, idx_train
    )
    x_test = build_student_features(
        args.student_input, features, s_type, test_matrixs, t_typess, idx_test
    )
    in_dim = student_in_dim(
        args.student_input, features, s_type, train_matrixs, t_typess, idx_train[:1]
    )
    y_train = labels[idx_train].squeeze(1)
    out_dim = int(labels.max().item()) + 1

    student = MLPStudent(
        in_dim=in_dim,
        hidden_dim=args.student_hidden,
        num_classes=out_dim,
        n_layers=args.student_layers,
        dropout=args.dropout,
    ).to(device)

    t_stu = time.perf_counter()
    train_student(student, x_train, y_train, teacher_train_logits, args, device)
    student_train_sec = time.perf_counter() - t_stu

    student_macro, student_micro = eval_student(
        student, x_test, idx_test, labels, args.dataset, device
    )

    teacher_inf_ms = measure_teacher_inference_ms(
        teacher, features, idx_test, test_matrixs, t_typess, s_type, metapaths, device, args.batch_size
    )
    student_inf_ms = measure_inference_time(student, x_test, repeat=20, device=device)
    speedup = teacher_inf_ms / student_inf_ms if student_inf_ms > 0 else None

    total_sec = preprocess_sec + metrics.get("teacher_train_time_sec", 0) + student_train_sec

    print("Teacher best macro={:.4f} micro={:.4f}".format(teacher_macro, teacher_micro))
    print("Student test macro={:.4f} micro={:.4f}".format(student_macro, student_micro))
    print("Teacher inference {:.3f} ms/batch | Student {:.3f} ms/full-test | speedup {:.2f}x".format(
        teacher_inf_ms, student_inf_ms, speedup or 0))

    csv_path = results / "pubmed_nc_distill_student_seed{}.csv".format(args.seed)
    row = {
        "method": "EHGNN-to-MLP Distillation",
        "dataset": args.dataset,
        "seed": args.seed,
        "student_input": args.student_input,
        "teacher_mode": args.teacher_mode,
        "teacher_macro_f1": teacher_macro,
        "teacher_micro_f1": teacher_micro,
        "student_macro_f1": student_macro,
        "student_micro_f1": student_micro,
        "teacher_inference_time_ms": round(teacher_inf_ms, 4),
        "student_inference_time_ms": round(student_inf_ms, 4),
        "speedup": round(speedup, 4) if speedup else "",
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "temperature": args.temperature,
        "kd_alpha": args.kd_alpha,
        "total_time_sec": round(total_sec, 4),
        "status": "complete",
        "note": "teacher forward ms=one test batch; student ms=full test tensor; excludes RW preprocess",
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print("Results saved to {}".format(csv_path))


if __name__ == "__main__":
    main()
