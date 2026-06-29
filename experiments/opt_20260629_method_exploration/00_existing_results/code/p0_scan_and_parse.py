#!/usr/bin/env python3
"""P0: Scan existing reproduction logs and produce summary CSV, figures, and inventory."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]  # project root (EHGNN)
EXP = Path(__file__).resolve().parents[2]  # opt_20260629_method_exploration
COMMON = EXP / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from path_utils import safe_relpath  # noqa: E402

P0 = EXP / "00_existing_results"
SEEDS = [42, 3407, 2026, 6666, 8888]

SCAN_PATHS = [
    "server_results/",
    "server_results/2026-05-16_core_reproduction/",
    "server_results/2026-06-02_/",
    "Node Classification/results/",
    "Link Prediction/results/",
    "MAG240M/results/",
    "Link Prediction/results/dblp_lp_5seeds_val1000/logs/",
]

DBLP_LP_LOG_DIRS = [
    ROOT / "server_results/2026-06-02_/Link Prediction/results/dblp_lp_5seeds_val1000/logs",
    ROOT / "Link Prediction/results/dblp_lp_5seeds_val1000/logs",
    ROOT / "server_results/2026-05-16_core_reproduction/Link Prediction/results/dblp_lp_5seeds",
]


def ensure_dirs():
    subdirs = [
        "00_existing_results/code",
        "00_existing_results/logs",
        "00_existing_results/results",
        "00_existing_results/figs",
        "01_sehgnn_lite_pubmed_nc/code",
        "01_sehgnn_lite_pubmed_nc/configs",
        "01_sehgnn_lite_pubmed_nc/logs",
        "01_sehgnn_lite_pubmed_nc/results",
        "01_sehgnn_lite_pubmed_nc/figs",
        "02_lp_pair_decoder_pubmed_lp/code",
        "02_lp_pair_decoder_pubmed_lp/configs",
        "02_lp_pair_decoder_pubmed_lp/logs",
        "02_lp_pair_decoder_pubmed_lp/results",
        "02_lp_pair_decoder_pubmed_lp/figs",
        "03_sampled_lp_training_pubmed_lp/code",
        "03_sampled_lp_training_pubmed_lp/configs",
        "03_sampled_lp_training_pubmed_lp/logs",
        "03_sampled_lp_training_pubmed_lp/results",
        "03_sampled_lp_training_pubmed_lp/figs",
        "04_distill_mlp_pubmed_nc/code",
        "04_distill_mlp_pubmed_nc/configs",
        "04_distill_mlp_pubmed_nc/logs",
        "04_distill_mlp_pubmed_nc/results",
        "04_distill_mlp_pubmed_nc/figs",
        "05_ppr_topk_lite_design/code",
        "05_ppr_topk_lite_design/configs",
        "05_ppr_topk_lite_design/logs",
        "05_ppr_topk_lite_design/results",
        "05_ppr_topk_lite_design/figs",
        "server_scripts",
        "summary",
    ]
    for sd in subdirs:
        (EXP / sd).mkdir(parents=True, exist_ok=True)


def run_git(cmd: list[str]) -> str:
    try:
        r = subprocess.run(
            ["git"] + cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def check_python_pkg(name: str, code: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            return r.stdout.strip()
        return f"NOT AVAILABLE: {(r.stderr or r.stdout).strip()}"
    except Exception as e:
        return f"ERROR: {e}"


def parse_dblp_lp_log(log_path: Path, seed: int | None = None) -> dict:
    text_flags = {"has_oom": False, "has_traceback": False, "has_cuda_error": False}
    load_data_time = sim_time = total_time = None
    best_auc = best_ap = best_epoch = None
    final_epoch = final_auc = final_ap = None
    eval_times: list[float] = []

    re_load = re.compile(r"Done Load Data, Running time: ([\d.]+) Seconds")
    re_sim = re.compile(r"Done my sim, Running time: ([\d.]+) Seconds")
    re_best = re.compile(
        r"Best Test AUC : ([\d.]+), AP : ([\d.]+), Epoch : (\d+)"
    )
    re_final = re.compile(
        r"Final Epoch : (\d+), Final Test AUC : ([\d.]+), AP : ([\d.]+)"
    )
    re_total = re.compile(r"Total training time: ([\d.]+) s")
    re_eval = re.compile(
        r"Test auc : [\d.]+, precision : [\d.]+, Time : ([\d.]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            low = line.lower()
            if "oom" in low or "out of memory" in low:
                text_flags["has_oom"] = True
            if "traceback" in low:
                text_flags["has_traceback"] = True
            if "cuda error" in low or "cuda out of memory" in low:
                text_flags["has_cuda_error"] = True

            m = re_load.search(line)
            if m:
                load_data_time = float(m.group(1))
            m = re_sim.search(line)
            if m:
                sim_time = float(m.group(1))
            m = re_best.search(line)
            if m:
                best_auc = float(m.group(1))
                best_ap = float(m.group(2))
                best_epoch = int(m.group(3))
            m = re_final.search(line)
            if m:
                final_epoch = int(m.group(1))
                final_auc = float(m.group(2))
                final_ap = float(m.group(3))
            m = re_total.search(line)
            if m:
                total_time = float(m.group(1))
            m = re_eval.search(line)
            if m:
                eval_times.append(float(m.group(1)))

    is_complete = (
        final_epoch == 99
        and final_auc is not None
        and total_time is not None
        and not text_flags["has_oom"]
        and not text_flags["has_traceback"]
        and not text_flags["has_cuda_error"]
    )

    mean_eval = statistics.mean(eval_times) if eval_times else None
    note_parts = []
    if text_flags["has_oom"]:
        note_parts.append("OOM")
    if text_flags["has_traceback"]:
        note_parts.append("Traceback")
    if text_flags["has_cuda_error"]:
        note_parts.append("CUDA error")
    if final_epoch is not None and final_epoch != 99:
        note_parts.append(f"final_epoch={final_epoch}")
    if "diag" in log_path.name or "_e2" in log_path.name:
        note_parts.append("diagnostic")

    return {
        "seed": seed,
        "log_path": safe_relpath(log_path, ROOT),
        "is_complete": is_complete,
        "has_oom": text_flags["has_oom"],
        "has_traceback": text_flags["has_traceback"],
        "best_auc": best_auc,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "final_auc": final_auc,
        "final_ap": final_ap,
        "total_training_time_sec": total_time,
        "total_training_time_hour": total_time / 3600 if total_time else None,
        "load_data_time_sec": load_data_time,
        "sim_time_sec": sim_time,
        "mean_eval_time_sec": mean_eval,
        "num_eval_calls": len(eval_times),
        "note": "; ".join(note_parts) if note_parts else "",
    }


def find_dblp_lp_logs() -> dict[int, Path]:
    pattern = re.compile(
        r"dblp_lp_seed(\d+)_val1000_log100_skipmetric_e100\.log$"
    )
    found: dict[int, Path] = {}
    for d in DBLP_LP_LOG_DIRS:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            m = pattern.match(p.name)
            if m:
                seed = int(m.group(1))
                found[seed] = p
    return found


def scan_all_logs() -> list[dict]:
    inventory = []
    keywords = ["smoke", "diag", "fail", "oom", "traceback"]
    for base_rel in SCAN_PATHS:
        base = ROOT / base_rel.rstrip("/")
        if not base.exists():
            inventory.append(
                {
                    "scan_path": base_rel,
                    "exists": False,
                    "files": [],
                }
            )
            continue
        files = []
        for p in base.rglob("*"):
            if p.is_file() and (
                p.suffix in {".log", ".txt", ".csv"}
                or "summary" in p.name.lower()
            ):
                rel = safe_relpath(p, ROOT)
                tag = []
                name_low = p.name.lower()
                for kw in keywords:
                    if kw in name_low or kw in rel.lower():
                        tag.append(kw)
                files.append({"path": rel, "size_bytes": p.stat().st_size, "tags": tag})
        inventory.append(
            {"scan_path": base_rel, "exists": True, "file_count": len(files), "files": files}
        )
    return inventory


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def compute_stats(rows: list[dict]) -> list[dict]:
    complete = [r for r in rows if r.get("is_complete")]
    if not complete:
        return [{"metric": "N/A", "mean": "", "std": "", "n": 0, "note": "no complete runs"}]

    def ms(key):
        vals = [r[key] for r in complete if r.get(key) is not None]
        if len(vals) < 2:
            m = statistics.mean(vals) if vals else None
            return m, 0.0, len(vals)
        return statistics.mean(vals), statistics.stdev(vals), len(vals)

    stats = []
    for key, label in [
        ("best_auc", "best_auc"),
        ("best_ap", "best_ap"),
        ("final_auc", "final_auc"),
        ("final_ap", "final_ap"),
        ("total_training_time_sec", "total_training_time_sec"),
        ("total_training_time_hour", "total_training_time_hour"),
        ("load_data_time_sec", "load_data_time_sec"),
        ("sim_time_sec", "sim_time_sec"),
        ("mean_eval_time_sec", "mean_eval_time_sec"),
    ]:
        m, s, n = ms(key)
        stats.append(
            {
                "metric": label,
                "mean": round(m, 6) if m is not None else "",
                "std": round(s, 6) if s is not None else "",
                "n": n,
                "note": "complete seeds only (Final Epoch 99)",
            }
        )
    return stats


def plot_dblp_lp(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    complete = sorted(
        [r for r in rows if r.get("is_complete")],
        key=lambda x: x["seed"],
    )
    if not complete:
        return []

    seeds = [r["seed"] for r in complete]
    generated = []

    # 1. final AUC by seed
    if all(r.get("final_auc") is not None for r in complete):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([str(s) for s in seeds], [r["final_auc"] for r in complete], color="#4C72B0")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random baseline 0.5")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Final Test AUC")
        ax.set_title("DBLP LP (val1000): Final Test AUC by Seed")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "dblp_lp_final_auc_by_seed.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p.name)

    # 2. best vs final AUC
    if all(r.get("best_auc") is not None and r.get("final_auc") is not None for r in complete):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(seeds))
        w = 0.35
        ax.bar([i - w / 2 for i in x], [r["best_auc"] for r in complete], w, label="Best AUC", color="#55A868")
        ax.bar([i + w / 2 for i in x], [r["final_auc"] for r in complete], w, label="Final AUC", color="#C44E52")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(s) for s in seeds])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Test AUC")
        ax.set_title("DBLP LP (val1000): Best vs Final Test AUC")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "dblp_lp_best_vs_final_auc.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p.name)

    # 3. runtime by seed
    if all(r.get("total_training_time_hour") is not None for r in complete):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([str(s) for s in seeds], [r["total_training_time_hour"] for r in complete], color="#8172B3")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Total Training Time (hours)")
        ax.set_title("DBLP LP (val1000): Total Training Time by Seed")
        fig.tight_layout()
        p = out_dir / "dblp_lp_runtime_by_seed.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p.name)

    # 4. runtime breakdown (stacked mean across seeds)
    load_vals = [r["load_data_time_sec"] for r in complete if r.get("load_data_time_sec")]
    sim_vals = [r["sim_time_sec"] for r in complete if r.get("sim_time_sec")]
    total_vals = [r["total_training_time_sec"] for r in complete if r.get("total_training_time_sec")]
    if load_vals and sim_vals and total_vals:
        mean_load = statistics.mean(load_vals)
        mean_sim = statistics.mean(sim_vals)
        mean_total = statistics.mean(total_vals)
        mean_train = max(mean_total - mean_load - mean_sim, 0)
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["Load Data", "Random Walk Sim", "Training Loop"]
        hours = [mean_load / 3600, mean_sim / 3600, mean_train / 3600]
        ax.bar(labels, hours, color=["#64B5CD", "#4C72B0", "#DD8452"])
        ax.set_ylabel("Mean Time (hours)")
        ax.set_title("DBLP LP (val1000): Mean Runtime Breakdown (5 complete seeds)")
        fig.tight_layout()
        p = out_dir / "dblp_lp_runtime_breakdown.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p.name)

    return generated


def load_core_summaries() -> dict:
    summaries = {}
    paths = {
        "pubmed_nc": ROOT / "server_results/2026-06-02_/Node Classification/results/pubmed_5seeds/pubmed_5seeds_summary.csv",
        "dblp_nc": ROOT / "server_results/2026-06-02_/Node Classification/results/dblp_5seeds/summary.csv",
        "yelp_nc": ROOT / "server_results/2026-06-02_/Node Classification/results/yelp_5seeds/summary.csv",
        "pubmed_lp": ROOT / "server_results/2026-06-02_/Link Prediction/results/pubmed_lp_5seeds/summary.csv",
    }
    for k, p in paths.items():
        if p.exists():
            with open(p, encoding="utf-8", errors="replace") as f:
                summaries[k] = list(csv.DictReader(f))
        else:
            summaries[k] = None
    return summaries


def main():
    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- DBLP LP parse ---
    log_map = find_dblp_lp_logs()
    dblp_rows = []
    for seed in SEEDS:
        if seed in log_map:
            dblp_rows.append(parse_dblp_lp_log(log_map[seed], seed))
        else:
            dblp_rows.append(
                {
                    "seed": seed,
                    "log_path": "",
                    "is_complete": False,
                    "has_oom": False,
                    "has_traceback": False,
                    "best_auc": None,
                    "best_ap": None,
                    "best_epoch": None,
                    "final_epoch": None,
                    "final_auc": None,
                    "final_ap": None,
                    "total_training_time_sec": None,
                    "total_training_time_hour": None,
                    "load_data_time_sec": None,
                    "sim_time_sec": None,
                    "mean_eval_time_sec": None,
                    "note": "log not found locally",
                }
            )

    summary_fields = [
        "seed", "log_path", "is_complete", "has_oom", "has_traceback",
        "best_auc", "best_ap", "best_epoch", "final_epoch", "final_auc", "final_ap",
        "total_training_time_sec", "total_training_time_hour",
        "load_data_time_sec", "sim_time_sec", "mean_eval_time_sec", "note",
    ]
    write_csv(P0 / "results" / "dblp_lp_5seed_summary.csv", summary_fields, dblp_rows)
    stats_rows = compute_stats(dblp_rows)
    write_csv(
        P0 / "results" / "dblp_lp_5seed_stats.csv",
        ["metric", "mean", "std", "n", "note"],
        stats_rows,
    )

    # runtime breakdown per seed
    rb_rows = []
    for r in dblp_rows:
        if not r.get("total_training_time_sec"):
            continue
        load_t = r.get("load_data_time_sec") or 0
        sim_t = r.get("sim_time_sec") or 0
        train_t = max(r["total_training_time_sec"] - load_t - sim_t, 0)
        rb_rows.append(
            {
                "seed": r["seed"],
                "load_data_sec": load_t,
                "sim_sec": sim_t,
                "train_loop_sec": train_t,
                "eval_total_est_sec": (r.get("mean_eval_time_sec") or 0) * (r.get("num_eval_calls") or 0),
                "total_sec": r["total_training_time_sec"],
            }
        )
    write_csv(
        P0 / "results" / "dblp_lp_runtime_breakdown.csv",
        ["seed", "load_data_sec", "sim_sec", "train_loop_sec", "eval_total_est_sec", "total_sec"],
        rb_rows,
    )

    inventory = scan_all_logs()
    with open(P0 / "results" / "scan_inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    core = load_core_summaries()
    with open(P0 / "results" / "core_reproduction_summary.json", "w", encoding="utf-8") as f:
        json.dump(core, f, indent=2, ensure_ascii=False)

    figs = plot_dblp_lp(dblp_rows, P0 / "figs")

    # --- summary CSVs ---
    all_results = []
    for r in dblp_rows:
        if r.get("is_complete"):
            all_results.append(
                {
                    "method": "P0",
                    "task": "Link Prediction",
                    "dataset": "DBLP",
                    "seed": r["seed"],
                    "macro_f1": "",
                    "micro_f1": "",
                    "auc": r["final_auc"],
                    "ap": r["final_ap"],
                    "best_epoch": r["best_epoch"],
                    "final_metric": r["final_auc"],
                    "runtime_sec": r["total_training_time_sec"],
                    "preprocess_time_sec": (r.get("load_data_time_sec") or 0) + (r.get("sim_time_sec") or 0),
                    "train_time_sec": max(
                        (r.get("total_training_time_sec") or 0)
                        - (r.get("load_data_time_sec") or 0)
                        - (r.get("sim_time_sec") or 0),
                        0,
                    ),
                    "eval_time_sec": r.get("mean_eval_time_sec"),
                    "status": "complete",
                    "note": "val1000 log100 skipmetric e100; existing reproduction log",
                }
            )
    write_csv(
        EXP / "summary" / "all_results.csv",
        [
            "method", "task", "dataset", "seed", "macro_f1", "micro_f1", "auc", "ap",
            "best_epoch", "final_metric", "runtime_sec", "preprocess_time_sec",
            "train_time_sec", "eval_time_sec", "status", "note",
        ],
        all_results,
    )

    all_runtime = []
    for r in dblp_rows:
        if r.get("total_training_time_sec"):
            all_runtime.append(
                {
                    "method": "P0",
                    "task": "Link Prediction",
                    "dataset": "DBLP",
                    "seed": r["seed"],
                    "load_data_sec": r.get("load_data_time_sec"),
                    "sim_sec": r.get("sim_time_sec"),
                    "train_loop_sec": max(
                        r["total_training_time_sec"]
                        - (r.get("load_data_time_sec") or 0)
                        - (r.get("sim_time_sec") or 0),
                        0,
                    ),
                    "mean_eval_sec": r.get("mean_eval_time_sec"),
                    "total_sec": r["total_training_time_sec"],
                    "total_hour": r.get("total_training_time_hour"),
                    "status": "complete" if r.get("is_complete") else "incomplete",
                    "note": r.get("note", ""),
                }
            )
    write_csv(
        EXP / "summary" / "all_runtime.csv",
        [
            "method", "task", "dataset", "seed", "load_data_sec", "sim_sec",
            "train_loop_sec", "mean_eval_sec", "total_sec", "total_hour", "status", "note",
        ],
        all_runtime,
    )

    complete_count = sum(1 for r in dblp_rows if r.get("is_complete"))
    print(f"P0 done: {complete_count}/5 DBLP LP seeds complete, figs: {figs}")
    return {
        "ts": ts,
        "dblp_rows": dblp_rows,
        "stats_rows": stats_rows,
        "figs": figs,
        "inventory": inventory,
        "core": core,
        "complete_count": complete_count,
    }


if __name__ == "__main__":
    main()
