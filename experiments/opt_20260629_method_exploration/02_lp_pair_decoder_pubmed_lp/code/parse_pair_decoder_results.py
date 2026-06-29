#!/usr/bin/env python3
"""Parse P2 LP Pair Decoder log files into results CSV."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_FILE = Path(__file__).resolve()
EXP_ROOT = _FILE.parents[2]
PROJECT_ROOT = _FILE.parents[4]
COMMON_DIR = EXP_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from path_utils import resolve_under_root, safe_relpath  # noqa: E402

P2 = EXP_ROOT / "02_lp_pair_decoder_pubmed_lp"
DEFAULT_LOG = P2 / "logs/pubmed_lp_pair_decoder_seed42_pair_mlp.log"
DEFAULT_OUT = P2 / "results/pubmed_lp_pair_decoder_seed42_pair_mlp_parsed.csv"

RE_BEST = re.compile(
    r"Best Test AUC\s*:\s*([\d.]+),\s*AP\s*:\s*([\d.]+),\s*Epoch\s*:\s*(-?\d+)"
)
RE_FINAL = re.compile(
    r"Final Epoch\s*:\s*(-?\d+),\s*Final Test AUC\s*:\s*([\d.]+),\s*AP\s*:\s*([\d.]+)"
)
RE_TOTAL = re.compile(r"Total training time:\s*([\d.]+)\s*s")
RE_TOTAL_ALL = re.compile(r"Total time \(preprocess \+ train\):\s*([\d.]+)\s*s")
RE_LOAD = re.compile(r"Done Load Data, Running time:\s*([\d.]+)\s*Seconds")
RE_SIM = re.compile(r"Done my sim, Running time:\s*([\d.]+)\s*Seconds")


def parse_log(log_path: Path) -> dict:
    best_auc = best_ap = None
    best_epoch = None
    final_auc = final_ap = None
    final_epoch = None
    train_sec = total_sec = load_sec = sim_sec = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_BEST.search(line)
        if m:
            best_auc, best_ap, best_epoch = float(m.group(1)), float(m.group(2)), int(m.group(3))
        m = RE_FINAL.search(line)
        if m:
            final_epoch, final_auc, final_ap = int(m.group(1)), float(m.group(2)), float(m.group(3))
        m = RE_TOTAL.search(line)
        if m:
            train_sec = float(m.group(1))
        m = RE_TOTAL_ALL.search(line)
        if m:
            total_sec = float(m.group(1))
        m = RE_LOAD.search(line)
        if m:
            load_sec = float(m.group(1))
        m = RE_SIM.search(line)
        if m:
            sim_sec = float(m.group(1))

    preprocess = None
    if load_sec is not None and sim_sec is not None:
        preprocess = load_sec + sim_sec

    return {
        "best_auc": best_auc,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "final_auc": final_auc,
        "final_ap": final_ap,
        "final_epoch": final_epoch,
        "train_time_sec": train_sec,
        "total_time_sec": total_sec,
        "load_data_time_sec": load_sec,
        "sim_time_sec": sim_sec,
        "preprocess_time_sec": preprocess,
        "log_path": safe_relpath(log_path, PROJECT_ROOT),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decoder", type=str, default="pair_mlp")
    args = ap.parse_args()

    log_path = resolve_under_root(args.log, PROJECT_ROOT)
    if not log_path.is_file():
        raise FileNotFoundError("Log not found: {}".format(log_path))

    p = parse_log(log_path)
    row = {
        "method": "LP Pair Decoder",
        "dataset": "PubMed",
        "seed": args.seed,
        "decoder": args.decoder,
        "best_auc": p["best_auc"],
        "best_ap": p["best_ap"],
        "best_epoch": p["best_epoch"],
        "final_auc": p["final_auc"],
        "final_ap": p["final_ap"],
        "total_time_sec": p["total_time_sec"],
        "preprocess_time_sec": p["preprocess_time_sec"],
        "train_time_sec": p["train_time_sec"],
        "load_data_time_sec": p["load_data_time_sec"],
        "sim_time_sec": p["sim_time_sec"],
        "log_path": p["log_path"],
        "status": "parsed",
        "note": "parsed from log",
    }

    out_path = resolve_under_root(args.out, PROJECT_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print("Wrote {}".format(out_path))
    print(row)


if __name__ == "__main__":
    main()
