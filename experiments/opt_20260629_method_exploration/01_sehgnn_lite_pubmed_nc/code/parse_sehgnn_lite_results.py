#!/usr/bin/env python3
"""Parse SeHGNN-lite PubMed NC log files into results CSV."""
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

P1 = EXP_ROOT / "01_sehgnn_lite_pubmed_nc"
DEFAULT_LOG = P1 / "logs/pubmed_nc_sehgnn_lite_seed42_concat.log"
DEFAULT_OUT = P1 / "results/pubmed_nc_sehgnn_lite_seed42_concat_parsed.csv"

RE_BEST_MACRO = re.compile(
    r"Best Test Macro-F1\s*:\s*([\d.]+),\s*Micro-F1\s*:\s*([\d.]+),\s*Epoch\s*:\s*(\d+)"
)
RE_FINAL = re.compile(
    r"Final Epoch\s*:\s*(\d+),\s*Final Test Macro-F1\s*:\s*([\d.]+),\s*Micro-F1\s*:\s*([\d.]+)"
)
RE_TOTAL_TRAIN = re.compile(r"Total training time:\s*([\d.]+)\s*s")
RE_TOTAL_ALL = re.compile(r"Total time \(preprocess \+ train\):\s*([\d.]+)\s*s")
RE_TEST = re.compile(
    r"Test macro f1\s*:\s*([\d.]+),\s*micro f1\s*:\s*([\d.]+)"
)


def parse_log(log_path: Path) -> dict:
    best_macro = best_micro = None
    best_epoch = None
    final_macro = final_micro = None
    final_epoch = None
    train_sec = total_sec = None
    test_lines = []

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = RE_BEST_MACRO.search(line)
        if m:
            best_macro = float(m.group(1))
            best_micro = float(m.group(2))
            best_epoch = int(m.group(3))
        m = RE_FINAL.search(line)
        if m:
            final_epoch = int(m.group(1))
            final_macro = float(m.group(2))
            final_micro = float(m.group(3))
        m = RE_TOTAL_TRAIN.search(line)
        if m:
            train_sec = float(m.group(1))
        m = RE_TOTAL_ALL.search(line)
        if m:
            total_sec = float(m.group(1))
        m = RE_TEST.search(line)
        if m:
            test_lines.append((float(m.group(1)), float(m.group(2))))

    if best_macro is None and test_lines:
        best_macro, best_micro = max(test_lines, key=lambda x: x[0])
        best_epoch = -1

    return {
        "log_path": safe_relpath(log_path, PROJECT_ROOT),
        "best_macro_f1": best_macro,
        "best_micro_f1": best_micro,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "final_macro_f1": final_macro,
        "final_micro_f1": final_micro,
        "train_time_sec": train_sec,
        "total_time_sec": total_sec,
        "num_test_eval_lines": len(test_lines),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fusion", type=str, default="concat")
    args = ap.parse_args()

    log_path = resolve_under_root(args.log, PROJECT_ROOT)
    if not log_path.is_file():
        raise FileNotFoundError("Log not found: {}".format(log_path))

    parsed = parse_log(log_path)
    row = {
        "method": "SeHGNN-lite",
        "dataset": "PubMed",
        "seed": args.seed,
        "fusion": args.fusion,
        "macro_f1": parsed["best_macro_f1"],
        "micro_f1": parsed["best_micro_f1"],
        "best_epoch": parsed["best_epoch"],
        "final_macro_f1": parsed["final_macro_f1"],
        "final_micro_f1": parsed["final_micro_f1"],
        "train_time_sec": parsed["train_time_sec"],
        "total_time_sec": parsed["total_time_sec"],
        "log_path": parsed["log_path"],
        "status": "parsed",
        "note": "from log via parse_sehgnn_lite_results.py",
    }

    out_path = resolve_under_root(args.out, PROJECT_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print("Wrote {}".format(out_path))
    print(row)


if __name__ == "__main__":
    main()
