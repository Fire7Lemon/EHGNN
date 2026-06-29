#!/usr/bin/env python3
"""Plot P5 PPR-TopK demo results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_FILE = Path(__file__).resolve()
EXP_ROOT = _FILE.parents[2]
COMMON_DIR = EXP_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from plot_utils import require_matplotlib_or_skip  # noqa: E402

plt = require_matplotlib_or_skip("plot_ppr_topk_results.py")

P5 = EXP_ROOT / "05_ppr_topk_lite_design"
FIGS = P5 / "figs"
RESULTS = P5 / "results"


def load_demo_jaccards(path: Path) -> list[float]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [float(r["jaccard"]) for r in csv.DictReader(f)]


def load_meta(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_csv", type=str, default=str(RESULTS / "pubmed_ppr_topk_demo.csv"))
    ap.add_argument("--meta_json", type=str, default=str(RESULTS / "pubmed_ppr_topk_demo_meta.json"))
    ap.add_argument("--out_dir", type=str, default=str(FIGS))
    args = ap.parse_args()

    meta = load_meta(Path(args.meta_json))
    jaccards = load_demo_jaccards(Path(args.demo_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_rw = meta is not None and meta.get("rw_runtime_sec") is not None

    if has_rw and meta.get("mean_jaccard") is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Mean Jaccard"], [meta["mean_jaccard"]], color="#4C72B0")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Jaccard")
        ax.set_title("RW vs PPR Top-{} Neighbor Overlap (mean)".format(meta.get("k", 20)))
        fig.tight_layout()
        p = out_dir / "rw_vs_ppr_topk_overlap.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("Saved {}".format(p))

    if meta and meta.get("ppr_runtime_sec") is not None:
        labels, vals, colors = ["PPR"], [meta["ppr_runtime_sec"]], ["#55A868"]
        if has_rw:
            labels.append("RW")
            vals.append(meta["rw_runtime_sec"])
            colors.append("#4C72B0")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("PPR-TopK vs RW Top-K Demo Runtime")
        fig.tight_layout()
        p = out_dir / "ppr_runtime_demo.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("Saved {}".format(p))

    if jaccards:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(jaccards, bins=30, color="#8172B3", edgecolor="white")
        ax.set_xlabel("Jaccard similarity")
        ax.set_ylabel("Count")
        ax.set_title("Per-node RW vs PPR Top-K Jaccard Distribution")
        fig.tight_layout()
        p = out_dir / "ppr_topk_jaccard_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("Saved {}".format(p))
    elif not has_rw:
        print("No RW results — skipped overlap/jaccard plots; run server demo first.")


if __name__ == "__main__":
    main()
