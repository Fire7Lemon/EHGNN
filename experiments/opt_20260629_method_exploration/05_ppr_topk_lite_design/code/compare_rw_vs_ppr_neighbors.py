#!/usr/bin/env python3
"""Aggregate RW vs PPR neighbor overlap from P5 demo outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
P5 = ROOT / "experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design"
RESULTS = P5 / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--demo_csv",
        type=str,
        default=str(RESULTS / "pubmed_ppr_topk_demo.csv"),
    )
    ap.add_argument(
        "--meta_json",
        type=str,
        default=str(RESULTS / "pubmed_ppr_topk_demo_meta.json"),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(RESULTS / "rw_vs_ppr_neighbor_overlap.csv"),
    )
    args = ap.parse_args()

    demo_path = Path(args.demo_csv)
    meta_path = Path(args.meta_json)
    row = {"status": "empty", "note": ""}

    if meta_path.is_file():
        row.update(json.loads(meta_path.read_text(encoding="utf-8")))
        row["status"] = "ok"

    if demo_path.is_file():
        with open(demo_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            j = [float(r["jaccard"]) for r in rows]
            o = [float(r["overlap_at_k"]) for r in rows]
            row["mean_jaccard_recomputed"] = float(np.mean(j))
            row["median_jaccard_recomputed"] = float(np.median(j))
            row["mean_overlap_recomputed"] = float(np.mean(o))
            row["num_nodes"] = len(rows)
    else:
        row["note"] = "demo CSV not found; PPR-only or not yet run on server"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
