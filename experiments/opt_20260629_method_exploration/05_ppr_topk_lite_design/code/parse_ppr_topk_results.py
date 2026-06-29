#!/usr/bin/env python3
"""Parse P5 logs and merge into ppr_topk_summary.csv."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMMON = ROOT / "experiments/opt_20260629_method_exploration/common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from path_utils import resolve_under_root, safe_relpath  # noqa: E402
P5 = ROOT / "experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design"
RESULTS = P5 / "results"

RE_SUMMARY = re.compile(
    r"PPR runtime ([\d.]+)s \| RW runtime ([\d.]+)s \| mean Jaccard ([\d.]+) \| mean overlap@(\d+) ([\d.]+)"
)


def parse_log(log_path: Path) -> dict:
    out = {}
    if not log_path.is_file():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_SUMMARY.search(line)
        if m:
            out["ppr_runtime_sec"] = float(m.group(1))
            out["rw_runtime_sec"] = float(m.group(2))
            out["mean_jaccard"] = float(m.group(3))
            out["mean_overlap_at_k"] = float(m.group(5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=str(P5 / "logs/pubmed_ppr_topk_demo_seed42.log"))
    ap.add_argument("--meta", type=str, default=str(RESULTS / "pubmed_ppr_topk_demo_meta.json"))
    ap.add_argument("--out", type=str, default=str(RESULTS / "ppr_topk_summary.csv"))
    args = ap.parse_args()

    row = parse_log(resolve_under_root(args.log, ROOT))
    meta_path = resolve_under_root(args.meta, ROOT)
    if meta_path.is_file():
        row.update(json.loads(meta_path.read_text(encoding="utf-8")))

    if not row:
        row = {"status": "missing", "note": "no log or meta found"}

    out_path = resolve_under_root(args.out, ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
