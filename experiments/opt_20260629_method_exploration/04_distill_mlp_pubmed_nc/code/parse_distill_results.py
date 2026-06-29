#!/usr/bin/env python3
"""Parse P4 distillation logs and merge CSV/JSON into summary."""
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
P4 = ROOT / "experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc"
RESULTS = P4 / "results"

RE_TEACHER = re.compile(r"Teacher best macro=([\d.]+) micro=([\d.]+)")
RE_STUDENT = re.compile(r"Student test macro=([\d.]+) micro=([\d.]+)")
RE_INF = re.compile(
    r"Teacher inference ([\d.]+) ms/batch \| Student ([\d.]+) ms/full-test \| speedup ([\d.]+)x"
)


def parse_log(log_path: Path) -> dict:
    out = {}
    if not log_path.is_file():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_TEACHER.search(line)
        if m:
            out["teacher_macro_f1"] = float(m.group(1))
            out["teacher_micro_f1"] = float(m.group(2))
        m = RE_STUDENT.search(line)
        if m:
            out["student_macro_f1"] = float(m.group(1))
            out["student_micro_f1"] = float(m.group(2))
        m = RE_INF.search(line)
        if m:
            out["teacher_inference_time_ms"] = float(m.group(1))
            out["student_inference_time_ms"] = float(m.group(2))
            out["speedup"] = float(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=str(P4 / "logs/pubmed_nc_distill_seed42_raw.log"))
    ap.add_argument("--csv", type=str, default=str(RESULTS / "pubmed_nc_distill_student_seed42.csv"))
    ap.add_argument("--out", type=str, default=str(RESULTS / "pubmed_nc_distill_summary.csv"))
    args = ap.parse_args()

    row = {}
    csv_path = resolve_under_root(args.csv, ROOT)
    if csv_path.is_file():
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row.update(rows[-1])

    row.update(parse_log(resolve_under_root(args.log, ROOT)))

    metrics_path = RESULTS / "pubmed_nc_teacher_seed42_metrics.json"
    if metrics_path.is_file():
        row["teacher_metrics_json"] = safe_relpath(metrics_path, ROOT)

    out_path = resolve_under_root(args.out, ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys()) if row else ["status"]
    if not row:
        row = {"status": "empty"}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
