#!/usr/bin/env python3
"""Plot P4 teacher vs student F1 and inference time."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
P4 = ROOT / "experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc"
FIGS = P4 / "figs"
BASELINE = P4 / "configs/ehgnn_pubmed_nc_baseline_seed42.json"


def load_row(csv_path: Path) -> dict | None:
    if not csv_path.is_file():
        return None
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def bar2(labels, vals, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#4C72B0", "#55A868"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved {}".format(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(P4 / "results/pubmed_nc_distill_student_seed42.csv"))
    ap.add_argument("--out_dir", type=str, default=str(FIGS))
    args = ap.parse_args()

    row = load_row(Path(args.csv))
    out_dir = Path(args.out_dir)
    if row is None:
        print("No distill CSV — skip plots.")
        return

    def f(key):
        v = row.get(key)
        return float(v) if v not in (None, "") else None

    t_macro, s_macro = f("teacher_macro_f1"), f("student_macro_f1")
    t_micro, s_micro = f("teacher_micro_f1"), f("student_micro_f1")
    t_ms, s_ms = f("teacher_inference_time_ms"), f("student_inference_time_ms")
    speedup = f("speedup")

    if t_macro is not None and s_macro is not None:
        bar2(
            ["Teacher", "Student"],
            [t_macro, s_macro],
            "Macro-F1",
            "P4 Distillation: Macro-F1",
            out_dir / "distill_teacher_student_macro_f1.png",
        )
    if t_micro is not None and s_micro is not None:
        bar2(
            ["Teacher", "Student"],
            [t_micro, s_micro],
            "Micro-F1",
            "P4 Distillation: Micro-F1",
            out_dir / "distill_teacher_student_micro_f1.png",
        )
    if t_ms is not None and s_ms is not None:
        bar2(
            ["Teacher", "Student"],
            [t_ms, s_ms],
            "Inference time (ms)",
            "P4 Distillation: Inference Time",
            out_dir / "distill_inference_time.png",
        )
        if speedup is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Teacher/Student"], [speedup], color="#8172B3")
            ax.set_ylabel("Speedup (x)")
            ax.set_title("P4 Distillation: Inference Speedup")
            fig.tight_layout()
            fig.savefig(out_dir / "distill_speedup.png", dpi=150)
            plt.close(fig)
            print("Saved {}".format(out_dir / "distill_speedup.png"))


if __name__ == "__main__":
    main()
