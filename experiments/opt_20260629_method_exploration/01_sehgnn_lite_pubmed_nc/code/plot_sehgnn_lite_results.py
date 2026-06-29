#!/usr/bin/env python3
"""Plot SeHGNN-lite vs EHGNN PubMed NC baseline comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
P1 = ROOT / "experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc"
FIGS = P1 / "figs"
BASELINE_JSON = P1 / "configs/ehgnn_pubmed_baseline_seed42.json"


def load_csv_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def load_baseline(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": "EHGNN",
        "dataset": "PubMed",
        "seed": 42,
        "best_test_macro": 0.6312616652690182,
        "best_test_micro": 0.6511627906976745,
        "total_training_time_sec": 3.6633235327899456,
        "source": "server_results/2026-06-02_/Node Classification/results/pubmed_5seeds/pubmed_seed_42.txt",
        "note": "embedded fallback baseline",
    }


def bar_compare(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4C72B0", "#55A868"][: len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved {}".format(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result_csv",
        type=str,
        default=str(P1 / "results/pubmed_nc_sehgnn_lite_seed42_concat.csv"),
    )
    ap.add_argument("--baseline_json", type=str, default=str(BASELINE_JSON))
    ap.add_argument("--out_dir", type=str, default=str(FIGS))
    args = ap.parse_args()

    lite = load_csv_row(Path(args.result_csv))
    base = load_baseline(Path(args.baseline_json))
    out_dir = Path(args.out_dir)

    if lite is None:
        print("Warning: SeHGNN-lite result CSV not found; plotting baseline-only placeholders.")
        lite = {
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "total_time_sec": 0.0,
        }

    lite_macro = float(lite.get("macro_f1") or 0)
    lite_micro = float(lite.get("micro_f1") or 0)
    lite_time = float(lite.get("total_time_sec") or lite.get("train_time_sec") or 0)

    base_macro = float(base["best_test_macro"])
    base_micro = float(base["best_test_micro"])
    base_time = float(base["total_training_time_sec"])

    bar_compare(
        ["EHGNN", "SeHGNN-lite"],
        [base_macro, lite_macro],
        "Best Test Macro-F1",
        "PubMed NC seed=42: Macro-F1",
        out_dir / "sehgnn_lite_vs_ehgnn_macro_f1.png",
    )
    bar_compare(
        ["EHGNN", "SeHGNN-lite"],
        [base_micro, lite_micro],
        "Best Test Micro-F1",
        "PubMed NC seed=42: Micro-F1",
        out_dir / "sehgnn_lite_vs_ehgnn_micro_f1.png",
    )
    bar_compare(
        ["EHGNN", "SeHGNN-lite"],
        [base_time, lite_time],
        "Total Time (seconds)",
        "PubMed NC seed=42: Runtime",
        out_dir / "sehgnn_lite_vs_ehgnn_runtime.png",
    )

    compare_csv = P1 / "results/pubmed_nc_sehgnn_lite_vs_ehgnn.csv"
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method", "macro_f1", "micro_f1", "total_time_sec", "seed", "note"
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "method": "EHGNN",
                "macro_f1": base_macro,
                "micro_f1": base_micro,
                "total_time_sec": base_time,
                "seed": 42,
                "note": base.get("source", ""),
            }
        )
        w.writerow(
            {
                "method": "SeHGNN-lite",
                "macro_f1": lite_macro,
                "micro_f1": lite_micro,
                "total_time_sec": lite_time,
                "seed": 42,
                "note": args.result_csv,
            }
        )
    print("Wrote {}".format(compare_csv))


if __name__ == "__main__":
    main()
