#!/usr/bin/env python3
"""Plot P2 Pair Decoder vs EHGNN PubMed LP baseline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMMON = ROOT / "experiments/opt_20260629_method_exploration/common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from plot_utils import require_matplotlib_or_skip  # noqa: E402

plt = require_matplotlib_or_skip("plot_pair_decoder_results.py")

ROOT = Path(__file__).resolve().parents[4]
P2 = ROOT / "experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp"
FIGS = P2 / "figs"
BASELINE_JSON = P2 / "configs/ehgnn_pubmed_lp_baseline_seed42.json"


def load_result_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def load_baseline(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "missing", "note": "baseline json not found"}


def bar_plot(labels, values, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#55A868"][: len(labels)]
    ax.bar(labels, values, color=colors)
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
        default=str(P2 / "results/pubmed_lp_pair_decoder_seed42_pair_mlp.csv"),
    )
    ap.add_argument("--baseline_json", type=str, default=str(BASELINE_JSON))
    ap.add_argument("--out_dir", type=str, default=str(FIGS))
    args = ap.parse_args()

    result = load_result_row(Path(args.result_csv))
    baseline = load_baseline(Path(args.baseline_json))
    out_dir = Path(args.out_dir)

    has_baseline = baseline.get("status") != "missing"
    labels = []
    auc_vals = []
    ap_vals = []
    rt_vals = []

    if has_baseline:
        labels.append("EHGNN")
        auc_vals.append(float(baseline["best_auc"]))
        ap_vals.append(float(baseline["best_ap"]))
        rt_vals.append(float(baseline["total_time_sec"]))

    if result is not None:
        labels.append("Pair Decoder")
        auc_vals.append(float(result.get("best_auc") or 0))
        ap_vals.append(float(result.get("best_ap") or 0))
        rt_vals.append(float(result.get("total_time_sec") or result.get("train_time_sec") or 0))
    elif not has_baseline:
        print("No result CSV and no baseline — skip plotting.")
        return

    if len(labels) >= 1:
        bar_plot(
            labels,
            auc_vals,
            "Best Test AUC",
            "PubMed LP seed=42: AUC",
            out_dir / "pair_decoder_auc_compare.png",
        )
        bar_plot(
            labels,
            ap_vals,
            "Best Test AP",
            "PubMed LP seed=42: AP",
            out_dir / "pair_decoder_ap_compare.png",
        )
        bar_plot(
            labels,
            rt_vals,
            "Total Time (seconds)",
            "PubMed LP seed=42: Runtime",
            out_dir / "pair_decoder_runtime_compare.png",
        )
    else:
        print("Insufficient data for plots.")

    compare_csv = P2 / "results/pubmed_lp_pair_decoder_vs_ehgnn.csv"
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "best_auc", "best_ap", "final_auc", "final_ap", "total_time_sec", "note"],
        )
        w.writeheader()
        if has_baseline:
            w.writerow(
                {
                    "method": "EHGNN",
                    "best_auc": baseline.get("best_auc"),
                    "best_ap": baseline.get("best_ap"),
                    "final_auc": baseline.get("final_auc"),
                    "final_ap": baseline.get("final_ap"),
                    "total_time_sec": baseline.get("total_time_sec"),
                    "note": baseline.get("source_log", ""),
                }
            )
        if result is not None:
            w.writerow(
                {
                    "method": "LP Pair Decoder",
                    "best_auc": result.get("best_auc"),
                    "best_ap": result.get("best_ap"),
                    "final_auc": result.get("final_auc"),
                    "final_ap": result.get("final_ap"),
                    "total_time_sec": result.get("total_time_sec"),
                    "note": args.result_csv,
                }
            )
    print("Wrote {}".format(compare_csv))


if __name__ == "__main__":
    main()
