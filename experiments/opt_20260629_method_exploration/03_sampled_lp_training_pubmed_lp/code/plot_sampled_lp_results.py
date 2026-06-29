#!/usr/bin/env python3
"""Plot P3 Sampled-LP trade-off and Pareto curves vs EHGNN baseline."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_FILE = Path(__file__).resolve()
EXP_ROOT = _FILE.parents[2]
COMMON_DIR = EXP_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from plot_utils import require_matplotlib_or_skip  # noqa: E402

plt = require_matplotlib_or_skip("plot_sampled_lp_results.py")
P3 = EXP_ROOT / "03_sampled_lp_training_pubmed_lp"
FIGS = P3 / "figs"
BASELINE_JSON = P3 / "configs/ehgnn_pubmed_lp_baseline_seed42.json"


def load_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_baseline(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") == "missing":
        return None
    return data


def collect_sampled_points(results_dir: Path) -> list[dict]:
    points = []
    for csv_path in sorted(results_dir.glob("pubmed_lp_sampled_training_*.csv")):
        rows = load_csv_rows(csv_path)
        if not rows:
            continue
        r = rows[-1]
        try:
            points.append(
                {
                    "sample_ratio": float(r.get("sample_ratio") or 0),
                    "best_auc": float(r.get("best_auc") or 0),
                    "best_ap": float(r.get("best_ap") or 0),
                    "total_time_sec": float(
                        r.get("total_time_sec") or r.get("train_time_sec") or 0
                    ),
                    "source": str(csv_path.name),
                }
            )
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda x: x["sample_ratio"])
    return points


def line_plot(xs, ys, xlabel, ylabel, title, out_path, baseline_x=None, baseline_y=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if xs:
        ax.plot(xs, ys, marker="o", color="#4C72B0", label="Sampled-LP")
    if baseline_x is not None and baseline_y is not None:
        ax.axhline(baseline_y, color="#55A868", linestyle="--", linewidth=1, label="EHGNN baseline")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved {}".format(out_path))


def pareto_plot(points, baseline, out_path, metric="auc"):
    fig, ax = plt.subplots(figsize=(7, 4))
    if baseline:
        ax.scatter(
            [baseline["total_time_sec"]],
            [baseline["best_auc"] if metric == "auc" else baseline["best_ap"]],
            color="#55A868",
            s=80,
            label="EHGNN baseline",
            zorder=3,
        )
    if points:
        xs = [p["total_time_sec"] for p in points]
        ys = [p["best_auc"] if metric == "auc" else p["best_ap"] for p in points]
        ax.scatter(xs, ys, color="#4C72B0", s=60, label="Sampled-LP")
        for p in points:
            ax.annotate(
                "{:.2f}".format(p["sample_ratio"]),
                (p["total_time_sec"], p["best_auc"] if metric == "auc" else p["best_ap"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("Total Time (seconds)")
    ax.set_ylabel("Best Test AUC" if metric == "auc" else "Best Test AP")
    ax.set_title("Sampled-LP Pareto: {} vs Runtime".format(metric.upper()))
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved {}".format(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=str(P3 / "results"))
    ap.add_argument("--baseline_json", type=str, default=str(BASELINE_JSON))
    ap.add_argument("--out_dir", type=str, default=str(FIGS))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    baseline = load_baseline(Path(args.baseline_json))
    points = collect_sampled_points(results_dir)

    if not points and not baseline:
        print("No data to plot.")
        return

    ratios = [p["sample_ratio"] for p in points]
    aucs = [p["best_auc"] for p in points]
    aps = [p["best_ap"] for p in points]
    times = [p["total_time_sec"] for p in points]

    b_auc = float(baseline["best_auc"]) if baseline else None
    b_ap = float(baseline["best_ap"]) if baseline else None

    if points:
        line_plot(
            ratios,
            aucs,
            "Sample Ratio",
            "Best Test AUC",
            "Sampled-LP: AUC vs Sample Ratio",
            out_dir / "sampled_lp_auc_tradeoff.png",
            baseline_y=b_auc,
        )
        line_plot(
            ratios,
            aps,
            "Sample Ratio",
            "Best Test AP",
            "Sampled-LP: AP vs Sample Ratio",
            out_dir / "sampled_lp_ap_tradeoff.png",
            baseline_y=b_ap,
        )
        line_plot(
            ratios,
            times,
            "Sample Ratio",
            "Total Time (seconds)",
            "Sampled-LP: Runtime vs Sample Ratio",
            out_dir / "sampled_lp_runtime_tradeoff.png",
        )
        pareto_plot(points, baseline, out_dir / "sampled_lp_pareto_auc_runtime.png", "auc")
    else:
        print("No sampled result CSVs yet; skipping trade-off plots (baseline only not plotted alone).")

    summary = P3 / "results" / "sampled_lp_tradeoff_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "sample_ratio", "best_auc", "best_ap", "total_time_sec", "note"],
        )
        w.writeheader()
        if baseline:
            w.writerow(
                {
                    "method": "EHGNN",
                    "sample_ratio": 1.0,
                    "best_auc": baseline.get("best_auc"),
                    "best_ap": baseline.get("best_ap"),
                    "total_time_sec": baseline.get("total_time_sec"),
                    "note": baseline.get("source_log", ""),
                }
            )
        for p in points:
            w.writerow(
                {
                    "method": "Sampled-LP",
                    "sample_ratio": p["sample_ratio"],
                    "best_auc": p["best_auc"],
                    "best_ap": p["best_ap"],
                    "total_time_sec": p["total_time_sec"],
                    "note": p["source"],
                }
            )
    print("Wrote {}".format(summary))


if __name__ == "__main__":
    main()
