#!/usr/bin/env bash
# P3 Sampled-LP Training PubMed LP — server runner
# Set PARSE_ONLY=1 to skip training and only parse + plot existing logs/CSVs.
# Quick try: change --epochs 100 to --epochs 30 in training blocks below.
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P3_DIR="experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp"
CODE="${P3_DIR}/code/main_pubmed_lp_sampled_training.py"

RATIO050_CSV="${P3_DIR}/results/pubmed_lp_sampled_training_seed42_ratio050.csv"
RATIO025_CSV="${P3_DIR}/results/pubmed_lp_sampled_training_seed42_ratio025.csv"
RATIO050_LOG="${P3_DIR}/logs/pubmed_lp_sampled_seed42_ratio050.log"
RATIO025_LOG="${P3_DIR}/logs/pubmed_lp_sampled_seed42_ratio025.log"
RATIO050_PARSED="${P3_DIR}/results/pubmed_lp_sampled_seed42_ratio050_parsed.csv"
RATIO025_PARSED="${P3_DIR}/results/pubmed_lp_sampled_seed42_ratio025_parsed.csv"

cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

mkdir -p "${P3_DIR}/logs" "${P3_DIR}/results" "${P3_DIR}/figs"

COMMON_ARGS=(
  --dataset PubMed
  --seed 42
  --epochs 100
  --resample_each_epoch
  --log_interval 100
  --skip_batch_metrics
  --root_out experiments/opt_20260629_method_exploration
)

parse_ratio050() {
  echo "=== Parse P3 ratio=0.50 ==="
  if ! python -u "${P3_DIR}/code/parse_sampled_lp_results.py" \
    --log "${RATIO050_LOG}" \
    --out "${RATIO050_PARSED}"; then
    echo "[WARN] P3 ratio=0.50 parse failed; main CSV may exist at ${RATIO050_CSV}"
  fi
}

parse_ratio025() {
  echo "=== Parse P3 ratio=0.25 ==="
  if ! python -u "${P3_DIR}/code/parse_sampled_lp_results.py" \
    --log "${RATIO025_LOG}" \
    --out "${RATIO025_PARSED}"; then
    echo "[WARN] P3 ratio=0.25 parse failed; main CSV may exist at ${RATIO025_CSV}"
  fi
}

plot_all() {
  echo "=== Plot trade-off ==="
  if ! python -u "${P3_DIR}/code/plot_sampled_lp_results.py" \
    --results_dir "${P3_DIR}/results" \
    --baseline_json "${P3_DIR}/configs/ehgnn_pubmed_lp_baseline_seed42.json" \
    --out_dir "${P3_DIR}/figs"; then
    echo "[WARN] P3 plot failed (matplotlib missing or no data)"
  fi
}

if [ "${PARSE_ONLY:-0}" = "1" ]; then
  echo "=== P3 PARSE_ONLY mode ==="
  parse_ratio050
  parse_ratio025
  plot_all
  echo "=== Done (parse-only). Outputs in ${P3_DIR} ==="
  exit 0
fi

echo "=== P3 Sampled-LP ratio=0.50 ==="
if [ -f "${RATIO050_CSV}" ]; then
  echo "Main CSV exists, skip training: ${RATIO050_CSV}"
else
  python -u "${CODE}" \
    "${COMMON_ARGS[@]}" \
    --sample_ratio 0.50 \
    2>&1 | tee "${RATIO050_LOG}"
fi
parse_ratio050

echo "=== P3 Sampled-LP ratio=0.25 ==="
if [ -f "${RATIO025_CSV}" ]; then
  echo "Main CSV exists, skip training: ${RATIO025_CSV}"
else
  python -u "${CODE}" \
    "${COMMON_ARGS[@]}" \
    --sample_ratio 0.25 \
    2>&1 | tee "${RATIO025_LOG}"
fi
parse_ratio025

plot_all

echo "=== Done. Outputs in ${P3_DIR} ==="
