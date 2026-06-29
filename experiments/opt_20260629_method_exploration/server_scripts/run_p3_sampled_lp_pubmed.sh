#!/usr/bin/env bash
# P3 Sampled-LP Training PubMed LP — server runner
# Quick try: change --epochs 100 to --epochs 30 in both runs below.
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P3_DIR="experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp"
CODE="${P3_DIR}/code/main_pubmed_lp_sampled_training.py"

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

echo "=== P3 Sampled-LP ratio=0.50 ==="
python -u "${CODE}" \
  "${COMMON_ARGS[@]}" \
  --sample_ratio 0.50 \
  2>&1 | tee "${P3_DIR}/logs/pubmed_lp_sampled_seed42_ratio050.log"

python -u "${P3_DIR}/code/parse_sampled_lp_results.py" \
  --log "${P3_DIR}/logs/pubmed_lp_sampled_seed42_ratio050.log" \
  --out "${P3_DIR}/results/pubmed_lp_sampled_seed42_ratio050_parsed.csv"

echo "=== P3 Sampled-LP ratio=0.25 ==="
python -u "${CODE}" \
  "${COMMON_ARGS[@]}" \
  --sample_ratio 0.25 \
  2>&1 | tee "${P3_DIR}/logs/pubmed_lp_sampled_seed42_ratio025.log"

python -u "${P3_DIR}/code/parse_sampled_lp_results.py" \
  --log "${P3_DIR}/logs/pubmed_lp_sampled_seed42_ratio025.log" \
  --out "${P3_DIR}/results/pubmed_lp_sampled_seed42_ratio025_parsed.csv"

echo "=== Plot trade-off ==="
python -u "${P3_DIR}/code/plot_sampled_lp_results.py" \
  --results_dir "${P3_DIR}/results" \
  --baseline_json "${P3_DIR}/configs/ehgnn_pubmed_lp_baseline_seed42.json" \
  --out_dir "${P3_DIR}/figs"

echo "=== Done. Outputs in ${P3_DIR} ==="
