#!/usr/bin/env bash
# P1 SeHGNN-lite PubMed NC — server runner
# PARSE_ONLY=1  → skip training, parse + plot only
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P1_DIR="experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc"
CODE="${P1_DIR}/code/run_pubmed_nc_sehgnn_lite.py"
LOG="${P1_DIR}/logs/pubmed_nc_sehgnn_lite_seed42_concat.log"
RESULT_CSV="${P1_DIR}/results/pubmed_nc_sehgnn_lite_seed42_concat.csv"
PARSED_CSV="${P1_DIR}/results/pubmed_nc_sehgnn_lite_seed42_concat_parsed.csv"

cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

mkdir -p "${P1_DIR}/logs" "${P1_DIR}/results" "${P1_DIR}/figs"

parse_and_plot() {
  echo "=== Parse log ==="
  if ! python -u "${P1_DIR}/code/parse_sehgnn_lite_results.py" \
    --log "${LOG}" \
    --out "${PARSED_CSV}" \
    --seed 42 \
    --fusion concat; then
    echo "[WARN] P1 parse failed; main CSV may still exist at ${RESULT_CSV}"
  fi

  echo "=== Plot vs EHGNN baseline ==="
  if ! python -u "${P1_DIR}/code/plot_sehgnn_lite_results.py" \
    --result_csv "${RESULT_CSV}" \
    --baseline_json "${P1_DIR}/configs/ehgnn_pubmed_baseline_seed42.json" \
    --out_dir "${P1_DIR}/figs"; then
    echo "[WARN] P1 plot failed (matplotlib missing or no data)"
  fi
}

if [ "${PARSE_ONLY:-0}" = "1" ]; then
  echo "=== P1 PARSE_ONLY mode ==="
  parse_and_plot
  echo "=== Done (parse-only). Outputs in ${P1_DIR} ==="
  exit 0
fi

if [ -f "${RESULT_CSV}" ]; then
  echo "=== P1 main CSV exists, skip training: ${RESULT_CSV} ==="
else
  echo "=== P1 SeHGNN-lite PubMed NC seed=42 concat ==="
  python -u "${CODE}" \
    --dataset PubMed \
    --seed 42 \
    --epochs 100 \
    --hidden 256 \
    --dropout 0.4 \
    --lr 0.001 \
    --fusion concat \
    --root_out experiments/opt_20260629_method_exploration \
    2>&1 | tee "${LOG}"
fi

parse_and_plot
echo "=== Done. Outputs in ${P1_DIR} ==="
