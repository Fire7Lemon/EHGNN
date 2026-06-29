#!/usr/bin/env bash
# P2 LP Pair Decoder PubMed LP — server runner
# Set PARSE_ONLY=1 to skip training and only run parse + plot.
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P2_DIR="experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp"
CODE="${P2_DIR}/code/main_pubmed_lp_pair_decoder.py"
LOG="${P2_DIR}/logs/pubmed_lp_pair_decoder_seed42_pair_mlp.log"
RESULT_CSV="${P2_DIR}/results/pubmed_lp_pair_decoder_seed42_pair_mlp.csv"
PARSED_CSV="${P2_DIR}/results/pubmed_lp_pair_decoder_seed42_pair_mlp_parsed.csv"

cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

mkdir -p "${P2_DIR}/logs" "${P2_DIR}/results" "${P2_DIR}/figs"

parse_and_plot() {
  echo "=== Parse log ==="
  if ! python -u "${P2_DIR}/code/parse_pair_decoder_results.py" \
    --log "${LOG}" \
    --out "${PARSED_CSV}" \
    --seed 42 \
    --decoder pair_mlp; then
    echo "[WARN] P2 parse failed; main result CSV may still exist at ${RESULT_CSV}"
  fi

  echo "=== Plot vs EHGNN baseline ==="
  if [ -f "${RESULT_CSV}" ]; then
    if ! python -u "${P2_DIR}/code/plot_pair_decoder_results.py" \
      --result_csv "${RESULT_CSV}" \
      --baseline_json "${P2_DIR}/configs/ehgnn_pubmed_lp_baseline_seed42.json" \
      --out_dir "${P2_DIR}/figs"; then
      echo "[WARN] P2 plot failed (matplotlib missing or no data)"
    fi
  else
    echo "[WARN] Skip plot — ${RESULT_CSV} not found"
  fi
}

if [ "${PARSE_ONLY:-0}" = "1" ]; then
  echo "=== P2 PARSE_ONLY mode ==="
  parse_and_plot
  echo "=== Done (parse-only). Outputs in ${P2_DIR} ==="
  exit 0
fi

if [ -f "${RESULT_CSV}" ]; then
  echo "=== P2 main CSV exists, skip training: ${RESULT_CSV} ==="
else
  echo "=== P2 LP Pair Decoder PubMed LP seed=42 pair_mlp ==="
  python -u "${CODE}" \
    --dataset PubMed \
    --seed 42 \
    --epochs 100 \
    --decoder pair_mlp \
    --log_interval 100 \
    --skip_batch_metrics \
    --root_out experiments/opt_20260629_method_exploration \
    2>&1 | tee "${LOG}"
fi

parse_and_plot

echo "=== Done. Outputs in ${P2_DIR} ==="
