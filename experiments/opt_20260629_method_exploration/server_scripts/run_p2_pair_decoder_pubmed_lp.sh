#!/usr/bin/env bash
# P2 LP Pair Decoder PubMed LP — server runner
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P2_DIR="experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp"
CODE="${P2_DIR}/code/main_pubmed_lp_pair_decoder.py"
LOG="${P2_DIR}/logs/pubmed_lp_pair_decoder_seed42_pair_mlp.log"
RESULT_CSV="${P2_DIR}/results/pubmed_lp_pair_decoder_seed42_pair_mlp.csv"

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

echo "=== Parse log ==="
python -u "${P2_DIR}/code/parse_pair_decoder_results.py" \
  --log "${LOG}" \
  --out "${P2_DIR}/results/pubmed_lp_pair_decoder_seed42_pair_mlp_parsed.csv" \
  --seed 42 \
  --decoder pair_mlp

echo "=== Plot vs EHGNN baseline ==="
python -u "${P2_DIR}/code/plot_pair_decoder_results.py" \
  --result_csv "${RESULT_CSV}" \
  --baseline_json "${P2_DIR}/configs/ehgnn_pubmed_lp_baseline_seed42.json" \
  --out_dir "${P2_DIR}/figs"

echo "=== Done. Outputs in ${P2_DIR} ==="

# Quick validation alternative (uncomment to use):
# python -u "${CODE}" --epochs 30 ... 2>&1 | tee "${P2_DIR}/logs/pubmed_lp_pair_decoder_seed42_pair_mlp_e30.log"
