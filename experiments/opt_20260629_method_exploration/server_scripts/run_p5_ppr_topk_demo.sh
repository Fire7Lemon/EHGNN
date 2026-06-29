#!/usr/bin/env bash
# P5 PPR-TopK-lite Design & Demo — NO model training
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P5_DIR="experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design"
CODE="${P5_DIR}/code/demo_pubmed_ppr_topk.py"
LOG="${P5_DIR}/logs/pubmed_ppr_topk_demo_seed42.log"

cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

export PYTHONPATH="${PROJECT_ROOT}/Node Classification:${PYTHONPATH:-}"

mkdir -p "${P5_DIR}/logs" "${P5_DIR}/results" "${P5_DIR}/figs"

echo "=== P5 PPR-TopK-lite PubMed Demo (design only, no training) ==="

python -u "${CODE}" \
  --dataset PubMed \
  --seed 42 \
  --num_target_nodes 500 \
  --k 20 \
  --alpha 0.15 \
  --num_iters 10 \
  --metapath_index 0 \
  --root_out experiments/opt_20260629_method_exploration \
  2>&1 | tee "${LOG}"

python -u "${P5_DIR}/code/compare_rw_vs_ppr_neighbors.py"
python -u "${P5_DIR}/code/parse_ppr_topk_results.py" --log "${LOG}"
python -u "${P5_DIR}/code/plot_ppr_topk_results.py"

echo "=== Done. Outputs in ${P5_DIR} ==="
