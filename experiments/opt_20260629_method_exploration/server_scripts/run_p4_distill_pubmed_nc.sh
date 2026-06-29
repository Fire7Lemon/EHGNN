#!/usr/bin/env bash
# P4 EHGNN-to-MLP Distillation PubMed NC
# Quick try: teacher_epochs=30 student_epochs=50
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P4_DIR="experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc"
CODE="${P4_DIR}/code/main_pubmed_nc_distill.py"
LOG="${P4_DIR}/logs/pubmed_nc_distill_seed42_raw.log"

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

mkdir -p "${P4_DIR}/logs" "${P4_DIR}/results" "${P4_DIR}/figs"

echo "=== P4 EHGNN-to-MLP Distillation seed=42 raw ==="

python -u "${CODE}" \
  --dataset PubMed \
  --seed 42 \
  --teacher_mode train \
  --teacher_epochs 100 \
  --student_epochs 200 \
  --student_input raw \
  --temperature 3.0 \
  --kd_alpha 0.5 \
  --root_out experiments/opt_20260629_method_exploration \
  2>&1 | tee "${LOG}"

python -u "${P4_DIR}/code/parse_distill_results.py" \
  --log "${LOG}" \
  --csv "${P4_DIR}/results/pubmed_nc_distill_student_seed42.csv"

python -u "${P4_DIR}/code/plot_distill_results.py" \
  --csv "${P4_DIR}/results/pubmed_nc_distill_student_seed42.csv" \
  --out_dir "${P4_DIR}/figs"

echo "=== Done. Outputs in ${P4_DIR} ==="
