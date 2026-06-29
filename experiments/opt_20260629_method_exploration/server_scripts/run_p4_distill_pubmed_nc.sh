#!/usr/bin/env bash
# P4 EHGNN-to-MLP Distillation PubMed NC
# PARSE_ONLY=1 → skip training, parse + plot only
# TEACHER_MODE=load + existing logits → skip teacher retrain (pass --teacher_mode load)
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
P4_DIR="experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc"
CODE="${P4_DIR}/code/main_pubmed_nc_distill.py"
LOG="${P4_DIR}/logs/pubmed_nc_distill_seed42_raw.log"
STUDENT_CSV="${P4_DIR}/results/pubmed_nc_distill_student_seed42.csv"
TEACHER_LOGITS="${P4_DIR}/results/pubmed_nc_teacher_seed42_logits.pt"

cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

mkdir -p "${P4_DIR}/logs" "${P4_DIR}/results" "${P4_DIR}/figs"

TEACHER_MODE="${TEACHER_MODE:-train}"
if [ -f "${TEACHER_LOGITS}" ] && [ "${TEACHER_MODE}" = "train" ] && [ -f "${STUDENT_CSV}" ]; then
  echo "=== P4 teacher logits + student CSV exist; use TEACHER_MODE=load or delete artifacts to retrain ==="
  TEACHER_MODE="load"
fi

parse_and_plot() {
  if ! python -u "${P4_DIR}/code/parse_distill_results.py" \
    --log "${LOG}" \
    --csv "${STUDENT_CSV}"; then
    echo "[WARN] P4 parse failed"
  fi
  if ! python -u "${P4_DIR}/code/plot_distill_results.py" \
    --csv "${STUDENT_CSV}" \
    --out_dir "${P4_DIR}/figs"; then
    echo "[WARN] P4 plot failed (matplotlib missing or no data)"
  fi
}

if [ "${PARSE_ONLY:-0}" = "1" ]; then
  echo "=== P4 PARSE_ONLY mode ==="
  parse_and_plot
  echo "=== Done (parse-only). Outputs in ${P4_DIR} ==="
  exit 0
fi

if [ -f "${STUDENT_CSV}" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
  echo "=== P4 student CSV exists, skip training: ${STUDENT_CSV} ==="
else
  echo "=== P4 EHGNN-to-MLP Distillation seed=42 raw (teacher_mode=${TEACHER_MODE}) ==="
  python -u "${CODE}" \
    --dataset PubMed \
    --seed 42 \
    --teacher_mode "${TEACHER_MODE}" \
    --teacher_epochs 100 \
    --student_epochs 200 \
    --student_input raw \
    --temperature 3.0 \
    --kd_alpha 0.5 \
    --root_out experiments/opt_20260629_method_exploration \
    2>&1 | tee "${LOG}"
fi

parse_and_plot
echo "=== Done. Outputs in ${P4_DIR} ==="
