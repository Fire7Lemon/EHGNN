#!/usr/bin/env bash
# Master orchestrator: P5 → P1 → P2 → P3 → P4
# All outputs stay under experiments/opt_20260629_method_exploration/
# Does NOT write to server_results/ or Node Classification|Link Prediction/results/
set -uo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
EXP_ROOT="experiments/opt_20260629_method_exploration"
SCRIPT_DIR="${EXP_ROOT}/server_scripts"
MASTER_LOG="${EXP_ROOT}/logs_run_all_method_exploration.log"
STATUS_CSV="${EXP_ROOT}/server_run_status.csv"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

# --- helpers ---
log_both() {
  echo "$@" | tee -a "${MASTER_LOG}"
}

csv_escape() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

init_status_csv() {
  if [ ! -f "${STATUS_CSV}" ]; then
    echo "stage,script,status,return_code,start_time,end_time,duration_sec,note" > "${STATUS_CSV}"
  fi
}

append_status_row() {
  local stage="$1"
  local script="$2"
  local status="$3"
  local rc="$4"
  local start_time="$5"
  local end_time="$6"
  local duration="$7"
  local note="${8:-}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(csv_escape "${stage}")" \
    "$(csv_escape "${script}")" \
    "$(csv_escape "${status}")" \
    "${rc}" \
    "$(csv_escape "${start_time}")" \
    "$(csv_escape "${end_time}")" \
    "${duration}" \
    "$(csv_escape "${note}")" >> "${STATUS_CSV}"
}

run_stage() {
  local stage="$1"
  local script_name="$2"
  local note="${3:-}"
  local script_path="${SCRIPT_DIR}/${script_name}"
  local start_ts end_ts duration rc status

  start_ts="$(date -Iseconds 2>/dev/null || date)"
  t0=$(date +%s)

  log_both "[START] ${stage} script=${script_path} start=${start_ts}"

  if [ ! -f "${script_path}" ]; then
    rc=127
    status="missing_script"
    log_both "[DONE] ${stage} return_code=${rc} (script not found)"
  else
    bash "${script_path}" >> "${MASTER_LOG}" 2>&1
    rc=$?
    if [ "${rc}" -eq 0 ]; then
      status="success"
    else
      status="failed"
    fi
    log_both "[DONE] ${stage} return_code=${rc}"
  fi

  end_ts="$(date -Iseconds 2>/dev/null || date)"
  t1=$(date +%s)
  duration=$((t1 - t0))
  append_status_row "${stage}" "${script_name}" "${status}" "${rc}" "${start_ts}" "${end_ts}" "${duration}" "${note}"
}

# --- main ---
init_status_csv

{
  echo "=== run_all_method_exploration ==="
  echo "timestamp: $(date -Iseconds 2>/dev/null || date)"
  echo "project: ${PROJECT_ROOT}"
  echo "master_log: ${MASTER_LOG}"
  echo "status_csv: ${STATUS_CSV}"
  echo "order: P5 → P1 → P2 → P3 → P4"
  echo ""
} | tee "${MASTER_LOG}"

run_stage "P5" "run_p5_ppr_topk_demo.sh" "PPR-TopK demo; no training"
run_stage "P1" "run_p1_sehgnn_lite_pubmed_nc.sh" "SeHGNN-lite PubMed NC"
run_stage "P2" "run_p2_pair_decoder_pubmed_lp.sh" "LP Pair Decoder PubMed LP"
run_stage "P3" "run_p3_sampled_lp_pubmed.sh" "Sampled-LP Training PubMed LP"
run_stage "P4" "run_p4_distill_pubmed_nc.sh" "EHGNN-to-MLP Distillation PubMed NC"

log_both ""
log_both "=== All stages finished. See ${STATUS_CSV} for summary. ==="
