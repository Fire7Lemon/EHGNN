#!/usr/bin/env bash
# Dry-run / smoke test for method exploration code (no full training).
# Output: experiments/opt_20260629_method_exploration/smoke_test_report.log
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/mayq/ehgnn/EHGNN}"
EXP="experiments/opt_20260629_method_exploration"
REPORT="${EXP}/smoke_test_report.log"

cd "${PROJECT_ROOT}" 2>/dev/null || cd "$(dirname "$0")/../../.." || true
if [ ! -d "${EXP}" ]; then
  # local workspace fallback
  PROJECT_ROOT="$(pwd)"
fi

mkdir -p "${EXP}"
: > "${REPORT}"

log() { echo "$@" | tee -a "${REPORT}"; }
pass() { log "[PASS] $*"; }
fail() { log "[FAIL] $*"; }
skip() { log "[SKIP] $*"; }
warn() { log "[WARN] $*"; }

log "=== smoke_test_method_exploration ==="
log "timestamp: $(date -Iseconds 2>/dev/null || date)"
log "project: $(pwd)"
log ""

# --- server scripts exist ---
log "--- server scripts ---"
for s in \
  check_server_before_opt_runs.sh \
  run_p1_sehgnn_lite_pubmed_nc.sh \
  run_p2_pair_decoder_pubmed_lp.sh \
  run_p3_sampled_lp_pubmed.sh \
  run_p4_distill_pubmed_nc.sh \
  run_p5_ppr_topk_demo.sh \
  run_all_method_exploration.sh \
  smoke_test_method_exploration.sh; do
  if [ -f "${EXP}/server_scripts/${s}" ]; then
    pass "server_scripts/${s}"
  else
    fail "missing server_scripts/${s}"
  fi
done

# --- output dirs ---
log ""
log "--- output directories ---"
for d in \
  "${EXP}/common" \
  "${EXP}/01_sehgnn_lite_pubmed_nc/logs" \
  "${EXP}/02_lp_pair_decoder_pubmed_lp/results" \
  "${EXP}/03_sampled_lp_training_pubmed_lp/results" \
  "${EXP}/04_distill_mlp_pubmed_nc/figs" \
  "${EXP}/05_ppr_topk_lite_design/results"; do
  mkdir -p "${d}" 2>/dev/null || true
  if [ -d "${d}" ]; then
    pass "dir ${d}"
  else
    fail "dir ${d}"
  fi
done

# --- py_compile ---
log ""
log "--- py_compile ---"
PY_FILES=(
  "${EXP}/common/path_utils.py"
  "${EXP}/common/plot_utils.py"
  "${EXP}/01_sehgnn_lite_pubmed_nc/code/run_pubmed_nc_sehgnn_lite.py"
  "${EXP}/01_sehgnn_lite_pubmed_nc/code/parse_sehgnn_lite_results.py"
  "${EXP}/01_sehgnn_lite_pubmed_nc/code/plot_sehgnn_lite_results.py"
  "${EXP}/02_lp_pair_decoder_pubmed_lp/code/main_pubmed_lp_pair_decoder.py"
  "${EXP}/02_lp_pair_decoder_pubmed_lp/code/parse_pair_decoder_results.py"
  "${EXP}/02_lp_pair_decoder_pubmed_lp/code/plot_pair_decoder_results.py"
  "${EXP}/03_sampled_lp_training_pubmed_lp/code/main_pubmed_lp_sampled_training.py"
  "${EXP}/03_sampled_lp_training_pubmed_lp/code/parse_sampled_lp_results.py"
  "${EXP}/03_sampled_lp_training_pubmed_lp/code/plot_sampled_lp_results.py"
  "${EXP}/04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py"
  "${EXP}/04_distill_mlp_pubmed_nc/code/parse_distill_results.py"
  "${EXP}/04_distill_mlp_pubmed_nc/code/plot_distill_results.py"
  "${EXP}/05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py"
  "${EXP}/05_ppr_topk_lite_design/code/ppr_topk_lite.py"
  "${EXP}/05_ppr_topk_lite_design/code/parse_ppr_topk_results.py"
  "${EXP}/common/import_smoke_checks.py"
)
for f in "${PY_FILES[@]}"; do
  if [ ! -f "${f}" ]; then
    fail "missing ${f}"
    continue
  fi
  if python -m py_compile "${f}" 2>>"${REPORT}"; then
    pass "py_compile ${f}"
  else
    fail "py_compile ${f}"
  fi
done

# --- import-level checks (path_utils, plot_utils, parsers, --help) ---
log ""
log "--- import_smoke_checks (real import / --help) ---"
if python "${EXP}/common/import_smoke_checks.py" 2>>"${REPORT}" | tee -a "${REPORT}"; then
  pass "import_smoke_checks.py"
else
  fail "import_smoke_checks.py — see output above"
fi

log ""
log "=== smoke test finished; see ${REPORT} ==="
