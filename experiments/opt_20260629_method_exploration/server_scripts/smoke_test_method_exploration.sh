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
  "${EXP}/05_ppr_topk_lite_design/code/plot_ppr_topk_results.py"
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

# --- path_utils import & safe_relpath ---
log ""
log "--- path_utils ---"
python - <<'PY' 2>>"${REPORT}" | tee -a "${REPORT}"
import sys
from pathlib import Path
root = Path(".").resolve()
common = root / "experiments/opt_20260629_method_exploration/common"
sys.path.insert(0, str(common))
from path_utils import (
    add_source_dir,
    bootstrap_paths,
    get_project_root,
    resolve_project_data_path,
    safe_relpath,
)
pr = get_project_root(common / "path_utils.py")
dp = resolve_project_data_path(pr)
assert dp.endswith("/"), dp
assert "data/" in dp
rel = safe_relpath("experiments/opt_20260629_method_exploration/foo.log", pr)
assert not rel.startswith("/") or "experiments" in rel
print("[PASS] path_utils import + resolve_project_data_path + safe_relpath")
print("  data_path sample:", dp)
print("  safe_relpath sample:", rel)
PY

# --- parser dry-run on relative log paths ---
log ""
log "--- parser relative log path ---"
python - <<'PY' 2>>"${REPORT}" | tee -a "${REPORT}"
import sys
from pathlib import Path
root = Path(".").resolve()
common = root / "experiments/opt_20260629_method_exploration/common"
sys.path.insert(0, str(common))
p2_parse = root / "experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/parse_pair_decoder_results.py"
sys.path.insert(0, str(p2_parse.parent))
# import parse module functions via exec minimal test
from path_utils import safe_relpath, resolve_under_root, get_project_root
ROOT = get_project_root(p2_parse)
log_rel = "experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/logs/test.log"
resolved = resolve_under_root(log_rel, ROOT)
out = safe_relpath(resolved, ROOT)
print("[PASS] parser path resolve:", out)
PY

# --- matplotlib ---
log ""
log "--- matplotlib ---"
if python -c "import matplotlib; print('[PASS] matplotlib', matplotlib.__version__)" 2>>"${REPORT}" | tee -a "${REPORT}"; then
  :
else
  warn "matplotlib not installed — plot scripts should exit 0 via plot_utils"
  python - <<'PY' 2>>"${REPORT}" | tee -a "${REPORT}"
import sys
from pathlib import Path
common = Path("experiments/opt_20260629_method_exploration/common")
sys.path.insert(0, str(common.resolve()))
from plot_utils import matplotlib_available
print("[INFO] matplotlib_available:", matplotlib_available())
PY
fi

# --- plot_utils skip (only if matplotlib missing) ---
if ! python -c "import matplotlib" 2>/dev/null; then
  log ""
  log "--- plot skip behavior (no matplotlib) ---"
  if python "${EXP}/05_ppr_topk_lite_design/code/plot_ppr_topk_results.py" 2>>"${REPORT}" | tee -a "${REPORT}"; then
    pass "plot_ppr_topk_results exits 0 without matplotlib"
  else
    fail "plot_ppr_topk_results should exit 0 without matplotlib"
  fi
else
  skip "plot skip test — matplotlib present locally"
fi

# --- no writes to forbidden paths in server scripts ---
log ""
log "--- forbidden output paths in server_scripts ---"
if grep -R "server_results/" "${EXP}/server_scripts/"*.sh 2>/dev/null | grep -v "Does NOT write" | grep -v "#" | tee -a "${REPORT}"; then
  warn "found server_results references in shell (review above)"
else
  pass "no active server_results writes in server_scripts"
fi

log ""
log "=== smoke test finished; see ${REPORT} ==="
