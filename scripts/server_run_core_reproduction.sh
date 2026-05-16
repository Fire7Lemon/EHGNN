#!/usr/bin/env bash
# Core paper reproduction: NC 5-seed, LP 5-seed, MAG240M checks + launcher.
# Prerequisites: conda env activated, data in place, cwd irrelevant.
# Does not install deps or switch git branch.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/server_core_reproduction.log"

exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "[CORE] EHGNN server_core_reproduction.sh"
echo "[CORE] ROOT=${ROOT}"
echo "[CORE] log=${LOG}"
echo "============================================================"

NC="${ROOT}/Node Classification"
LP="${ROOT}/Link Prediction"

run_step() {
  echo ""
  echo "[RUN] $*"
  set +e
  eval "$@"
  code=$?
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "[FAIL] exit_code=${code} cmd: $*"
    exit "${code}"
  fi
  echo "[OK] $*"
}

run_step "cd \"${NC}\" && python run_pubmed_5seeds.py"
run_step "cd \"${NC}\" && python run_dblp_5seeds.py --seeds 42 3407 2026 6666 8888 --skip_existing"
run_step "cd \"${NC}\" && python run_yelp_5seeds.py --seeds 42 3407 2026 6666 8888 --skip_existing"

run_step "cd \"${LP}\" && python run_pubmed_lp_5seeds.py --seeds 42 3407 2026 6666 8888 --skip_existing"
run_step "cd \"${LP}\" && python run_dblp_lp_5seeds.py --seeds 42 3407 2026 6666 8888 --skip_existing"
run_step "cd \"${LP}\" && python run_yelp_lp_5seeds.py --seeds 42 3407 2026 6666 8888 --skip_existing"

MAG_ROOT="${MAG240_DATA_ROOT:-${HOME}/data}"
run_step "cd \"${ROOT}\" && python scripts/check_mag240m_data.py --root \"${MAG_ROOT}\""
chmod +x "${ROOT}/scripts/run_mag240m_server.sh"
run_step "cd \"${ROOT}\" && ./scripts/run_mag240m_server.sh"

echo "[CORE] All steps finished."
