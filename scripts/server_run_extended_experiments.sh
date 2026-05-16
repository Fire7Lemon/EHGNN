#!/usr/bin/env bash
# Extended experiments: ablations + sensitivities (NC/LP).
# Run after core reproduction when resources allow.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/server_extended_experiments.log"

exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "[EXT] server_extended_experiments.sh"
echo "[EXT] ROOT=${ROOT}"
echo "[EXT] log=${LOG}"
echo "============================================================"

NC="${ROOT}/Node Classification"
LP="${ROOT}/Link Prediction"

SEEDS="42 3407 2026 6666 8888"
SKIP="--skip_existing"

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

run_step "cd \"${NC}\" && python run_nc_ablation_all_datasets.py --seeds ${SEEDS} ${SKIP}"
run_step "cd \"${LP}\" && python run_lp_ablation_all_datasets.py --seeds ${SEEDS} ${SKIP}"

run_step "cd \"${NC}\" && python run_nc_sensitivity_k.py --seeds ${SEEDS} ${SKIP}"
run_step "cd \"${LP}\" && python run_lp_sensitivity_k.py --seeds ${SEEDS} ${SKIP}"

run_step "cd \"${NC}\" && python run_nc_sensitivity_alpha.py --seeds ${SEEDS} ${SKIP}"
run_step "cd \"${LP}\" && python run_lp_sensitivity_alpha.py --seeds ${SEEDS} ${SKIP}"

run_step "cd \"${NC}\" && python run_nc_sensitivity_walk_num.py --seeds ${SEEDS} ${SKIP}"
run_step "cd \"${LP}\" && python run_lp_sensitivity_walk_num.py --seeds ${SEEDS} ${SKIP}"

echo "[EXT] All extended steps finished."
