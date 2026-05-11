#!/usr/bin/env bash
# MAG240M server pipeline: conda env -> data check -> training.
# Usage:
#   export MAG240_DATA_ROOT="${HOME}/data"
#   export MAG240M_CONDA_ENV="${MAG240M_CONDA_ENV:-ehgnn}"
#   ./scripts/run_mag240m_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAG_DIR="${REPO_ROOT}/MAG240M"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

MAG240_DATA_ROOT="${MAG240_DATA_ROOT:-${HOME}/data}"
MAG240M_CONDA_ENV="${MAG240M_CONDA_ENV:-ehgnn}"

LOG_FILE="${LOG_DIR}/mag240m_run.log"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH" | tee -a "${LOG_FILE}"
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MAG240M_CONDA_ENV}"

{
  echo "========== $(date -Is) =========="
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "MAG240_DATA_ROOT=${MAG240_DATA_ROOT}"
  echo "CONDA_ENV=${MAG240M_CONDA_ENV}"
  echo ""

  echo "---- check_mag240m_data ----"
  python "${SCRIPT_DIR}/check_mag240m_data.py" --root "${MAG240_DATA_ROOT}"

  echo ""
  echo "---- MAG240M main.py ----"
  cd "${MAG_DIR}"
  python main.py --data_root "${MAG240_DATA_ROOT}" "$@"
} 2>&1 | tee -a "${LOG_FILE}"
