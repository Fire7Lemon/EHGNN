#!/usr/bin/env bash
# MAG240M server pipeline: preflight -> conda -> data check -> training.
#
# Usage:
#   export MAG240_DATA_ROOT="${HOME}/data"
#   export MAG240M_CONDA_ENV="${MAG240M_CONDA_ENV:-ehgnn}"
#   ./scripts/run_mag240m_server.sh
#
# Disk guard: requires >= 500 GB free on filesystem hosting MAG240_DATA_ROOT
# unless MAG240_SKIP_DISK_CHECK=1 (not recommended).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAG_DIR="${REPO_ROOT}/MAG240M"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/mag240m_run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========== $(date -Is) MAG240M run_mag240m_server.sh =========="

_PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "${_PY}" ]]; then
  echo "ERROR: need python3 or python in PATH for path canonicalization"
  exit 1
fi

export MAG240_DATA_ROOT="${MAG240_DATA_ROOT:-${HOME}/data}"
mkdir -p "${MAG240_DATA_ROOT}"
MAG240_DATA_ROOT="$("${_PY}" -c "import os; print(os.path.abspath(os.path.expanduser(os.environ['MAG240_DATA_ROOT'])))")"
export MAG240_DATA_ROOT

MAG240M_CONDA_ENV="${MAG240M_CONDA_ENV:-ehgnn}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MAG240_DATA_ROOT=${MAG240_DATA_ROOT}"
echo "LOG_FILE=${LOG_FILE}"

# --- Repo layout sanity (expects EHGNN repo root containing MAG240M/) ---
if [[ ! -d "${MAG_DIR}" ]]; then
  echo "ERROR: MAG240M directory not found at ${MAG_DIR}"
  echo "       scripts/ must live next to MAG240M/ inside the cloned repository."
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/check_mag240m_data.py" ]]; then
  echo "ERROR: Missing ${SCRIPT_DIR}/check_mag240m_data.py"
  exit 1
fi

# --- conda ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH"
  exit 1
fi

# --- Disk space (500 GB minimum recommended hard gate) ---
echo ""
echo "---- disk usage (df -h ${MAG240_DATA_ROOT}) ----"
df -h "${MAG240_DATA_ROOT}"

avail_kb="$(df -Pk "${MAG240_DATA_ROOT}" | awk 'NR==2 {print $4}')"
if [[ -z "${avail_kb}" ]] || ! [[ "${avail_kb}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse available KiB from df -Pk ${MAG240_DATA_ROOT}"
  exit 1
fi
need_kb=$((500 * 1024 * 1024))
echo "Available KiB on filesystem (POSIX df -Pk Avail column): ${avail_kb}"
echo "Required KiB for guardrail (500 GB):                  ${need_kb}"

if [[ "${avail_kb}" -lt "${need_kb}" ]]; then
  echo ""
  echo "******************************************************************************
WARNING: Free space on ${MAG240_DATA_ROOT} is below 500 GB.
MAG240M download + processing typically needs >= 500 GB (1 TB strongly recommended).
******************************************************************************"
  if [[ "${MAG240_SKIP_DISK_CHECK:-0}" != "1" ]]; then
    echo "Aborting. To bypass (NOT recommended): export MAG240_SKIP_DISK_CHECK=1"
    exit 1
  fi
  echo "MAG240_SKIP_DISK_CHECK=1 set — continuing despite low disk space."
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MAG240M_CONDA_ENV}"

echo ""
echo "---- check_mag240m_data ----"
python "${SCRIPT_DIR}/check_mag240m_data.py" --root "${MAG240_DATA_ROOT}"

echo ""
echo "---- MAG240M main.py ----"
cd "${MAG_DIR}"
python main.py --data_root "${MAG240_DATA_ROOT}" "$@"

echo "========== $(date -Is) finished =========="
