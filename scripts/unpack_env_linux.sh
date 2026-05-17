#!/usr/bin/env bash
# =============================================================================
# Unpack a Linux-produced conda-pack archive and run conda-unpack.
# NOT for Windows-built conda environments or archives.
#
# Usage (from repo root):
#   bash scripts/unpack_env_linux.sh <env_tar_gz> <target_dir>
# Example:
#   bash scripts/unpack_env_linux.sh ehgnn-linux-py310-cu121-torch231.tar.gz /home/ehgnn-env
# =============================================================================
set -eo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: This unpack helper is intended for Linux hosts only. Current OS: $(uname -s)"
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/unpack_env_linux.sh <env_tar_gz> <target_dir>"
  exit 1
fi

ARCHIVE="$1"
TARGET_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "ERROR: Archive not found: ${ARCHIVE}"
  exit 1
fi

echo "============================================================"
echo "[unpack_env_linux] Extract ${ARCHIVE}"
echo "             -> ${TARGET_DIR}"
echo "============================================================"
mkdir -p "${TARGET_DIR}"
tar -xzf "${ARCHIVE}" -C "${TARGET_DIR}"

if [[ ! -x "${TARGET_DIR}/bin/conda-unpack" ]]; then
  echo "ERROR: ${TARGET_DIR}/bin/conda-unpack not found or not executable after extract."
  exit 1
fi

echo "============================================================"
echo "[unpack_env_linux] Running conda-unpack"
echo "============================================================"
"${TARGET_DIR}/bin/conda-unpack"

echo "============================================================"
echo "[unpack_env_linux] Running environment check (repo scripts/check_env.py)"
echo "============================================================"
"${TARGET_DIR}/bin/python" "${REPO_ROOT}/scripts/check_env.py"

echo "============================================================"
echo "[unpack_env_linux] Done."
echo "Activate with: source ${TARGET_DIR}/bin/activate"
echo "   or run:       ${TARGET_DIR}/bin/python <your_script.py>"
echo "============================================================"
