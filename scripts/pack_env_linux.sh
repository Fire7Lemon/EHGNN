#!/usr/bin/env bash
# =============================================================================
# Pack conda env "ehgnn" on Linux using conda-pack (for Release / object storage).
# Do NOT commit the resulting .tar.gz to Git.
# After unpack on a Linux machine, run conda-unpack inside the extracted prefix.
# =============================================================================
set -eo pipefail

echo "============================================================"
echo "[pack_env_linux] Step 1: OS check"
echo "============================================================"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: conda-pack must be produced on Linux. Current OS: $(uname -s)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_NAME="ehgnn-linux-py310-cu121-torch231.tar.gz"
OUT_PATH="${REPO_ROOT}/${OUT_NAME}"

echo "============================================================"
echo "[pack_env_linux] Step 2: conda availability"
echo "============================================================"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH."
  exit 1
fi

echo "============================================================"
echo "[pack_env_linux] Step 3: ehgnn env exists"
echo "============================================================"
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '!/^#/ && NF>=1 {print $1}' | grep -qx "ehgnn"; then
  echo "ERROR: conda env 'ehgnn' not found. Create it first (e.g. setup_env_linux_cu121.sh)."
  exit 1
fi

echo "============================================================"
echo "[pack_env_linux] Step 4: install conda-pack into base (if missing)"
echo "============================================================"
conda install -n base -y conda-pack

echo "============================================================"
echo "[pack_env_linux] Step 5: pack env -> ${OUT_PATH}"
echo "============================================================"
conda pack -n ehgnn -o "${OUT_PATH}"

echo "============================================================"
echo "[pack_env_linux] DONE"
echo "============================================================"
echo "Artifact: ${OUT_PATH}"
echo ""
echo "IMPORTANT:"
echo "  - Do NOT git add or push this tar.gz into the repository."
echo "  - Upload to GitHub Release, object storage, or internal artifact storage."
echo "  - On the target Linux server, extract and run bin/conda-unpack inside the unpacked tree."
echo "    Example unpack helper: bash scripts/unpack_env_linux.sh ${OUT_NAME} /path/to/target_dir"
echo "============================================================"
