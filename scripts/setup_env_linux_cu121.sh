#!/usr/bin/env bash
# =============================================================================
# EHGNN — create/use conda env "ehgnn" on Linux with PyTorch 2.3.1 + CUDA 12.1,
# DGL + torch-scatter wheels, then generic pip deps from requirements-server-cu121.txt
#
# Windows historical conda envs may use dgl==1.1.2; this SERVER-oriented script pins
# dgl==2.2.1 to match PyTorch 2.3 + cu121 wheels from DGL official wheel repo (recommended).
# Do not confuse with README/local snapshots that mention older DGL builds.
# =============================================================================
set -eo pipefail

echo "============================================================"
echo "[setup_env_linux_cu121] Step 0: detect OS"
echo "============================================================"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: This script is only for Linux servers. Current OS: $(uname -s)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo "[setup_env_linux_cu121] Step 1: locate conda"
echo "============================================================"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda or initialize conda for bash."
  exit 1
fi

echo "============================================================"
echo "[setup_env_linux_cu121] Step 2: create or reuse env ehgnn"
echo "============================================================"
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '!/^#/ && NF>=1 {print $1}' | grep -qx "ehgnn"; then
  echo "Conda env 'ehgnn' already exists — reusing."
else
  echo "Creating env ehgnn from ${REPO_ROOT}/environment.yml ..."
  conda env create -f "${REPO_ROOT}/environment.yml"
fi

conda activate ehgnn

echo "============================================================"
echo "[setup_env_linux_cu121] Step 3: upgrade pip tooling"
echo "============================================================"
python -m pip install --upgrade pip setuptools wheel

echo "============================================================"
echo "[setup_env_linux_cu121] Step 4: PyTorch 2.3.1 + CUDA 12.1 wheels"
echo "============================================================"
python -m pip install torch==2.3.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "============================================================"
echo "[setup_env_linux_cu121] Step 5: DGL for PyTorch 2.3 + cu121"
echo "============================================================"
# Server-side pairing (torch 2.3.x / cu121). Differs from some older Windows refs using dgl 1.1.2.
python -m pip install dgl==2.2.1 \
  -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

echo "============================================================"
echo "[setup_env_linux_cu121] Step 6: torch-scatter (PyG wheels)"
echo "============================================================"
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

echo "============================================================"
echo "[setup_env_linux_cu121] Step 7: requirements-server-cu121.txt"
echo "============================================================"
python -m pip install -r "${REPO_ROOT}/requirements-server-cu121.txt"

echo "============================================================"
echo "[setup_env_linux_cu121] Step 8: verify via scripts/check_env.py"
echo "============================================================"
python "${REPO_ROOT}/scripts/check_env.py"

echo "============================================================"
echo "[setup_env_linux_cu121] Done."
echo "============================================================"
