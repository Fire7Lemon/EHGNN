#!/usr/bin/env bash
# Pre-flight environment check for opt_20260629_method_exploration server runs.
# Does NOT train models; only prints environment info.
set -euo pipefail

PROJECT_ROOT="/home/mayq/ehgnn/EHGNN"
EXP_ROOT="experiments/opt_20260629_method_exploration"
OUT_LOG="${EXP_ROOT}/server_env_check.log"

cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate ehgnn

mkdir -p "${EXP_ROOT}"

{
  echo "=== Server environment check ==="
  echo "timestamp: $(date -Iseconds 2>/dev/null || date)"
  echo ""
  echo "--- paths ---"
  echo "pwd: $(pwd)"
  echo "exp_root_exists: $([ -d "${EXP_ROOT}" ] && echo yes || echo no)"
  echo ""
  echo "--- git ---"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
  echo ""
  echo "--- python / torch / dgl ---"
  echo "python_path: $(command -v python)"
  python --version 2>&1 | sed 's/^/python_version: /'
  python - <<'PY'
import sys
try:
    import torch
    print("torch_version:", torch.__version__)
    print("torch_cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch_cuda_device_count:", torch.cuda.device_count())
        print("torch_cuda_device_name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch_error:", e)
try:
    import dgl
    print("dgl_version:", dgl.__version__)
except Exception as e:
    print("dgl_error:", e)
PY
  echo ""
  echo "--- nvidia-smi ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi: not found"
  fi
  echo ""
  echo "=== End check ==="
} 2>&1 | tee "${OUT_LOG}"

echo "Saved to ${OUT_LOG}"
