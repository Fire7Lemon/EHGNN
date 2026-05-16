#!/usr/bin/env bash
# Record GPU / RAM / disk / ps while running a long command (Linux server).
# Usage: bash scripts/run_server_resource_monitor.sh 'python "Node Classification/run_pubmed_5seeds.py"'
# Logs: logs/resource_monitor_<timestamp>.log every 60s. No extra Python deps.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT}/logs/resource_monitor_${TS}.log"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command>" >&2
  exit 1
fi
CMD="$*"

sample_once() {
  {
    echo "======== $(date -Is 2>/dev/null || date) ========"
    nvidia-smi 2>/dev/null || echo "(nvidia-smi unavailable)"
    echo "--- free -h ---"
    free -h 2>/dev/null || true
    echo "--- df -h (repo) ---"
    df -h "${ROOT}" 2>/dev/null || df -h . 2>/dev/null || true
    echo "--- ps (top memory, head) ---"
    ps aux --sort=-%mem 2>/dev/null | head -n 15 || ps aux 2>/dev/null | head -n 15 || true
    echo ""
  } >> "${LOG}"
}

(
  while true; do
    sample_once
    sleep 60
  done
) &
MON_PID=$!

cleanup() {
  kill "${MON_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[monitor] logging to ${LOG}" | tee -a "${LOG}"
echo "[monitor] command: ${CMD}" | tee -a "${LOG}"
sample_once

# shellcheck disable=SC2086
set +e
eval "${CMD}"
EXIT_CODE=$?
set -e

echo "[monitor] command exit_code=${EXIT_CODE}" | tee -a "${LOG}"
exit "${EXIT_CODE}"
