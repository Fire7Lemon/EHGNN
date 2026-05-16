"""Yelp Link Prediction 本地 smoke：epochs=3、val_epochs=1、seed=42，README LP 超参。

README.md Link Prediction 表未列出 walk_num；本 smoke 沿用 main_yelp.py 默认 --walk_num 100，
若与论文不一致请人工复核。

调用 **`main_yelp.py`**。服务器完整复现使用 **`run_yelp_lp_5seeds.py`**。
步级日志 **`precision`** = AP。"""
from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_YELP = os.path.join(SCRIPT_DIR, 'main_yelp.py')
DATA_YELP = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'Yelp'))
OUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'yelp_lp_smoke')
LOG_PATH = os.path.join(OUT_DIR, 'yelp_lp_smoke_seed_42.log')

CORE_MARKERS = (
    'Done Load Data',
    'Done my sim',
    'Begin Train',
)


def _preflight_or_exit():
    if not os.path.isfile(MAIN_YELP):
        print('ERROR: main_yelp.py not found at {}'.format(MAIN_YELP), file=sys.stderr, flush=True)
        sys.exit(1)
    if not os.path.isdir(DATA_YELP):
        print('ERROR: Yelp data directory not found: {}'.format(DATA_YELP), file=sys.stderr, flush=True)
        sys.exit(1)


def _tail_lines(text, n_lines=80):
    if not text:
        return '(empty)'
    lines = text.splitlines()
    if len(lines) <= n_lines:
        return text
    return '\n'.join(lines[-n_lines:])


def main():
    _preflight_or_exit()
    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        MAIN_YELP,
        '--dataset', 'Yelp',
        '--path', '../data/',
        '--seed', '42',
        '--epochs', '3',
        '--val_epochs', '1',
        '--alpha', '0.1',
        '--K', '20',
        '--lr', '3e-4',
        '--dropout', '0.5',
        '--hidden', '256',
        '--n_layers', '4',
        '--batch_size', '100',
    ]

    print('[SMOKE] cwd=', SCRIPT_DIR, flush=True)
    print('[SMOKE] log=', LOG_PATH, flush=True)
    print('[SMOKE] cmd=', ' '.join(cmd), flush=True)

    proc = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )

    merged = (proc.stdout or '') + ('\n--- stderr ---\n' + proc.stderr if proc.stderr else '')
    with open(LOG_PATH, 'w', encoding='utf-8') as lf:
        lf.write(proc.stdout or '')
        if proc.stderr:
            lf.write('\n--- stderr ---\n')
            lf.write(proc.stderr)

    if proc.returncode != 0:
        print('[FAIL] return_code={}'.format(proc.returncode), file=sys.stderr, flush=True)
        print('[FAIL] cmd:', ' '.join(cmd), file=sys.stderr, flush=True)
        print('[FAIL] --- stdout (last 80 lines) ---', file=sys.stderr, flush=True)
        print(_tail_lines(proc.stdout, 80), file=sys.stderr, flush=True)
        print('[FAIL] --- stderr (last 80 lines) ---', file=sys.stderr, flush=True)
        print(_tail_lines(proc.stderr, 80), file=sys.stderr, flush=True)
        sys.exit(proc.returncode)

    text = merged
    missing_core = [h for h in CORE_MARKERS if h not in text]
    has_metric_line = ('Test auc' in text) or ('Best Test AUC' in text)
    if missing_core or not has_metric_line:
        print('[WARN] Smoke finished exit 0 but log may be incomplete. Check:', LOG_PATH, flush=True)
        if missing_core:
            print('[WARN] missing:', missing_core, flush=True)
        if not has_metric_line:
            print('[WARN] no Test auc nor Best Test AUC line found', flush=True)

    print('[OK] wrote', LOG_PATH, flush=True)
    print(_tail_lines(proc.stdout or merged, 25))


if __name__ == '__main__':
    main()
