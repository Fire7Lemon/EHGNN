"""PubMed Link Prediction 本地 smoke test：极少 epoch，打通 README 超参与日志管线。

本脚本不写汇总 CSV；完整训练请在服务器上使用 **`run_pubmed_lp_3seeds.py`**（或其它 README `--epochs` 设定）。

成功标准（日志中建议包含）：
- `Done Load Data`、`Done my sim`、`Begin Train`
- 至少一次 `Test auc` / `precision`（**precision** = AP），或文末 **`Best Test AUC`** / **`Final Epoch`** / **`Total training time`**

脚本退出后若上述不完整会打印 **[WARN]**（仍以 main.py 退出码为准）。
"""
from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(SCRIPT_DIR, 'main.py')
DATA_PUBMED = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'PubMed'))
OUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'pubmed_lp_smoke')
LOG_PATH = os.path.join(OUT_DIR, 'pubmed_lp_smoke_seed_42.log')

CORE_MARKERS = (
    'Done Load Data',
    'Done my sim',
    'Begin Train',
)


def _preflight_or_exit():
    if not os.path.isfile(MAIN_PY):
        print('ERROR: main.py not found at {}'.format(MAIN_PY), file=sys.stderr, flush=True)
        sys.exit(1)
    if not os.path.isdir(DATA_PUBMED):
        print(
            'ERROR: PubMed data directory not found: {}'.format(DATA_PUBMED),
            file=sys.stderr,
            flush=True,
        )
        print('Place PubMed under EHGNN/data/PubMed (see README.md).', file=sys.stderr, flush=True)
        sys.exit(1)


def _tail_lines(text, n_lines=40):
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
        MAIN_PY,
        '--dataset', 'PubMed',
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
        '--batch_size', '40',
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
        print('[FAIL] --- stdout tail ---', file=sys.stderr, flush=True)
        print(_tail_lines(proc.stdout), file=sys.stderr, flush=True)
        print('[FAIL] --- stderr tail ---', file=sys.stderr, flush=True)
        print(_tail_lines(proc.stderr), file=sys.stderr, flush=True)
        sys.exit(proc.returncode)

    text = merged
    missing_core = [h for h in CORE_MARKERS if h not in text]
    has_metric_line = ('Test auc' in text) or ('Best Test AUC' in text)
    if missing_core or not has_metric_line:
        print(
            '[WARN] Smoke finished exit 0 but log may be incomplete. '
            'Check markers / metrics manually:',
            LOG_PATH,
            flush=True,
        )
        if missing_core:
            print('[WARN] missing:', missing_core, flush=True)
        if not has_metric_line:
            print('[WARN] no Test auc nor Best Test AUC line found', flush=True)

    print('[OK] wrote', LOG_PATH, flush=True)
    print(_tail_lines(proc.stdout or merged, 25))


if __name__ == '__main__':
    main()
