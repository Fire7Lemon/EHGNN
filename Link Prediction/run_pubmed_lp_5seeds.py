"""PubMed Link Prediction：README LP 超参 × 五种子（服务器正式复现推荐）。

调用 **`main.py`**。默认 seeds = [42, 3407, 2026, 6666, 8888]；历史 **`run_pubmed_lp_3seeds.py`** 仍保留兼容。

输出：`results/pubmed_lp_5seeds/` 下各 seed 日志、`summary.csv`、`summary.txt`。
日志中 **`precision`** 与 **`AP`** 均指 Average Precision（平均精确率），非 Accuracy。"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(SCRIPT_DIR, 'main.py')
DATA_PUBMED = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'PubMed'))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'pubmed_lp_5seeds')

DEFAULT_SEEDS = [42, 3407, 2026, 6666, 8888]

README_LP_CMD = [
    '--dataset', 'PubMed',
    '--path', '../data/',
    '--alpha', '0.1',
    '--K', '20',
    '--lr', '3e-4',
    '--dropout', '0.5',
    '--hidden', '256',
    '--n_layers', '4',
    '--batch_size', '40',
]

_RE_BEST = re.compile(
    r'Best Test AUC\s*:\s*([0-9.+-eE]+)\s*,\s*AP\s*:\s*([0-9.+-eE]+)\s*,\s*Epoch\s*:\s*([-0-9]+)',
)
_RE_FINAL = re.compile(
    r'Final Epoch\s*:\s*([-0-9]+)\s*,\s*Final Test AUC\s*:\s*([0-9.+-eE]+)\s*,\s*AP\s*:\s*([0-9.+-eE]+)',
)
_RE_TIME = re.compile(
    r'Total training time:\s*([0-9.+-eE]+)\s*s',
)

CSV_FIELDNAMES = [
    'seed',
    'best_test_auc',
    'best_test_ap',
    'best_epoch',
    'final_epoch',
    'final_test_auc',
    'final_test_ap',
    'total_training_time_sec',
]


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


def _run_main_capture(cmd):
    return subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )


def _tail_lines(text, n_lines=80):
    if not text:
        return '(empty)'
    lines = text.splitlines()
    if len(lines) <= n_lines:
        return text
    return '\n'.join(lines[-n_lines:])


def _fail(proc, cmd):
    print('[FAIL] return_code={}'.format(proc.returncode), file=sys.stderr, flush=True)
    print('[FAIL] cmd:', ' '.join(cmd), file=sys.stderr, flush=True)
    print('[FAIL] --- stdout (last 80 lines) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stdout or '', 80), file=sys.stderr, flush=True)
    print('[FAIL] --- stderr (last 80 lines) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stderr or '', 80), file=sys.stderr, flush=True)
    sys.exit(proc.returncode if proc.returncode != 0 else 1)


def _parse_metrics_dict(text):
    out = text or ''
    mb = _RE_BEST.search(out)
    mf = _RE_FINAL.search(out)
    mt = _RE_TIME.search(out)
    if not mb or not mf or not mt:
        return None
    try:
        return {
            'best_test_auc': float(mb.group(1)),
            'best_test_ap': float(mb.group(2)),
            'best_epoch': int(mb.group(3)),
            'final_epoch': int(mf.group(1)),
            'final_test_auc': float(mf.group(2)),
            'final_test_ap': float(mf.group(3)),
            'total_training_time_sec': float(mt.group(1)),
        }
    except ValueError:
        return None


def parse_main_output(text, log_path_for_error):
    metrics = _parse_metrics_dict(text)
    if metrics is None:
        print('[FAIL] parse_main_output: missing Best/Final/Total lines', file=sys.stderr, flush=True)
        print('[FAIL] log_path:', log_path_for_error, file=sys.stderr, flush=True)
        print('[FAIL] --- merged output (last 80 lines) ---', file=sys.stderr, flush=True)
        print(_tail_lines(text or '', 80), file=sys.stderr, flush=True)
        sys.exit(1)
    return metrics


def _try_parse_log(log_path):
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as lf:
            content = lf.read()
    except OSError:
        return None
    return _parse_metrics_dict(content)


def _std_ddof1_safe(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def main():
    ap = argparse.ArgumentParser(description='PubMed Link Prediction — formal 5-seed reproduction (README LP)')
    ap.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        metavar='SEED',
        help='Random seeds (default: {})'.format(DEFAULT_SEEDS),
    )
    ap.add_argument(
        '--skip_existing',
        action='store_true',
        help='If log exists and parses, skip re-run',
    )
    args_cli = ap.parse_args()
    seeds = list(args_cli.seeds) if args_cli.seeds is not None else list(DEFAULT_SEEDS)
    skip_existing = args_cli.skip_existing

    _preflight_or_exit()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    base_cmd = [sys.executable, MAIN_PY] + README_LP_CMD

    for seed in seeds:
        cmd = base_cmd + ['--seed', str(seed)]
        log_path = os.path.join(OUT_DIR, 'pubmed_lp_seed_{}.log'.format(seed))

        reused = False
        if skip_existing and os.path.isfile(log_path):
            parsed = _try_parse_log(log_path)
            if parsed is not None:
                reused = True
                metrics = parsed
                print('[SKIP] seed={} (parsed {})'.format(seed, log_path), flush=True)
            else:
                print(
                    '[WARN] skip_existing but log not parseable, re-run: {}'.format(log_path),
                    file=sys.stderr,
                    flush=True,
                )

        if not reused:
            print('[RUN] seed={}'.format(seed), flush=True)
            print('[RUN] cmd={}'.format(' '.join(cmd)), flush=True)
            proc = _run_main_capture(cmd)
            with open(log_path, 'w', encoding='utf-8') as lf:
                lf.write(proc.stdout or '')
                if proc.stderr:
                    lf.write('\n--- stderr ---\n')
                    lf.write(proc.stderr)
            if proc.returncode != 0:
                _fail(proc, cmd)
            merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
            metrics = parse_main_output(merged, log_path)

        row = {'seed': seed, **metrics}
        rows.append(row)

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_FIELDNAMES})

    bauc = [r['best_test_auc'] for r in rows]
    bap = [r['best_test_ap'] for r in rows]
    fauc = [r['final_test_auc'] for r in rows]
    fap = [r['final_test_ap'] for r in rows]
    tt = [r['total_training_time_sec'] for r in rows]

    mean_bauc = float(np.mean(bauc)) if bauc else float('nan')
    mean_bap = float(np.mean(bap)) if bap else float('nan')
    mean_fauc = float(np.mean(fauc)) if fauc else float('nan')
    mean_fap = float(np.mean(fap)) if fap else float('nan')
    std_bauc = _std_ddof1_safe(bauc)
    std_bap = _std_ddof1_safe(bap)
    std_fauc = _std_ddof1_safe(fauc)
    std_fap = _std_ddof1_safe(fap)
    avg_time = float(np.mean(tt)) if tt else float('nan')

    ibest = int(np.argmax(bauc)) if bauc else -1
    iworst = int(np.argmin(bauc)) if bauc else -1
    best_seed = rows[ibest]['seed'] if ibest >= 0 else None
    worst_seed = rows[iworst]['seed'] if iworst >= 0 else None

    lines = [
        'PubMed Link Prediction — README LP hyperparameters',
        'seeds (this run): {}'.format(seeds),
        'Best tracked by Test AUC during training (tie-break by AP); worst seed = lowest best_test_auc.',
        '',
        'Per seed:',
        'seed\tbest_test_auc\tbest_test_ap\tbest_epoch\tfinal_epoch\tfinal_test_auc\tfinal_test_ap\ttotal_training_time_sec',
    ]
    for r in rows:
        lines.append(
            '{seed}\t{best_test_auc:.6f}\t{best_test_ap:.6f}\t{best_epoch}\t{final_epoch}\t'
            '{final_test_auc:.6f}\t{final_test_ap:.6f}\t{total_training_time_sec:.4f}'.format(**r),
        )
    lines.extend([
        '',
        'Aggregate (sample std ddof=1; single seed → std = 0)',
        'best_test_auc  {:.6f} ± {:.6f}'.format(mean_bauc, std_bauc),
        'best_test_ap    {:.6f} ± {:.6f}'.format(mean_bap, std_bap),
        'final_test_auc  {:.6f} ± {:.6f}'.format(mean_fauc, std_fauc),
        'final_test_ap   {:.6f} ± {:.6f}'.format(mean_fap, std_fap),
        'average training time (s) {:.4f}'.format(avg_time),
        '',
        'best seed (by best_test_auc):  {}'.format(best_seed),
        'worst seed (by best_test_auc): {}'.format(worst_seed),
        '',
        OUT_DIR,
        csv_path,
    ])
    summary_txt = os.path.join(OUT_DIR, 'summary.txt')
    text = '\n'.join(lines) + '\n'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text)

    print('\n' + text)


if __name__ == '__main__':
    main()
