"""PubMed Link Prediction：README 超参 × 多种子，调用现有 main.py。

输出：`results/pubmed_lp_3seeds/` 下各 seed 的 `.log`、`summary.csv`、`summary.txt`。
指标从 stdout 解析；不依赖 Node Classification。"""
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
OUT_DIR = os.path.join(RESULTS_ROOT, 'pubmed_lp_3seeds')

DEFAULT_SEEDS = [42, 3407, 2026]

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
    print('[FAIL] --- stdout (tail) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stdout or ''), file=sys.stderr, flush=True)
    print('[FAIL] --- stderr (tail) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stderr or ''), file=sys.stderr, flush=True)
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
        print(
            'ERROR: failed to parse metrics. log={}'.format(log_path_for_error),
            file=sys.stderr,
            flush=True,
        )
        print('--- log/output (last 80 lines) ---', file=sys.stderr, flush=True)
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
    ap = argparse.ArgumentParser(description='PubMed Link Prediction multi-seed (README hyperparameters)')
    ap.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(DEFAULT_SEEDS),
        metavar='SEED',
        help='Random seeds (default: %(default)s)',
    )
    ap.add_argument(
        '--skip_existing',
        action='store_true',
        help='If log exists and parses, skip re-run',
    )
    args_cli = ap.parse_args()
    seeds = list(args_cli.seeds)
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

    aucs = [r['best_test_auc'] for r in rows]
    aps = [r['best_test_ap'] for r in rows]
    mean_auc = float(np.mean(aucs)) if aucs else float('nan')
    mean_ap = float(np.mean(aps)) if aps else float('nan')
    std_auc = _std_ddof1_safe(aucs)
    std_ap = _std_ddof1_safe(aps)
    avg_time = float(np.mean([r['total_training_time_sec'] for r in rows])) if rows else float('nan')

    ibest = int(np.argmax(aucs)) if aucs else -1
    iworst = int(np.argmin(aucs)) if aucs else -1
    best_seed = rows[ibest]['seed'] if ibest >= 0 else None
    worst_seed = rows[iworst]['seed'] if iworst >= 0 else None

    def fmt_pm(mean, std):
        if len(seeds) <= 1:
            return '{:.6f} ± {:.6f}'.format(mean, std)
        return '{:.6f} ± {:.6f}'.format(mean, std)

    lines = [
        'PubMed Link Prediction — README hyperparameters',
        'seeds: {}'.format(seeds),
        'best tracked by Test AUC during training (tie-break by AP); worst seed = lowest best_test_auc',
        '',
        'Per seed:',
        'seed\tbest_auc\tbest_ap\tbest_epoch\tfinal_auc\tfinal_ap\ttrain_time_s',
    ]
    for r in rows:
        lines.append(
            '{}\t{:.6f}\t{:.6f}\t{}\t{:.6f}\t{:.6f}\t{:.4f}'.format(
                r['seed'],
                r['best_test_auc'],
                r['best_test_ap'],
                r['best_epoch'],
                r['final_test_auc'],
                r['final_test_ap'],
                r['total_training_time_sec'],
            ),
        )
    lines.extend([
        '',
        'Aggregate best_test_auc: {}'.format(fmt_pm(mean_auc, std_auc)),
        'Aggregate best_test_ap:  {}'.format(fmt_pm(mean_ap, std_ap)),
        'Average total_training_time_sec: {:.4f}'.format(avg_time),
        'Best seed (by best_test_auc): {}'.format(best_seed),
        'Worst seed (by best_test_auc): {}'.format(worst_seed),
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
