"""Yelp 节点分类：README 超参 × 五种子（服务器正式复现推荐）。

调用 **`main_yelp.py`**。输出：`results/yelp_5seeds/`。

历史 **`run_yelp_3seeds.py`**（`results/yelp_3seeds/`）保留兼容；文档中的 **selected 3-seed** 数值不等于默认 5 seeds。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_YELP = os.path.join(SCRIPT_DIR, 'main_yelp.py')
DATA_YELP = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'Yelp'))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'yelp_5seeds')

DEFAULT_SEEDS = [42, 3407, 2026, 6666, 8888]

_RE_BEST = re.compile(
    r'Best Test Macro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Micro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Epoch\s*:\s*([-0-9]+)',
)
_RE_FINAL = re.compile(
    r'Final Epoch\s*:\s*([-0-9]+)\s*,\s*Final Test Macro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Micro-F1\s*:\s*([0-9.+-eE]+)',
)
_RE_TIME = re.compile(
    r'Total training time:\s*([0-9.+-eE]+)\s*s',
)


def parse_metrics_from_text(text):
    """解析 main_yelp 文末 Best / Final / Total time；失败抛出 ValueError。"""
    out = text if text is not None else ''
    if not str(out).strip():
        raise ValueError('parse_metrics_from_text: empty text')

    mb = _RE_BEST.search(out)
    mf = _RE_FINAL.search(out)
    mt = _RE_TIME.search(out)

    missing = []
    if not mb:
        missing.append(
            "line matching 'Best Test Macro-F1 : …, Micro-F1 : …, Epoch : …'",
        )
    if not mf:
        missing.append(
            "line matching 'Final Epoch : …, Final Test Macro-F1 : …, Micro-F1 : …'",
        )
    if not mt:
        missing.append("line matching 'Total training time: … s'")

    if missing:
        raise ValueError(
            'parse_metrics_from_text failed: missing ' + '; '.join(missing),
        )

    return {
        'best_test_macro': float(mb.group(1)),
        'best_test_micro': float(mb.group(2)),
        'best_epoch': int(mb.group(3)),
        'final_epoch': int(mf.group(1)),
        'final_test_macro': float(mf.group(2)),
        'final_test_micro': float(mf.group(3)),
        'total_training_time_sec': float(mt.group(1)),
    }


def _preflight_or_exit():
    if not os.path.isfile(MAIN_YELP):
        print('ERROR: main_yelp.py not found at {}'.format(MAIN_YELP), file=sys.stderr, flush=True)
        sys.exit(1)
    if not os.path.isdir(DATA_YELP):
        print('ERROR: Yelp data directory not found: {}'.format(DATA_YELP), file=sys.stderr, flush=True)
        sys.exit(1)


def _run_yelp(cmd):
    return subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )


def _tail_lines(s, n):
    lines = (s or '').splitlines()
    return '\n'.join(lines[-n:])


def _fail_subprocess(proc, cmd, log_path):
    print('[FAIL] return_code={}'.format(proc.returncode), file=sys.stderr, flush=True)
    print('[FAIL] cmd:', ' '.join(cmd), file=sys.stderr, flush=True)
    print('[FAIL] log_path:', log_path, file=sys.stderr, flush=True)
    print('[FAIL] --- stdout (last 80 lines) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stdout or '', 80), file=sys.stderr, flush=True)
    print('[FAIL] --- stderr (last 80 lines) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stderr or '', 80), file=sys.stderr, flush=True)
    sys.exit(1)


def _fail_parse(log_path, text):
    stdout_only = text.split('--- stderr ---')[0].strip() if '--- stderr ---' in text else text
    tail = '\n'.join(stdout_only.splitlines()[-80:])
    print('[FAIL] parse_metrics_from_text failed', file=sys.stderr, flush=True)
    print('[FAIL] log_path:', log_path, file=sys.stderr, flush=True)
    print('[FAIL] --- last 80 lines (stdout portion if marked) ---', file=sys.stderr, flush=True)
    print(tail, file=sys.stderr, flush=True)
    sys.exit(1)


def _parse_or_exit(text, log_path):
    try:
        return parse_metrics_from_text(text)
    except ValueError:
        _fail_parse(log_path, text)


def parse_args():
    p = argparse.ArgumentParser(description='Yelp NC — formal 5-seed (README hyperparameters).')
    p.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        metavar='SEED',
        help='Random seeds; omit for {}'.format(DEFAULT_SEEDS),
    )
    p.add_argument(
        '--skip_existing',
        action='store_true',
        help='If yelp_seed_<seed>.log exists and parses, skip training',
    )
    return p.parse_args()


def main():
    args = parse_args()
    selected_seeds = args.seeds if args.seeds is not None else list(DEFAULT_SEEDS)

    print('[INFO] seeds = {}'.format(selected_seeds), flush=True)

    _preflight_or_exit()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    base = [
        sys.executable,
        MAIN_YELP,
        '--dataset', 'Yelp',
        '--path', '../data/',
        '--alpha', '0.7',
        '--dropout', '0.5',
        '--K', '20',
        '--lr', '0.0003',
        '--hidden', '256',
        '--n_layers', '4',
        '--batch_size', '3000',
    ]

    for seed in selected_seeds:
        log_path = os.path.join(OUT_DIR, 'yelp_seed_{}.log'.format(seed))

        if args.skip_existing and os.path.isfile(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as lf:
                existing = lf.read()
            try:
                metrics = parse_metrics_from_text(existing)
            except ValueError:
                _fail_parse(log_path, existing)
            print('[SKIP] seed={} use existing log'.format(seed), flush=True)
            rows.append({'seed': seed, **metrics})
            continue

        cmd = base + ['--seed', str(seed)]
        print('[RUN] seed={}'.format(seed), flush=True)
        print('[RUN] cmd={}'.format(' '.join(cmd)), flush=True)

        proc = _run_yelp(cmd)
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write(proc.stdout or '')
            if proc.stderr:
                lf.write('\n--- stderr ---\n')
                lf.write(proc.stderr)

        if proc.returncode != 0:
            _fail_subprocess(proc, cmd, log_path)

        merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
        metrics = _parse_or_exit(merged, log_path)
        rows.append({'seed': seed, **metrics})
        print('[DONE] seed={}'.format(seed), flush=True)

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    keys = [
        'seed', 'best_test_macro', 'best_test_micro', 'best_epoch',
        'final_epoch', 'final_test_macro', 'final_test_micro', 'total_training_time_sec',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    std_ddof = 1 if len(rows) > 1 else 0

    def stat(arr):
        a = np.array(arr, dtype=np.float64)
        return float(np.mean(a)), float(np.std(a, ddof=std_ddof))

    bm = [r['best_test_macro'] for r in rows]
    bmi = [r['best_test_micro'] for r in rows]
    fm = [r['final_test_macro'] for r in rows]
    fmi = [r['final_test_micro'] for r in rows]
    tt = [r['total_training_time_sec'] for r in rows]

    bm_m, bm_s = stat(bm)
    bmi_m, bmi_s = stat(bmi)
    fm_m, fm_s = stat(fm)
    fmi_m, fmi_s = stat(fmi)
    tt_avg = float(np.mean(tt))

    seeds_arr = np.array([r['seed'] for r in rows])
    ibest = int(np.argmax(bm))
    iworst = int(np.argmin(bm))

    summary_txt = os.path.join(OUT_DIR, 'summary.txt')
    lines = [
        'Yelp Node Classification — README hyperparameters (5-seed formal batch)',
        'seeds (this run): {}'.format(selected_seeds),
        '',
        'Per seed:',
        'seed\tbest_macro\tbest_micro\tbest_epoch\tfinal_epoch\tfinal_macro\tfinal_micro\ttrain_time_s',
    ]
    for r in rows:
        lines.append(
            '{seed}\t{best_test_macro:.6f}\t{best_test_micro:.6f}\t{best_epoch}\t{final_epoch}\t'
            '{final_test_macro:.6f}\t{final_test_micro:.6f}\t{total_training_time_sec:.4f}'.format(**r))

    lines.extend([
        '',
        'Aggregate (n={}, std ddof={})'.format(len(rows), std_ddof),
        'best_test_macro      {:.6f} ± {:.6f}'.format(bm_m, bm_s),
        'best_test_micro      {:.6f} ± {:.6f}'.format(bmi_m, bmi_s),
        'final_test_macro     {:.6f} ± {:.6f}'.format(fm_m, fm_s),
        'final_test_micro     {:.6f} ± {:.6f}'.format(fmi_m, fmi_s),
        'avg training time (s) {:.4f}'.format(tt_avg),
        '',
        'Best seed (by best_test_macro):  {}  macro={:.6f}'.format(seeds_arr[ibest], bm[ibest]),
        'Worst seed (by best_test_macro): {}  macro={:.6f}'.format(seeds_arr[iworst], bm[iworst]),
        '',
        OUT_DIR,
        csv_path,
    ])

    text = '\n'.join(lines) + '\n'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text)

    print('\n' + text)
    print('CSV:', csv_path)


if __name__ == '__main__':
    main()
