"""Node Classification：alpha 敏感性（对应论文 Fig.4）。

README 给出各数据集 NC 默认 alpha=0.7；本脚本在敏感度实验中扫描 alpha，
README 其余字段固定（含 K=20）。不改变默认 5-seed 正式脚本行为。

输出 results/nc_sensitivity_alpha/；日志 <dataset>_alpha_<alpha>_seed_<seed>.log。"""
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
MAIN_YELP = os.path.join(SCRIPT_DIR, 'main_yelp.py')
DATA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))
OUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'nc_sensitivity_alpha')

DEFAULT_SEEDS = [42, 3407, 2026, 6666, 8888]
DEFAULT_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

_RE_BEST = re.compile(
    r'Best Test Macro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Micro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Epoch\s*:\s*([-0-9]+)',
)
_RE_FINAL = re.compile(
    r'Final Epoch\s*:\s*([-0-9]+)\s*,\s*Final Test Macro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Micro-F1\s*:\s*([0-9.+-eE]+)',
)
_RE_TIME = re.compile(r'Total training time:\s*([0-9.+-eE]+)\s*s')

CSV_FIELDS = [
    'dataset', 'alpha', 'seed', 'best_test_macro', 'best_test_micro', 'best_epoch',
    'final_epoch', 'final_test_macro', 'final_test_micro', 'total_training_time_sec',
]


def readme_nc(ds):
    if ds == 'PubMed':
        return MAIN_PY, [
            '--dataset', 'PubMed', '--path', '../data/',
            '--K', '20', '--lr', '1e-3', '--dropout', '0.4',
            '--hidden', '256', '--n_layers', '4', '--batch_size', '3000',
        ]
    if ds == 'DBLP':
        return MAIN_PY, [
            '--dataset', 'DBLP', '--path', '../data/',
            '--K', '20', '--lr', '5e-4', '--dropout', '0.5',
            '--hidden', '512', '--n_layers', '5', '--batch_size', '3000',
        ]
    if ds == 'Yelp':
        return MAIN_YELP, [
            '--dataset', 'Yelp', '--path', '../data/',
            '--K', '20', '--lr', '0.0003', '--dropout', '0.5',
            '--hidden', '256', '--n_layers', '4', '--batch_size', '3000',
        ]
    raise ValueError(ds)


def parse_nc(text):
    if not (text or '').strip():
        raise ValueError('empty')
    mb = _RE_BEST.search(text)
    mf = _RE_FINAL.search(text)
    mt = _RE_TIME.search(text)
    if not mb or not mf or not mt:
        raise ValueError('parse')
    return {
        'best_test_macro': float(mb.group(1)),
        'best_test_micro': float(mb.group(2)),
        'best_epoch': int(mb.group(3)),
        'final_epoch': int(mf.group(1)),
        'final_test_macro': float(mf.group(2)),
        'final_test_micro': float(mf.group(3)),
        'total_training_time_sec': float(mt.group(1)),
    }


def _tail(s, n):
    return '\n'.join((s or '').splitlines()[-n:])


def fail_proc(proc, cmd, log_path):
    print('[FAIL] return_code={}'.format(proc.returncode), file=sys.stderr, flush=True)
    print('[FAIL] cmd:', ' '.join(cmd), file=sys.stderr, flush=True)
    print('[FAIL] log_path:', log_path, file=sys.stderr, flush=True)
    print('[FAIL] stdout tail 80:', file=sys.stderr, flush=True)
    print(_tail(proc.stdout, 80), file=sys.stderr, flush=True)
    print('[FAIL] stderr tail 80:', file=sys.stderr, flush=True)
    print(_tail(proc.stderr, 80), file=sys.stderr, flush=True)
    sys.exit(1)


def fail_parse(log_path, text):
    print('[FAIL] parse_nc failed', file=sys.stderr, flush=True)
    print('[FAIL] log_path:', log_path, file=sys.stderr, flush=True)
    print('[FAIL] log tail 80:', file=sys.stderr, flush=True)
    print(_tail(text, 80), file=sys.stderr, flush=True)
    sys.exit(1)


def run_cmd(cmd):
    return subprocess.run(
        cmd, cwd=SCRIPT_DIR, capture_output=True, text=True,
        encoding='utf-8', errors='replace',
    )


def try_log(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return parse_nc(f.read())
    except (OSError, ValueError):
        return None


def std_ddof(vals):
    a = np.array(vals, dtype=np.float64)
    return 0.0 if a.size <= 1 else float(np.std(a, ddof=1))


def _alpha_tag(a):
    s = '{:.4g}'.format(a).replace('.', 'p')
    return s


def main():
    ap = argparse.ArgumentParser(description='NC alpha sensitivity (Fig.4)')
    ap.add_argument('--datasets', nargs='+', default=['PubMed', 'DBLP', 'Yelp'],
                    choices=['PubMed', 'DBLP', 'Yelp'])
    ap.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--skip_existing', action='store_true')
    args = ap.parse_args()
    seeds = args.seeds if args.seeds is not None else list(DEFAULT_SEEDS)

    for ds in args.datasets:
        if not os.path.isdir(os.path.join(DATA_ROOT, ds)):
            print('ERROR missing data', file=sys.stderr)
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for ds in args.datasets:
        exe, base = readme_nc(ds)
        for alpha in args.alphas:
            for seed in seeds:
                tag = _alpha_tag(alpha)
                log_name = '{}_alpha_{}_seed_{}.log'.format(ds, tag, seed)
                log_path = os.path.join(OUT_DIR, log_name)
                if args.skip_existing:
                    m = try_log(log_path)
                    if m:
                        print('[SKIP]', log_name, flush=True)
                        rows.append({'dataset': ds, 'alpha': alpha, 'seed': seed, **m})
                        continue
                cmd = [sys.executable, exe] + base + ['--alpha', str(alpha), '--seed', str(seed)]
                print('[RUN]', log_name, flush=True)
                proc = run_cmd(cmd)
                merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(proc.stdout or '')
                    if proc.stderr:
                        f.write('\n--- stderr ---\n')
                        f.write(proc.stderr)
                if proc.returncode != 0:
                    fail_proc(proc, cmd, log_path)
                try:
                    m = parse_nc(merged)
                except ValueError:
                    fail_parse(log_path, merged)
                rows.append({'dataset': ds, 'alpha': alpha, 'seed': seed, **m})

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_FIELDS})

    lines = ['NC alpha sensitivity (Fig.4)', 'seeds: {}'.format(seeds), '']
    for ds in args.datasets:
        lines.append('=== {} ==='.format(ds))
        lines.append('alpha\tbest_macro mean±std\tbest_micro mean±std')
        for alpha in args.alphas:
            ms = [r['best_test_macro'] for r in rows if r['dataset'] == ds and r['alpha'] == alpha]
            is_ = [r['best_test_micro'] for r in rows if r['dataset'] == ds and r['alpha'] == alpha]
            if not ms:
                continue
            lines.append(
                '{}\t{:.6f} ± {:.6f}\t{:.6f} ± {:.6f}'.format(
                    alpha, np.mean(ms), std_ddof(ms), np.mean(is_), std_ddof(is_),
                ),
            )
        lines.append('')
    lines.extend(['OUT:', OUT_DIR, csv_path])
    txt = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)


if __name__ == '__main__':
    main()
