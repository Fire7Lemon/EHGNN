"""Link Prediction 全数据集消融（论文 Table VII LP 部分）。

README.md Link Prediction 表给出 α、K、lr、dropout、hidden、layers、batch。
未给出 walk_num：PubMed/DBLP 使用 Link Prediction/main.py 默认 --walk_num 100；
Yelp 使用 Link Prediction/main_yelp.py 默认 --walk_num 100（与 NC 默认 40 不一致，不在本轮改动）。

训练步日志中的 precision = AP（Average Precision）。

数据集：PubMed / DBLP → main.py；Yelp → main_yelp.py。
输出：results/lp_ablation_all_datasets/，日志 <dataset>_<method>_seed_<seed>.log。"""
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
OUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'lp_ablation_all_datasets')

DEFAULT_SEEDS = [42, 3407, 2026, 6666, 8888]
METHODS_ORDER = ['full', 'r_neighbor', 'wo_mweight', 'wo_tweight', 'wo_l2']

_RE_BEST = re.compile(
    r'Best Test AUC\s*:\s*([0-9.+-eE]+)\s*,\s*AP\s*:\s*([0-9.+-eE]+)\s*,\s*Epoch\s*:\s*([-0-9]+)',
)
_RE_FINAL = re.compile(
    r'Final Epoch\s*:\s*([-0-9]+)\s*,\s*Final Test AUC\s*:\s*([0-9.+-eE]+)\s*,\s*AP\s*:\s*([0-9.+-eE]+)',
)
_RE_TIME = re.compile(
    r'Total training time:\s*([0-9.+-eE]+)\s*s',
)

CSV_FIELDS = [
    'dataset', 'method', 'seed', 'best_test_auc', 'best_test_ap', 'best_epoch',
    'final_epoch', 'final_test_auc', 'final_test_ap', 'total_training_time_sec',
]


def _readme_lp_pubmed():
    return [
        '--dataset', 'PubMed', '--path', '../data/',
        '--alpha', '0.1', '--K', '20', '--lr', '3e-4', '--dropout', '0.5',
        '--hidden', '256', '--n_layers', '4', '--batch_size', '40',
    ]


def _readme_lp_dblp():
    # README LP DBLP：默认 alpha = 0.7（与 PubMed LP 的 0.1 不同）。
    return [
        '--dataset', 'DBLP', '--path', '../data/',
        '--alpha', '0.7', '--K', '20', '--lr', '5e-4', '--dropout', '0.5',
        '--hidden', '512', '--n_layers', '5', '--batch_size', '1000',
    ]


def _readme_lp_yelp():
    return [
        '--dataset', 'Yelp', '--path', '../data/',
        '--alpha', '0.1', '--K', '20', '--lr', '3e-4', '--dropout', '0.5',
        '--hidden', '256', '--n_layers', '4', '--batch_size', '100',
    ]


def dataset_readme_base(dataset: str):
    if dataset == 'PubMed':
        return _readme_lp_pubmed(), MAIN_PY
    if dataset == 'DBLP':
        return _readme_lp_dblp(), MAIN_PY
    if dataset == 'Yelp':
        return _readme_lp_yelp(), MAIN_YELP
    raise ValueError('unknown dataset')


def data_dir_ok(dataset: str) -> bool:
    return os.path.isdir(os.path.join(DATA_ROOT, dataset))


def method_extra_args(method: str) -> list[str]:
    if method == 'full':
        return []
    if method == 'r_neighbor':
        return ['--r_neighbor']
    if method == 'wo_mweight':
        return ['--wo_mweight']
    if method == 'wo_tweight':
        return ['--wo_tweight']
    if method == 'wo_l2':
        return ['--wo_l2']
    raise ValueError('unknown method')


def parse_lp_metrics(text: str):
    out = text or ''
    if not str(out).strip():
        raise ValueError('empty')
    mb = _RE_BEST.search(out)
    mf = _RE_FINAL.search(out)
    mt = _RE_TIME.search(out)
    if not mb or not mf or not mt:
        raise ValueError('missing lines')
    return {
        'best_test_auc': float(mb.group(1)),
        'best_test_ap': float(mb.group(2)),
        'best_epoch': int(mb.group(3)),
        'final_epoch': int(mf.group(1)),
        'final_test_auc': float(mf.group(2)),
        'final_test_ap': float(mf.group(3)),
        'total_training_time_sec': float(mt.group(1)),
    }


def _tail_lines(s, n):
    lines = (s or '').splitlines()
    return '\n'.join(lines[-n:])


def _fail_subprocess(proc, cmd, log_path):
    print('[FAIL] return_code={}'.format(proc.returncode), file=sys.stderr, flush=True)
    print('[FAIL] cmd:', ' '.join(cmd), file=sys.stderr, flush=True)
    print('[FAIL] log_path:', log_path, file=sys.stderr, flush=True)
    print('[FAIL] --- stdout (last 80 lines) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stdout or '', 80) or '(empty)', file=sys.stderr, flush=True)
    print('[FAIL] --- stderr (last 80 lines) ---', file=sys.stderr, flush=True)
    print(_tail_lines(proc.stderr or '', 80) or '(empty)', file=sys.stderr, flush=True)
    sys.exit(1)


def _fail_parse(log_path, text):
    tail = '\n'.join((text or '').splitlines()[-80:])
    print('[FAIL] parse_lp_metrics failed', file=sys.stderr, flush=True)
    print('[FAIL] log_path:', log_path, file=sys.stderr, flush=True)
    print('[FAIL] --- last 80 lines ---', file=sys.stderr, flush=True)
    print(tail, file=sys.stderr, flush=True)
    sys.exit(1)


def _run(cmd):
    return subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )


def _try_parse_log(log_path):
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as lf:
            t = lf.read()
    except OSError:
        return None
    try:
        return parse_lp_metrics(t)
    except ValueError:
        return None


def _std_ddof(arr):
    a = np.array(arr, dtype=np.float64)
    if a.size <= 1:
        return 0.0
    return float(np.std(a, ddof=1))


def parse_args():
    p = argparse.ArgumentParser(description='LP ablation all datasets (Table VII LP).')
    p.add_argument(
        '--datasets',
        nargs='+',
        default=['PubMed', 'DBLP', 'Yelp'],
        choices=['PubMed', 'DBLP', 'Yelp'],
    )
    p.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        metavar='SEED',
        help='default: {}'.format(DEFAULT_SEEDS),
    )
    p.add_argument('--skip_existing', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = list(args.seeds) if args.seeds is not None else list(DEFAULT_SEEDS)

    for ds in args.datasets:
        if not data_dir_ok(ds):
            print('ERROR: missing data for {}'.format(ds), file=sys.stderr, flush=True)
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        base, exe = dataset_readme_base(dataset)
        for method in METHODS_ORDER:
            extras = method_extra_args(method)
            for seed in seeds:
                log_name = '{}_{}_seed_{}.log'.format(dataset, method, seed)
                log_path = os.path.join(OUT_DIR, log_name)

                if args.skip_existing:
                    parsed = _try_parse_log(log_path)
                    if parsed is not None:
                        print('[SKIP] {}'.format(log_name), flush=True)
                        rows.append({'dataset': dataset, 'method': method, 'seed': seed, **parsed})
                        continue

                cmd = [sys.executable, exe] + base + extras + ['--seed', str(seed)]
                print('[RUN]', log_name, flush=True)
                proc = _run(cmd)
                merged = (proc.stdout or '') + '\n' + (proc.stderr or '')
                with open(log_path, 'w', encoding='utf-8') as lf:
                    lf.write(proc.stdout or '')
                    if proc.stderr:
                        lf.write('\n--- stderr ---\n')
                        lf.write(proc.stderr)

                if proc.returncode != 0:
                    _fail_subprocess(proc, cmd, log_path)

                try:
                    metrics = parse_lp_metrics(merged)
                except ValueError:
                    _fail_parse(log_path, merged)

                rows.append({'dataset': dataset, 'method': method, 'seed': seed, **metrics})

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_FIELDS})

    lines_out = ['Link Prediction ablation — all datasets', 'seeds: {}'.format(seeds), '']
    for dataset in args.datasets:
        subset = [r for r in rows if r['dataset'] == dataset]
        lines_out.append('=== dataset: {} ==='.format(dataset))
        lines_out.append(
            'Method\tROC-AUC (mean ± std)\tAP (mean ± std)\tΔAUC vs full\tΔAP vs full',
        )

        f_aucs = [r['best_test_auc'] for r in subset if r['method'] == 'full']
        f_aps = [r['best_test_ap'] for r in subset if r['method'] == 'full']
        ba = float(np.mean(f_aucs)) if f_aucs else float('nan')
        bp = float(np.mean(f_aps)) if f_aps else float('nan')

        for method in METHODS_ORDER:
            au = [r['best_test_auc'] for r in subset if r['method'] == method]
            ap = [r['best_test_ap'] for r in subset if r['method'] == method]
            if not au:
                continue
            am, sa = float(np.mean(au)), _std_ddof(au)
            pm, sp = float(np.mean(ap)), _std_ddof(ap)
            lines_out.append(
                '{}\t{:.6f} ± {:.6f}\t{:.6f} ± {:.6f}\t{:+.6f}\t{:+.6f}'.format(
                    method, am, sa, pm, sp, am - ba, pm - bp,
                ),
            )
        lines_out.append('')
        lines_out.append(
            'Note: step logs may print precision; final summary AP matches sklearn AP.',
        )
        lines_out.append('')

    lines_out.extend(['OUT_DIR:', OUT_DIR, csv_path])
    txt_path = os.path.join(OUT_DIR, 'summary.txt')
    text = '\n'.join(lines_out) + '\n'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
