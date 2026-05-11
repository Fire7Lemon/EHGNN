"""
PubMed ablation study: multiple configurations x seeds, aggregate mean ± std.
Uses existing CLI flags only; does not modify training or model code.
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'pubmed_ablation')

SEEDS = [42, 3407, 2026]

# (paper_row_name, file_tag, extra_argv)
ABLATIONS = [
    ('EHGNN', 'full', []),
    ('w/o L2', 'wo_l2', ['--wo_l2']),
    ('w/o MWeight', 'wo_mweight', ['--wo_mweight']),
    ('w/o TWeight', 'wo_tweight', ['--wo_tweight']),
    ('Random Neighbor', 'r_neighbor', ['--r_neighbor']),
]


def parse_kv_txt(path):
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for paper_name, tag, extra in ABLATIONS:
        for seed in SEEDS:
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'main.py'), '--seed', str(seed)] + extra
            print('\n>>> {}'.format(' '.join(cmd)), flush=True)
            subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
            src = os.path.join(RESULTS_ROOT, 'pubmed_nc_result.txt')
            if not os.path.isfile(src):
                raise FileNotFoundError('Missing {} after {} seed={}'.format(src, tag, seed))
            dst = os.path.join(OUT_DIR, '{}_seed_{}.txt'.format(tag, seed))
            shutil.copy2(src, dst)
            kv = parse_kv_txt(src)
            rows.append({
                'method': paper_name,
                'tag': tag,
                'seed': int(kv['seed']),
                'best_test_macro': float(kv['best_test_macro']),
                'best_test_micro': float(kv['best_test_micro']),
            })

    # Per-method aggregates (best_test_macro / micro only)
    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    fieldnames = ['method', 'tag', 'seed', 'best_test_macro', 'best_test_micro']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    agg = []
    std_ddof = 1
    for paper_name, tag, _ in ABLATIONS:
        sel = [r for r in rows if r['tag'] == tag]
        bm = np.array([r['best_test_macro'] for r in sel], dtype=np.float64)
        bmi = np.array([r['best_test_micro'] for r in sel], dtype=np.float64)
        agg.append({
            'method': paper_name,
            'tag': tag,
            'macro_mean': float(np.mean(bm)),
            'macro_std': float(np.std(bm, ddof=std_ddof)),
            'micro_mean': float(np.mean(bmi)),
            'micro_std': float(np.std(bmi, ddof=std_ddof)),
        })

    summary_txt = os.path.join(OUT_DIR, 'summary.txt')
    lines = [
        'PubMed Node Classification — Ablation (3 seeds: {})'.format(SEEDS),
        '',
        '| Method | Macro-F1 | Micro-F1 |',
        '|--------|----------|----------|',
    ]
    for a in agg:
        lines.append('| {} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} |'.format(
            a['method'], a['macro_mean'], a['macro_std'],
            a['micro_mean'], a['micro_std']))
    lines.append('')

    full = agg[0]
    deltas = []
    for a in agg[1:]:
        deltas.append({
            'method': a['method'],
            'macro_drop': full['macro_mean'] - a['macro_mean'],
            'micro_drop': full['micro_mean'] - a['micro_mean'],
        })
    deltas.sort(key=lambda x: x['macro_drop'], reverse=True)

    wo_only = [d for d in deltas if d['method'] != 'Random Neighbor']
    rn_only = [d for d in deltas if d['method'] == 'Random Neighbor']
    wo_only.sort(key=lambda x: x['macro_drop'], reverse=True)

    lines.extend([
        'Δmacro / Δmicro vs EHGNN mean: positive ⇒ worse (macro/micro lower than full).',
        '(Destructive ablations: w/o L2, w/o MWeight, w/o TWeight)',
        '',
    ])
    for d in wo_only:
        lines.append('  {:24s}  Δmacro={:+.4f}  Δmicro={:+.4f}'.format(
            d['method'], d['macro_drop'], d['micro_drop']))
    if rn_only:
        r = rn_only[0]
        lines.append('  {:24s}  Δmacro={:+.4f}  Δmicro={:+.4f}  (changes RW neighbor sampling, not a weight removal)'.format(
            r['method'], r['macro_drop'], r['micro_drop']))
    lines.append('')

    if wo_only:
        lines.extend([
            'Among weight/normalization ablations, largest degradation: {} (Δmacro={:.4f})'.format(
                wo_only[0]['method'], wo_only[0]['macro_drop']),
            'Among weight/normalization ablations, smallest degradation: {} (Δmacro={:.4f})'.format(
                wo_only[-1]['method'], wo_only[-1]['macro_drop']),
        ])

    text = '\n'.join(lines) + '\n'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text)

    print('\n' + text)
    print('CSV:', csv_path)


if __name__ == '__main__':
    run_all()
