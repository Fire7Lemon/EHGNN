"""
PubMed neighbor selection strategies: freq / random / hybrid / temp x seeds.
Runs main.py subprocesses; copies pubmed_nc_result.txt per run; writes summary.csv and summary.txt.
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'pubmed_neighbor_strategy')

SEEDS = [42, 3407, 2026]

# (strategy_display_name, file_tag, extra_argv)
CONFIGS = [
    ('freq', 'freq', ['--neighbor_strategy', 'freq']),
    ('random', 'random', ['--neighbor_strategy', 'random']),
    ('hybrid (ratio=0.8)', 'hybrid_r08', ['--neighbor_strategy', 'hybrid', '--hybrid_ratio', '0.8']),
    ('hybrid (ratio=0.5)', 'hybrid_r05', ['--neighbor_strategy', 'hybrid', '--hybrid_ratio', '0.5']),
    ('temp (t=0.5)', 'temp_05', ['--neighbor_strategy', 'temp', '--temp', '0.5']),
    ('temp (t=1.0)', 'temp_10', ['--neighbor_strategy', 'temp', '--temp', '1.0']),
    ('temp (t=2.0)', 'temp_20', ['--neighbor_strategy', 'temp', '--temp', '2.0']),
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

    base_cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'main.py'), '--dataset', 'PubMed']

    for strat_name, tag, extra in CONFIGS:
        for seed in SEEDS:
            cmd = base_cmd + ['--seed', str(seed)] + extra
            print('\n>>> {}'.format(' '.join(cmd)), flush=True)
            subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
            src = os.path.join(RESULTS_ROOT, 'pubmed_nc_result.txt')
            if not os.path.isfile(src):
                raise FileNotFoundError('Missing {} after {} seed={}'.format(src, tag, seed))
            dst = os.path.join(OUT_DIR, '{}_seed_{}.txt'.format(tag, seed))
            shutil.copy2(src, dst)
            kv = parse_kv_txt(src)
            rows.append({
                'strategy': strat_name,
                'tag': tag,
                'seed': int(kv['seed']),
                'neighbor_strategy': kv.get('neighbor_strategy', 'freq'),
                'hybrid_ratio': float(kv.get('hybrid_ratio', '0.8')),
                'temp': float(kv.get('temp', '1.0')),
                'best_test_macro': float(kv['best_test_macro']),
                'best_test_micro': float(kv['best_test_micro']),
                'final_test_macro': float(kv['final_test_macro']),
                'final_test_micro': float(kv['final_test_micro']),
                'best_epoch': int(kv['best_epoch']),
                'total_training_time_sec': float(kv['total_training_time_sec']),
            })

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    fieldnames = [
        'strategy', 'tag', 'seed', 'neighbor_strategy', 'hybrid_ratio', 'temp',
        'best_test_macro', 'best_test_micro', 'final_test_macro', 'final_test_micro',
        'best_epoch', 'total_training_time_sec',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    std_ddof = 1
    agg = []
    for strat_name, tag, _ in CONFIGS:
        sel = [r for r in rows if r['tag'] == tag]
        bm = np.array([r['best_test_macro'] for r in sel], dtype=np.float64)
        bmi = np.array([r['best_test_micro'] for r in sel], dtype=np.float64)
        agg.append({
            'strategy': strat_name,
            'tag': tag,
            'macro_mean': float(np.mean(bm)),
            'macro_std': float(np.std(bm, ddof=std_ddof)),
            'micro_mean': float(np.mean(bmi)),
            'micro_std': float(np.std(bmi, ddof=std_ddof)),
        })

    full = agg[0]
    freq_macro = full['macro_mean']
    freq_micro = full['micro_mean']
    rnd = agg[1]

    summary_txt_path = os.path.join(OUT_DIR, 'summary.txt')
    lines = [
        'PubMed Node Classification — Neighbor strategy sweep (seeds: {})'.format(SEEDS),
        'Baseline “Full” = freq (most_common K). Δ = strategy_mean - freq_mean.',
        '',
        '| Strategy | Macro-F1 | Micro-F1 | ΔMacro vs Full | ΔMicro vs Full |',
        '|----------|----------|----------|----------------|----------------|',
    ]
    for a in agg:
        dm = a['macro_mean'] - freq_macro
        di = a['micro_mean'] - freq_micro
        lines.append('| {} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:+.4f} | {:+.4f} |'.format(
            a['strategy'], a['macro_mean'], a['macro_std'],
            a['micro_mean'], a['micro_std'], dm, di))
    lines.append('')

    best_a = max(agg, key=lambda x: x['macro_mean'])
    lines.extend([
        '--- Automatic notes ---',
        'Best mean Macro-F1: {} (Macro {:.4f}, Micro {:.4f})'.format(
            best_a['strategy'], best_a['macro_mean'], best_a['micro_mean']),
        'Beats Full (freq) on Macro: {}'.format('yes' if best_a['macro_mean'] > freq_macro else 'no'),
        'Beats Random on Macro: {}'.format('yes' if best_a['macro_mean'] > rnd['macro_mean'] else 'no'),
        '',
        'Hybrid stability (Macro-F1 std across seeds, ddof=1):',
    ])
    for a in agg:
        if a['tag'].startswith('hybrid'):
            lines.append('  {} : std={:.4f}'.format(a['strategy'], a['macro_std']))
    temp_aggs = [a for a in agg if a['tag'].startswith('temp')]
    if temp_aggs:
        best_t = max(temp_aggs, key=lambda x: x['macro_mean'])
        lines.append('')
        lines.append('Best temperature setting (by mean Macro-F1): {}'.format(best_t['strategy']))
    lines.append('')

    text = '\n'.join(lines) + '\n'
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print('\n' + text)
    print('CSV:', csv_path)


if __name__ == '__main__':
    run_all()
