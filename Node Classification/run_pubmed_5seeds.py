"""
Run PubMed node classification with multiple seeds and aggregate statistics.
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np

SEEDS = [42, 3407, 2026, 6666, 8888]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'pubmed_5seeds')


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for seed in SEEDS:
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'main.py'), '--seed', str(seed)]
        print('\n>>> {}'.format(' '.join(cmd)), flush=True)
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
        src = os.path.join(RESULTS_ROOT, 'pubmed_nc_result.txt')
        if not os.path.isfile(src):
            raise FileNotFoundError('Missing {} after run seed={}'.format(src, seed))
        dst = os.path.join(OUT_DIR, 'pubmed_seed_{}.txt'.format(seed))
        shutil.copy2(src, dst)
        kv = parse_kv_txt(src)
        row = {
            'seed': int(kv['seed']),
            'best_test_macro': float(kv['best_test_macro']),
            'best_test_micro': float(kv['best_test_micro']),
            'final_test_macro': float(kv['final_test_macro']),
            'final_test_micro': float(kv['final_test_micro']),
            'best_epoch': int(float(kv['best_epoch'])),
            'total_training_time_sec': float(kv['total_training_time_sec']),
        }
        rows.append(row)

    keys = ['seed', 'best_test_macro', 'best_test_micro', 'final_test_macro',
            'final_test_micro', 'best_epoch', 'total_training_time_sec']
    csv_path = os.path.join(OUT_DIR, 'pubmed_5seeds_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def stat_arr(name_key):
        return np.array([r[name_key] for r in rows], dtype=np.float64)

    bm = stat_arr('best_test_macro')
    bmi = stat_arr('best_test_micro')
    fm = stat_arr('final_test_macro')
    fmi = stat_arr('final_test_micro')
    tt = stat_arr('total_training_time_sec')

    std_ddof = 1 if len(rows) > 1 else 0

    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals, ddof=std_ddof)), float(np.max(vals)), float(np.min(vals))

    bm_m, bm_s, bm_mx, bm_mn = ms(bm)
    bmi_m, bmi_s, bmi_mx, bmi_mn = ms(bmi)
    fm_m, fm_s, fm_mx, fm_mn = ms(fm)
    fmi_m, fmi_s, fmi_mx, fmi_mn = ms(fmi)

    seeds_arr = np.array([r['seed'] for r in rows])
    ibest = int(np.argmax(bm))
    iworst = int(np.argmin(bm))

    summary_path = os.path.join(OUT_DIR, 'pubmed_5seeds_summary.txt')
    lines = [
        'PubMed Node Classification — 5-seed runs',
        '',
        'Per-seed results (test set)',
        'seed\tbest_macro\tbest_micro\tfinal_macro\tfinal_micro\tbest_epoch\ttrain_time_s',
    ]
    for r in rows:
        lines.append('{seed}\t{best_test_macro:.6f}\t{best_test_micro:.6f}\t'
                     '{final_test_macro:.6f}\t{final_test_micro:.6f}\t{best_epoch}\t'
                     '{total_training_time_sec:.4f}'.format(**r))
    lines.extend([
        '',
        'Aggregate (n={})'.format(len(rows)),
        'Metric                  mean ± std              max         min',
        'best_test_macro      {:.6f} ± {:.6f}    {:.6f}    {:.6f}'.format(bm_m, bm_s, bm_mx, bm_mn),
        'best_test_micro      {:.6f} ± {:.6f}    {:.6f}    {:.6f}'.format(bmi_m, bmi_s, bmi_mx, bmi_mn),
        'final_test_macro     {:.6f} ± {:.6f}    {:.6f}    {:.6f}'.format(fm_m, fm_s, fm_mx, fm_mn),
        'final_test_micro     {:.6f} ± {:.6f}    {:.6f}    {:.6f}'.format(fmi_m, fmi_s, fmi_mx, fmi_mn),
        '',
        'Best seed (by best_test_macro):   {}  (macro={:.6f})'.format(seeds_arr[ibest], bm[ibest]),
        'Worst seed (by best_test_macro): {}  (macro={:.6f})'.format(seeds_arr[iworst], bm[iworst]),
        'Mean training time (s): {:.4f}'.format(float(np.mean(tt))),
        '',
        'Artifacts:',
        '  ' + OUT_DIR,
        '  ' + csv_path,
        '  ' + summary_path,
    ])
    text = '\n'.join(lines) + '\n'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
