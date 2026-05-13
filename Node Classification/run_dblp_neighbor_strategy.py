"""DBLP 邻居策略快速对照：freq / random / hybrid ratio=0.9 × 多种子。

仅调用现有 main.py CLI；结果从 stdout 解析，不写 PubMed 专用 txt。
输出：results/dblp_neighbor_strategy/ 下逐次 .log、summary.csv、summary.txt。
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(SCRIPT_DIR, 'main.py')
DATA_DBLP = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'DBLP'))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'dblp_neighbor_strategy')

SEEDS = [42, 3407, 2026]

# (表格 Method 名, 日志/文件名 tag, main.py 额外参数)
CONFIGS = [
    ('freq', 'freq', ['--neighbor_strategy', 'freq']),
    ('random', 'random', ['--neighbor_strategy', 'random']),
    ('hybrid_0.9', 'hybrid_0.9', ['--neighbor_strategy', 'hybrid', '--hybrid_ratio', '0.9']),
]

# main.py 打印格式（需与 Node Classification/main.py 一致）
_RE_BEST = re.compile(
    r'Best Test Macro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Micro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Epoch\s*:\s*([-0-9]+)',
)
_RE_FINAL = re.compile(
    r'Final Epoch\s*:\s*([-0-9]+)\s*,\s*Final Test Macro-F1\s*:\s*([0-9.+-eE]+)\s*,\s*Micro-F1\s*:\s*([0-9.+-eE]+)',
)
_RE_TIME = re.compile(
    r'Total training time:\s*([0-9.+-eE]+)\s*s',
)


def _preflight_or_exit():
    if not os.path.isfile(MAIN_PY):
        print('ERROR: main.py not found at {}'.format(MAIN_PY), file=sys.stderr, flush=True)
        sys.exit(1)
    if not os.path.isdir(DATA_DBLP):
        print(
            'ERROR: DBLP data directory not found: {}'.format(DATA_DBLP),
            file=sys.stderr,
            flush=True,
        )
        print(
            'Please download/placement DBLP under EHGNN/data/DBLP (see README.md).',
            file=sys.stderr,
            flush=True,
        )
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


def _fail(proc, cmd):
    print('[FAIL] return_code={}'.format(proc.returncode), file=sys.stderr, flush=True)
    print('[FAIL] cmd:', ' '.join(cmd), file=sys.stderr, flush=True)
    print('[FAIL] --- stdout ---', file=sys.stderr, flush=True)
    print(proc.stdout or '(empty)', file=sys.stderr, flush=True)
    print('[FAIL] --- stderr ---', file=sys.stderr, flush=True)
    print(proc.stderr or '(empty)', file=sys.stderr, flush=True)
    sys.exit(proc.returncode if proc.returncode != 0 else 1)


def parse_main_stdout(text, log_path_for_error):
    """从单次运行的合并 stdout 解析指标；失败则打印路径与片段并退出。"""
    out = text or ''
    mb = _RE_BEST.search(out)
    mf = _RE_FINAL.search(out)
    mt = _RE_TIME.search(out)
    if not mb or not mf or not mt:
        snippet = out[-4000:] if len(out) > 4000 else out
        print(
            'ERROR: failed to parse metrics from stdout. log={}'.format(log_path_for_error),
            file=sys.stderr,
            flush=True,
        )
        print('--- stdout (tail or full) ---', file=sys.stderr, flush=True)
        print(snippet, file=sys.stderr, flush=True)
        sys.exit(1)

    best_macro = float(mb.group(1))
    best_micro = float(mb.group(2))
    best_epoch = int(mb.group(3))
    final_epoch = int(mf.group(1))
    final_macro = float(mf.group(2))
    final_micro = float(mf.group(3))
    total_sec = float(mt.group(1))

    return {
        'best_test_macro': best_macro,
        'best_test_micro': best_micro,
        'best_epoch': best_epoch,
        'final_epoch': final_epoch,
        'final_test_macro': final_macro,
        'final_test_micro': final_micro,
        'total_training_time_sec': total_sec,
    }


def main():
    _preflight_or_exit()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    base_cmd = [
        sys.executable,
        MAIN_PY,
        '--dataset', 'DBLP',
        '--path', '../data/',
    ]

    for method_name, tag, extra in CONFIGS:
        for seed in SEEDS:
            cmd = base_cmd + ['--seed', str(seed)] + extra
            log_path = os.path.join(OUT_DIR, '{}_seed_{}.log'.format(tag, seed))

            print('[RUN] method={} seed={}'.format(method_name, seed), flush=True)
            print('[RUN] cmd={}'.format(' '.join(cmd)), flush=True)

            proc = _run_main_capture(cmd)
            with open(log_path, 'w', encoding='utf-8') as lf:
                lf.write(proc.stdout or '')
                if proc.stderr:
                    lf.write('\n--- stderr ---\n')
                    lf.write(proc.stderr)

            if proc.returncode != 0:
                _fail(proc, cmd)

            merged_for_parse = (proc.stdout or '') + '\n' + (proc.stderr or '')
            metrics = parse_main_stdout(merged_for_parse, log_path)

            rows.append({
                'method': method_name,
                'seed': seed,
                **metrics,
            })

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    keys = [
        'method', 'seed', 'best_test_macro', 'best_test_micro', 'best_epoch',
        'final_test_macro', 'final_test_micro', 'final_epoch', 'total_training_time_sec',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    std_ddof = 1
    agg = []
    for method_name, tag, _ in CONFIGS:
        sel = [r for r in rows if r['method'] == method_name]
        bm = np.array([r['best_test_macro'] for r in sel], dtype=np.float64)
        bmi = np.array([r['best_test_micro'] for r in sel], dtype=np.float64)
        agg.append({
            'method': method_name,
            'macro_mean': float(np.mean(bm)),
            'macro_std': float(np.std(bm, ddof=std_ddof)),
            'micro_mean': float(np.mean(bmi)),
            'micro_std': float(np.std(bmi, ddof=std_ddof)),
        })

    freq = next(a for a in agg if a['method'] == 'freq')
    rnd = next(a for a in agg if a['method'] == 'random')
    hyb = next(a for a in agg if a['method'] == 'hybrid_0.9')

    agg_sorted = sorted(agg, key=lambda x: x['macro_mean'], reverse=True)
    best_method = agg_sorted[0]['method']

    lines = [
        'DBLP neighbor strategy (freq / random / hybrid_0.9), seeds: {}'.format(SEEDS),
        'Baseline for Δ: freq.',
        '',
        '| Method | Macro-F1 | Micro-F1 | ΔMacro vs freq | ΔMicro vs freq |',
        '|--------|----------|----------|----------------|----------------|',
    ]
    for a in agg_sorted:
        dm = a['macro_mean'] - freq['macro_mean']
        di = a['micro_mean'] - freq['micro_mean']
        lines.append('| {} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:+.4f} | {:+.4f} |'.format(
            a['method'],
            a['macro_mean'], a['macro_std'],
            a['micro_mean'], a['micro_std'],
            dm, di,
        ))
    lines.append('')

    rnd_beats_freq_macro = rnd['macro_mean'] > freq['macro_mean']
    hyb_beats_freq_macro = hyb['macro_mean'] > freq['macro_mean']
    hyb_beats_rnd_macro = hyb['macro_mean'] > rnd['macro_mean']

    # PubMed 参考（用户提供）：Macro 上 freq < hybrid_0.9 < random
    pubmed_macro_order = 'freq < hybrid(0.9) < random（Macro 均值）'
    dblp_macro_order = (
        freq['macro_mean'], hyb['macro_mean'], rnd['macro_mean'])
    dblp_strict_same_shape = (
        dblp_macro_order[0] < dblp_macro_order[1] < dblp_macro_order[2])

    if dblp_strict_same_shape and rnd_beats_freq_macro:
        pubmed_note = (
            'DBLP 上 Macro 均值严格满足 freq < hybrid_0.9 < random，与 PubMed 报告的趋势一致。')
    elif rnd_beats_freq_macro and hyb_beats_freq_macro:
        pubmed_note = (
            'DBLP 上 random 与 hybrid_0.9 的 Macro 均高于 freq，与「random/hybrid 相对 freq 抬升」方向一致，'
            '但三者排序未必与 PubMed 完全相同。')
    elif rnd_beats_freq_macro:
        pubmed_note = (
            'DBLP 仅体现 random > freq（Macro），hybrid_0.9 相对 freq 未更高；与 PubMed 三者关系不完全一致。')
    else:
        pubmed_note = (
            'DBLP 未观察到 random Macro 明显高于 freq；与 PubMed 现象不一致，需谨慎外推。')

    yelp_note = (
        '建议在 Yelp（main_yelp.py）上再做小规模对照：若 DBLP 与 PubMed 结论分叉，'
        '第三个数据集有助于判断是否为数据集特异；若一致则增强外推信心。')

    lines.extend([
        '--- 结论 ---',
        '1. random 的 Macro 均值是否高于 freq: {}'.format('是' if rnd_beats_freq_macro else '否'),
        '2. hybrid_0.9 的 Macro 均值是否高于 freq: {}'.format('是' if hyb_beats_freq_macro else '否'),
        '3. hybrid_0.9 的 Macro 均值是否高于 random: {}'.format('是' if hyb_beats_rnd_macro else '否'),
        '4. PubMed 现象是否在 DBLP 上复现（定性）: {}'.format(pubmed_note),
        '   （参考 PubMed 口径：{}）'.format(pubmed_macro_order),
        '5. 是否建议继续扩展到 Yelp: {}'.format(yelp_note),
        '',
        '全局按 Macro 均值最优方法: {}'.format(best_method),
        '',
        'Artifacts:',
        '  {}'.format(OUT_DIR),
        '  {}'.format(csv_path),
    ])

    summary_txt = os.path.join(OUT_DIR, 'summary.txt')
    text = '\n'.join(lines) + '\n'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text)

    print('\n' + text)
    print('CSV:', csv_path)


if __name__ == '__main__':
    main()
