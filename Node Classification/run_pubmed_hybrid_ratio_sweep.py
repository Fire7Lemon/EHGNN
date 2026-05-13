"""PubMed hybrid_ratio 扫描：基线 freq/random + hybrid ratio 0.0~1.0 × 多种子，汇总 summary。"""
import csv
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, 'results')
OUT_DIR = os.path.join(RESULTS_ROOT, 'pubmed_hybrid_ratio_sweep')

SEEDS = [42, 3407, 2026]
HYBRID_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# (method, ratio 展示用, 保存文件名 tag, main.py 额外参数)
CONFIGS = [('freq', '-', 'freq', ['--neighbor_strategy', 'freq'])]
CONFIGS.append(('random', '-', 'random', ['--neighbor_strategy', 'random']))
for r in HYBRID_RATIOS:
    rs = '{:.1f}'.format(r)
    tag = 'hybrid_ratio_{}'.format(rs)
    CONFIGS.append(('hybrid', rs, tag, ['--neighbor_strategy', 'hybrid', '--hybrid_ratio', rs]))


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
    base_cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'main.py'), '--dataset', 'PubMed']

    for method, ratio_disp, tag, extra in CONFIGS:
        for seed in SEEDS:
            cmd = base_cmd + ['--seed', str(seed)] + extra
            print('\n>>> {}'.format(' '.join(cmd)), flush=True)
            proc = subprocess.run(
                cmd,
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            if proc.returncode != 0:
                print('命令失败 (exit {}):'.format(proc.returncode), file=sys.stderr)
                print(' '.join(cmd), file=sys.stderr)
                if proc.stdout:
                    print(proc.stdout, file=sys.stderr)
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                sys.exit(proc.returncode)

            src = os.path.join(RESULTS_ROOT, 'pubmed_nc_result.txt')
            if not os.path.isfile(src):
                print('缺少结果文件: {}'.format(src), file=sys.stderr)
                print('命令: {}'.format(' '.join(cmd)), file=sys.stderr)
                sys.exit(1)

            dst = os.path.join(OUT_DIR, '{}_seed_{}.txt'.format(tag, seed))
            shutil.copy2(src, dst)
            kv = parse_kv_txt(src)
            rows.append({
                'method': method,
                'ratio': ratio_disp,
                'tag': tag,
                'seed': int(kv['seed']),
                'neighbor_strategy': kv.get('neighbor_strategy', ''),
                'hybrid_ratio': float(kv.get('hybrid_ratio', 'nan')),
                'best_test_macro': float(kv['best_test_macro']),
                'best_test_micro': float(kv['best_test_micro']),
                'final_test_macro': float(kv['final_test_macro']),
                'final_test_micro': float(kv['final_test_micro']),
                'best_epoch': int(kv['best_epoch']),
                'total_training_time_sec': float(kv['total_training_time_sec']),
            })

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    fieldnames = [
        'method', 'ratio', 'tag', 'seed', 'neighbor_strategy', 'hybrid_ratio',
        'best_test_macro', 'best_test_micro', 'final_test_macro', 'final_test_micro',
        'best_epoch', 'total_training_time_sec',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    std_ddof = 1
    agg_map = {}
    for method, ratio_disp, tag, _ in CONFIGS:
        key = (method, ratio_disp)
        sel = [r for r in rows if r['tag'] == tag]
        bm = np.array([r['best_test_macro'] for r in sel], dtype=np.float64)
        bmi = np.array([r['best_test_micro'] for r in sel], dtype=np.float64)
        agg_map[key] = {
            'method': method,
            'ratio': ratio_disp,
            'tag': tag,
            'macro_mean': float(np.mean(bm)),
            'macro_std': float(np.std(bm, ddof=std_ddof)),
            'micro_mean': float(np.mean(bmi)),
            'micro_std': float(np.std(bmi, ddof=std_ddof)),
        }

    freq_macro = agg_map[('freq', '-')]['macro_mean']
    freq_micro = agg_map[('freq', '-')]['micro_mean']
    rnd_macro = agg_map[('random', '-')]['macro_mean']
    rnd_micro = agg_map[('random', '-')]['micro_mean']

    table_rows = list(agg_map.values())
    table_rows.sort(key=lambda x: x['macro_mean'], reverse=True)

    hybrid_aggs = [agg_map[('hybrid', '{:.1f}'.format(r))] for r in HYBRID_RATIOS]
    best_hybrid = max(hybrid_aggs, key=lambda x: x['macro_mean'])

    ratios_arr = np.array(HYBRID_RATIOS, dtype=np.float64)
    macros_h = np.array([agg_map[('hybrid', '{:.1f}'.format(r))]['macro_mean'] for r in HYBRID_RATIOS])
    if len(ratios_arr) > 2 and np.std(macros_h) > 1e-12 and np.std(ratios_arr) > 1e-12:
        slope = float(np.polyfit(ratios_arr, macros_h, 1)[0])
    else:
        slope = float('nan')

    summary_txt_path = os.path.join(OUT_DIR, 'summary.txt')
    lines = [
        'PubMed hybrid_ratio sweep (seeds: {})'.format(SEEDS),
        'Baseline freq Macro mean: {:.4f}, Micro mean: {:.4f}'.format(freq_macro, freq_micro),
        '',
        '| Method | Ratio | Macro-F1 | Micro-F1 | ΔMacro vs freq | ΔMicro vs freq |',
        '|--------|-------|----------|----------|----------------|----------------|',
    ]
    for a in table_rows:
        dm = a['macro_mean'] - freq_macro
        di = a['micro_mean'] - freq_micro
        lines.append('| {} | {} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:+.4f} | {:+.4f} |'.format(
            a['method'], a['ratio'], a['macro_mean'], a['macro_std'],
            a['micro_mean'], a['micro_std'], dm, di))
    lines.append('')

    lines.extend([
        '--- 结论 ---',
        '按 mean Macro-F1 排序如上；表中第一行为全局最优配置。',
        '最优 hybrid_ratio（仅在 hybrid 策略内比较）: {}  (Macro {:.4f} ± ..., Micro {:.4f} ± ...)'.format(
            best_hybrid['ratio'], best_hybrid['macro_mean'], best_hybrid['micro_mean']),
        '最优 hybrid 是否超过 freq（Macro 均值）: {}'.format('是' if best_hybrid['macro_mean'] > freq_macro else '否'),
        '最优 hybrid 是否超过 random（Macro 均值）: {}'.format('是' if best_hybrid['macro_mean'] > rnd_macro else '否'),
        'hybrid 扫描段上 Macro 对 ratio 的一阶线性斜率（越大表示随 ratio 升高 Macro 越高）: {:.6f}'.format(slope),
        '是否存在「ratio 越高反而越差」的整体趋势: {}'.format(
            '倾向存在（斜率为负，ratio 升高 Macro 总体下降）' if not np.isnan(slope) and slope < -1e-6
            else ('不明显或斜率接近 0 / 为正' if not np.isnan(slope) else '样本不足无法判断')),
        '',
        '简短解释: hybrid_ratio 控制 Top-K 频次槽与随机槽占比；若最优 ratio 靠近 0 则随机多样性更重要，',
        '靠近 1 则频次邻居更重要。freq/random 两行可与 hybrid 曲线对照；建议在 DBLP 上复测以防 PubMed 特例。',
    ])

    text = '\n'.join(lines) + '\n'
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print('\n' + text)
    print('CSV:', csv_path)


if __name__ == '__main__':
    main()
