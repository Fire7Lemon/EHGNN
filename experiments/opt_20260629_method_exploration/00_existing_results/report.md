# P0：已有结果与瓶颈分析

> 生成时间：2026-06-03  
> 解析脚本：`code/p0_scan_and_parse.py`

## 1. 已扫描路径

| 路径 | 存在 | 说明 |
|------|------|------|
| `server_results/` | 是 | 含 `2026-05-16_core_reproduction/`、`2026-06-02_/` |
| `server_results/2026-06-02_/Link Prediction/results/dblp_lp_5seeds_val1000/logs/` | 是 | **DBLP LP 正式 5-seed 来源** |
| `server_results/2026-06-02_/Link Prediction/results/pubmed_lp_5seeds/` | 是 | 5 seed + summary.csv |
| `server_results/2026-06-02_/Node Classification/results/{pubmed,dblp,yelp}_5seeds/` | 是 | NC 5-seed + summary |
| `server_results/2026-05-16_core_reproduction/` | 是 | 早期批次；DBLP LP 仅不完整 seed42 |
| `Node Classification/results/` | 是（空或无正式文件） | 本地无正式 5-seed |
| `Link Prediction/results/` | 是（空或无正式文件） | 本地无正式 5-seed |
| `MAG240M/results/` | **未在本地找到** | — |

完整文件清单见 `results/scan_inventory.json`。

## 2. 已发现的正式结果（2026-06-02 归档批次）

| 任务 | 5-seed 完整 | summary | 备注 |
|------|-------------|---------|------|
| PubMed NC | 是 | `pubmed_5seeds_summary.csv` | 训练 ~3–4 s/seed |
| DBLP NC | 是 | `summary.csv` | macro F1 较低（~0.13–0.21 best） |
| Yelp NC | 是 | `summary.csv` | 5 seed 均 complete |
| PubMed LP | 是 | `summary.csv` | Final AUC ~0.53 |
| **DBLP LP (val1000)** | **是（5/5）** | **无 summary.csv** | 本轮 P0 从日志解析 |
| Yelp LP | **未找到** | — | 两批次归档均无 |
| MAG240M | **未找到** | — | — |

## 3. DBLP LP 5-seed 是否完整

**结论：运行完整（5/5 seed）。**

判定标准（每个 seed）：
- 有 `Final Epoch : 99`
- 有 `Final Test AUC`
- 有 `Total training time`
- 无 OOM / Traceback / CUDA error

配置：`run_dblp_lp_5seeds_val1000.py` — `--val_epochs 1000 --log_interval 100 --skip_batch_metrics`，100 epochs。

**注意**：运行完整 ≠ 性能成功复现。Final AUC 均值 **0.5008 ± 0.0039**，接近随机（0.5）。

## 4. DBLP LP 每个 seed 结果

| seed | best_auc | best_ap | best_ep | final_auc | final_ap | total_h | load_s | sim_s | mean_eval_s |
|------|----------|---------|---------|-----------|----------|---------|--------|-------|-------------|
| 42 | 0.5117 | 0.5216 | 0 | 0.4989 | 0.5019 | 67.31 | 552 | 1839 | 477 |
| 3407 | 0.5109 | 0.5175 | 0 | 0.4988 | 0.5019 | 54.54 | 536 | 1528 | 424 |
| 2026 | 0.5123 | 0.5187 | 2 | 0.4995 | 0.5022 | 54.83 | 550 | 1847 | 432 |
| 6666 | 0.5205 | 0.5130 | 20 | 0.4989 | 0.5019 | 46.03 | 669 | 1376 | 376 |
| 8888 | 0.5350 | 0.5249 | 20 | 0.5078 | 0.5074 | 39.72 | 542 | 1135 | 348 |

日志路径：`server_results/2026-06-02_/Link Prediction/results/dblp_lp_5seeds_val1000/logs/dblp_lp_seed{SEED}_val1000_log100_skipmetric_e100.log`

## 5. DBLP LP mean ± std（5 complete seeds）

| 指标 | mean | std |
|------|------|-----|
| best_auc | 0.5181 | 0.0102 |
| best_ap | 0.5191 | 0.0045 |
| final_auc | 0.5008 | 0.0039 |
| final_ap | 0.5031 | 0.0024 |
| total_training_time_h | 52.49 | 10.42 |
| load_data_s | 569.7 | 56.1 |
| sim_s | 1544.7 | 305.9 |
| mean_eval_s | 411.4 | 50.1 |

详见 `results/dblp_lp_5seed_stats.csv`。

## 6. DBLP LP 主要瓶颈

1. **训练 step 极多**：`num_sample = g.num_nodes('author')` ≈ 1.77M → ~1767 step/epoch × 100 epoch ≈ **17.7 万 step**。
2. **单次 evaluate_lp 开销大**：mean eval **~348–477 s**（val_epochs=1000 时约每 epoch 1 次）；100 epoch ≈ **11–13 h** 纯评估（占总量 ~20–25%）。
3. **随机游走预计算**：`Done my sim` 均值 **~1545 s（~26 min）/seed**，占总时间 <1%，不是主瓶颈。
4. **数据加载**：~570 s/seed，可忽略相对总时长。
5. **主瓶颈 = 训练 loop 本身**（~46–67 h/seed）：每 step ~1.3 s × 大量 step。

工程优化（val1000、log100、skip_batch_metrics）使 100 epoch **可跑完**，但未解决 **AUC≈0.5** 的方法/协议问题。

## 7. 不能作为正式结果的日志

| 日志 | 原因 |
|------|------|
| `dblp_lp_diag_seed42_val1000_e2.log` | diagnostic，2 epoch |
| `dblp_lp_diag_seed42_val1000_log100_skipmetric_e2.log` | diagnostic，2 epoch |
| `server_results/2026-05-16_.../dblp_lp_seed_42.log` | SIGTERM / 未完成（return_code=-15） |
| `Link Prediction/results/dblp_lp_smoke/`（若存在） | smoke / MemoryError |
| 任何含 Traceback / OOM 的日志 | 失败运行 |

## 8. 对今日方法优化的启发

1. **DBLP LP 不适合作为今日多方法快速迭代载体**（单 seed ~40–67 h）；应用 **PubMed LP** 做 P2/P3 原型。
2. **Final AUC≈0.5** 表明当前 LP 解码 + 全 author 逐步训练协议在大图上可能失效 → **P2 Pair Decoder**、**P3 Sampled-LP** 有明确动机。
3. **NC 侧 PubMed 成本低（~4 s）** → **P1 SeHGNN-lite**、**P4 蒸馏** 适合 PubMed NC。
4. **预计算 sim ~26 min** 可接受；瓶颈在训练与 eval → 蒸馏/轻量头主要减 **训练期** 负担。
5. **DBLP NC macro F1 偏低** 是独立问题；今日不展开。

## 9. 后续方法探索建议

| 下一步 | 建议 |
|--------|------|
| P1 | PubMed NC seed=42；导出 EHGNN 聚合特征 + MLP 头 |
| P2 | PubMed LP；pair MLP 解码替换 dot product |
| P3 | PubMed LP；edge mini-batch 训练（标注 Quick Validation） |
| P4 | 用 PubMed NC EHGNN 作 teacher 蒸馏 MLP |
| P5 | PubMed 子图上 PPR Top-K demo，对比 RW 频率 |
| 服务器 | 全部输出到 `experiments/opt_20260629_method_exploration/<method>/` |

## 产出文件

- `results/dblp_lp_5seed_summary.csv`
- `results/dblp_lp_5seed_stats.csv`
- `results/dblp_lp_runtime_breakdown.csv`
- `results/core_reproduction_summary.json`
- `figs/dblp_lp_*.png`（4 张）
