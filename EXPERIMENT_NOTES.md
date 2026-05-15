# 实验记录笔记（PubMed / DBLP 等节点分类）

**分支：`reproduce-baseline`。** 本节描述的 **`main.py` 默认行为**为 RW 后 **`Counter.most_common(K)` 频次 Top-K**；**不包含** `neighbor_strategy` / hybrid / temp 等 CLI。**邻居策略相关脚本与代码均在分支 `neighbor-strategy-dev`。**

以下内容摘自本机 **`Node Classification/results/`** 下已有汇总文件（2026-05 左右及后续补充）。**仅代表各节所列数据集、默认或 README 对齐设定与所列种子切片**；不向其它数据集或论文表格做强泛化推断。

---

## 1. PubMed「标准」复现口径

与仓库根目录 **`README.md`** 中 PubMed 一行一致的超参：`α=0.7, K=20, lr=1e-3, dropout=0.4, hidden=256, layers=4, batch_size=3000`。

单次运行可通过：

```bash
cd Node Classification
python main.py --dataset PubMed --seed 42
```

说明：`results/pubmed_nc_result.txt` 保存**最后一次** PubMed `main.py` 运行的快照（会被下一次 PubMed 运行覆盖）；字段含 `seed`、best/final F1、训练时间与若干固定超参行（无邻居策略开关）。

---

## 2. 五种子结果（`pubmed_5seeds`）

来源：`results/pubmed_5seeds/pubmed_5seeds_summary.txt`。

| seed | best_macro | best_micro | final_macro | final_micro |
|------|------------|------------|-------------|-------------|
| 42 | 0.593386 | 0.604651 | 0.585391 | 0.593023 |
| 3407 | 0.590784 | 0.604651 | 0.576410 | 0.593023 |
| 2026 | 0.609246 | 0.627907 | 0.571351 | 0.593023 |
| 6666 | 0.646022 | 0.662791 | 0.575014 | 0.604651 |
| 8888 | 0.660291 | 0.686047 | 0.642208 | 0.674419 |

聚合（n=5，`best_test_macro`）：**mean ± std = 0.619946 ± 0.031536**（std ddof=1）；micro：**0.637209 ± 0.036215**。

---

## 3. DBLP Node Classification baseline（selected 3 seeds，初步复现）

**分支：`reproduce-baseline`**；**不包含** `neighbor_strategy` / hybrid / temp 等优化 CLI。

来源：`results/dblp_5seeds/summary.txt`。**本次汇总的 seeds = [42, 3407, 2026]**（n=3，std ddof=1）。这是 **selected 3-seed** 下的初步复现记录，**不是** 5-seed；后续时间允许可用 `run_dblp_5seeds.py` 补跑至默认 5 种子并更新汇总。

**工程观察**：DBLP 在 **`load_dblp` 与 meta-path RW 相似度预计算**阶段耗时相对 PubMed **明显更长**；下表训练时间仅对应 `main.py` 文末 **`Total training time`** 字段的均值（脚本 `summary.txt` 中 `avg training time`），**不包含**数据加载与 RW 预计算时间。

| 指标 | mean ± std |
|------|------------|
| best_test_macro | 0.150100 ± 0.010967 |
| best_test_micro | 0.387500 ± 0.026121 |
| final_test_macro | 0.151867 ± 0.017625 |
| final_test_micro | 0.396000 ± 0.035596 |

- 平均单次训练时间（上述字段）：**10.1355 s**  
- **best seed（按 best_test_macro）**：2026  
- **worst seed（按 best_test_macro）**：3407  

---

## 4. Yelp Node Classification baseline（selected 3 seeds）

**分支：`reproduce-baseline`**；**不包含** `neighbor_strategy` / hybrid / temp；入口为 **`main_yelp.py`**（多标签：**sigmoid + BCELoss**）。

来源：`results/yelp_3seeds/summary.txt`。**汇总的 seeds = [42, 3407, 2026]**（n=3，std ddof=1），为 **selected 3-seed**，**不是**更大规模种子扫描的结论。

| 指标 | mean ± std |
|------|------------|
| best_test_macro | 0.666300 ± 0.000000 |
| best_test_micro | 0.875800 ± 0.000000 |
| final_test_macro | 0.666300 ± 0.000000 |
| final_test_micro | 0.875800 ± 0.000000 |

- 平均单次训练时间（脚本 `avg training time`）：**964.1782 s**  
- **best seed / worst seed（按 best_test_macro）**：汇总上均为 **42**（三 seed 的 test 宏微 F1 **逐数相同**，故脚本中 best/worst 重合）。  

**任务与度量（事实）**：`utils.accuracy(..., dataset='Yelp')` 对 **sigmoid 输出先以 0.5 阈值二值化**，再按样本计算 sklearn F1 后平均（见 **`Node Classification/utils.py`**）。

**异常现象（记录，不夸大）**：三份 seed 下 **训练过程（如 loss / 训练段 macro/micro）不同**（例如各 run 的 `train_time_s` 不同），但日志中 **测试集 Macro-F1 / Micro-F1 在所有记录点均为 0.6663 / 0.8758 且跨 seed 完全一致**。  
**可能原因（仅推断，未在本文档中当作已证结论）**：更可能来自 **测试侧二值预测在 0.5 阈值后未发生变化**、或 **指标对概率微小变化不敏感**；**不应**把 **std=0** 直接解读为「Yelp 模型强稳定」或「复现极其可靠」。  

**后续可查（建议）**：检查测试集 **sigmoid 分布**、**阈值敏感性**、以及是否需要与论文对齐的 **多标签整体评估方式**（不改代码的讨论项）。

---

## 5. 消融实验（3 seeds: 42, 3407, 2026）

来源：`results/pubmed_ablation/summary.txt`。

| Method | Macro-F1 | Micro-F1 |
|--------|----------|----------|
| EHGNN（full） | 0.6032 ± 0.0107 | 0.6202 ± 0.0134 |
| w/o L2 | 0.5920 ± 0.0103 | 0.6085 ± 0.0067 |
| w/o MWeight | 0.5558 ± 0.0301 | 0.5775 ± 0.0242 |
| w/o TWeight | 0.5753 ± 0.0313 | 0.5969 ± 0.0178 |
| Random Neighbor | 0.6227 ± 0.0099 | 0.6395 ± 0.0116 |

相对 full，`Random Neighbor`（`--r_neighbor`）在汇总脚本定义的 Δ 意义上 **Macro/Micro 更高**（汇总文件中说明：Random 改变的是 RW 邻居抽样，不应与删权重消融混为一谈）。

---

## 6. hybrid / temp / neighbor_strategy 扫描（优化分支归档）

**不在 `reproduce-baseline`：** 下列批量脚本与 CLI 仅存在于 **`neighbor-strategy-dev`**（例如 `run_pubmed_neighbor_strategy.py`、`run_pubmed_hybrid_ratio_sweep.py`）。本分支代码已移除对应参数。

以下为曾在优化分支下生成的 **`results/pubmed_neighbor_strategy/summary.txt`** 摘录，**仅作历史对照**，不代表当前分支可复跑：

| Strategy | Macro-F1 | Micro-F1 | ΔMacro vs freq | ΔMicro vs freq |
|----------|----------|----------|----------------|----------------|
| freq | 0.6019 ± 0.0067 | 0.6202 ± 0.0067 | 0.0000 | 0.0000 |
| random | 0.6107 ± 0.0099 | 0.6279 ± 0.0116 | +0.0089 | +0.0078 |
| hybrid (0.8) | 0.5981 ± 0.0043 | 0.6202 ± 0.0067 | −0.0037 | 0.0000 |
| hybrid (0.5) | 0.6053 ± 0.0117 | 0.6202 ± 0.0134 | +0.0034 | 0.0000 |
| temp (0.5) | 0.5995 ± 0.0083 | 0.6163 ± 0.0116 | −0.0024 | −0.0039 |
| temp (1.0) | 0.6016 ± 0.0097 | 0.6163 ± 0.0116 | −0.0002 | −0.0039 |
| temp (2.0) | 0.6010 ± 0.0073 | 0.6163 ± 0.0116 | −0.0008 | −0.0039 |

---

## 7. 与 reproduce-baseline 对齐的观察

- **论文复现默认**：频次 Top-K（无双参数策略扫描）。  
- **`--r_neighbor`**：消融脚本「Random Neighbor」条目仍在 `run_pubmed_ablation.py` 中使用，含义不变。  
- 上表及其他优化分支结论 **不得** 当作本分支 `main.py` 的默认行为说明。

---

## 8. Link Prediction（PubMed）：本地 smoke test 与服务器完整复现

- **本地（笔记本类环境）**：**只做 smoke**，验证数据加载、meta-path RW 预计算、训练与评测日志链路；**不建议**在此跑 **`README.md`** 默认 **100 epoch** 的完整 PubMed LP（CPU/RW 与训练耗时过长）。典型资源约束示例：**约 16GB RAM**、长时间 CPU 计算。
- **Smoke 脚本**：在 **`Link Prediction/`** 下运行 **`python run_pubmed_lp_smoke.py`**，README PubMed LP 超参 + **`--epochs 3`**、**`--val_epochs 1`**，日志：`results/pubmed_lp_smoke/pubmed_lp_smoke_seed_42.log`。日志中 `precision` 实际对应 **AP（Average Precision，平均精确率）**（见 **`Link Prediction/utils.py`**）。
- **服务器**：完整 **3-seed**（或后续扩展 **5-seed**）与 **`README.md`** 对齐 epoch 的 PubMed LP，计划在 **GPU/高内存** 机器上使用 **`run_pubmed_lp_3seeds.py`**（或等价显式 **`main.py --epochs 100`**）执行。
- **MAG240M**：大规模节点分类仍按 **`README_MAG240M.md`** 在 **Linux 服务器** 部署与训练（本地不占位跑全量）。

---

## 9. 后续工作（建议）

| 方向 | 说明 |
|------|------|
| DBLP / Yelp / MAG240M | 在本分支口径下复现或对齐论文设置 |
| 邻居策略研究 | 切换到 **`neighbor-strategy-dev`** 分支阅读脚本与历史 `results/` |
| 统计严谨性 | PubMed 测试集较小，可多 seed、报告置信区间或与论文表对齐的设定 |
