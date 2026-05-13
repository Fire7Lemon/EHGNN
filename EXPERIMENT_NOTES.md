# 实验记录笔记（PubMed 节点分类）

**分支：`reproduce-baseline`。** 本节描述的 **`main.py` 默认行为**为 RW 后 **`Counter.most_common(K)` 频次 Top-K**；**不包含** `neighbor_strategy` / hybrid / temp 等 CLI。**邻居策略相关脚本与代码均在分支 `neighbor-strategy-dev`。**

以下内容摘自本机 **`Node Classification/results/`** 下已有汇总文件（2026-05 左右生成）。**仅代表 PubMed、当前默认训练设定与所列种子切片**；不向其它数据集或论文表格做强泛化推断。

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

## 3. 消融实验（3 seeds: 42, 3407, 2026）

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

## 4. hybrid / temp / neighbor_strategy 扫描（优化分支归档）

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

## 5. 与 reproduce-baseline 对齐的观察

- **论文复现默认**：频次 Top-K（无双参数策略扫描）。  
- **`--r_neighbor`**：消融脚本「Random Neighbor」条目仍在 `run_pubmed_ablation.py` 中使用，含义不变。  
- 上表及其他优化分支结论 **不得** 当作本分支 `main.py` 的默认行为说明。

---

## 6. 后续工作（建议）

| 方向 | 说明 |
|------|------|
| DBLP / Yelp / MAG240M | 在本分支口径下复现或对齐论文设置 |
| 邻居策略研究 | 切换到 **`neighbor-strategy-dev`** 分支阅读脚本与历史 `results/` |
| 统计严谨性 | PubMed 测试集较小，可多 seed、报告置信区间或与论文表对齐的设定 |
