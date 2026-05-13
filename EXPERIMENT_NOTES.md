# Experiment Notes（PubMed Node Classification）

以下内容摘自本机 **`Node Classification/results/`** 下已有汇总文件（2026-05 左右生成）。**仅代表 PubMed、当前默认训练设定与所列种子切片**；不向其它数据集或论文表格做强泛化推断。

---

## 1. PubMed「标准」复现口径

与仓库根目录 **`README.md`** 中 PubMed 一行一致的超参：`α=0.7, K=20, lr=1e-3, dropout=0.4, hidden=256, layers=4, batch_size=3000`。

单次运行可通过：

```bash
cd Node Classification
python main.py --dataset PubMed --seed 42
```

说明：`results/pubmed_nc_result.txt` 保存**最后一次** PubMed 运行的快照，可能与 seed 42 或 README 默认不完全一致（例如邻居策略实验最后一次运行会覆盖该文件）。

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

相对 full，`Random Neighbor` 在汇总脚本定义的 Δ 意义上 **Macro/Micro 更高**（汇总文件中说明：Random 改变的是 RW 邻居抽样，不应与删权重消融混为一谈）。

---

## 4. Neighbor strategy sweep（3 seeds: 42, 3407, 2026）

来源：`results/pubmed_neighbor_strategy/summary.txt`。

| Strategy | Macro-F1 | Micro-F1 | ΔMacro vs freq | ΔMicro vs freq |
|----------|----------|----------|----------------|----------------|
| freq | 0.6019 ± 0.0067 | 0.6202 ± 0.0067 | 0.0000 | 0.0000 |
| random | 0.6107 ± 0.0099 | 0.6279 ± 0.0116 | +0.0089 | +0.0078 |
| hybrid (0.8) | 0.5981 ± 0.0043 | 0.6202 ± 0.0067 | −0.0037 | 0.0000 |
| hybrid (0.5) | 0.6053 ± 0.0117 | 0.6202 ± 0.0134 | +0.0034 | 0.0000 |
| temp (0.5) | 0.5995 ± 0.0083 | 0.6163 ± 0.0116 | −0.0024 | −0.0039 |
| temp (1.0) | 0.6016 ± 0.0097 | 0.6163 ± 0.0116 | −0.0002 | −0.0039 |
| temp (2.0) | 0.6010 ± 0.0073 | 0.6163 ± 0.0116 | −0.0008 | −0.0039 |

脚本自动摘要：**mean Macro-F1 最优为 random**；在此 sweep 设置下 hybrid/temp 未整体超过 random。

Hybrid 跨 seed 的 Macro std：**0.8 → 0.0043**，**0.5 → 0.0117**（越小通常表示跨 seed 更稳，但仍需结合均值解读）。

---

## 5. 当前观察（PubMed）

- **Random Neighbor** 在消融切片与 **neighbor strategy sweep** 中，**平均 Macro-F1 优于 freq（most_common Top-K）**。  
- **初步解释（假设性）**：纯频次 Top-K 容易过度依赖随机游走中出现次数极高的邻居（hub / 高频共现），削弱多样性；随机或放宽选择的邻居集合可能改善归纳偏置。**需在更大验证集与其它数据集上检验**。  
- **不得过度外推**：上述结论目前主要来自 **PubMed 节点分类、固定训练流程与小规模测试指标**；DBLP / Yelp / MAG240M 未在同一套邻居策略实验下完整汇报。

---

## 6. 后续优化方向（建议）

| 方向 | 说明 |
|------|------|
| `hybrid_ratio` 细扫 | 当前仅 0.5 / 0.8；可在验证集上网格或贝叶斯搜索。 |
| DBLP / Yelp | 验证 Random / hybrid / temp 是否仍优于 freq。 |
| MAG240M | 先做 **`scripts/check_mag240m_data.py`** 或小批量 dry-run；全量训练.resource 受限时单独规划。 |
| 统计严谨性 | PubMed 测试集较小，可多 seed、报告置信区间或与论文表对齐的设定。 |
