# TODO：HPPR strategy 研究（Table X）

**分支：`reproduce-baseline`。**  
Table X 描述的是 **PubMed Node Classification** 上不同 **HPPR / 相似度构造策略** 的对比：**AvgSim**、**HeteSim**、**Ours** 等。

---

## 1. Table X 论文要求（理解层面）

- **任务**：PubMed **Node Classification**
- **对比项**：AvgSim、HeteSim、本文方法（Ours）
- **指标**：Macro-F1、Micro-F1（与仓库 NC 评测一致为佳）

---

## 2. 当前代码是否已有 AvgSim / HeteSim

- **`Node Classification/utils.py`** 中与 meta-path 相关的主要路径为 **`random_walk_sim`** 及基于随机游走统计的相似度构造；**未发现** 命名为 AvgSim、HeteSim 的独立实现或与论文 Table X 逐项对齐的开关。
- **结论**：**AvgSim / HeteSim：Not Implemented**（在本仓库 `reproduce-baseline` 入口与 utils 可见范围内）。

---

## 3. 与「random / hybrid / temp」邻居策略探索的区别

- 仓库历史中 **`neighbor_strategy-dev`** 等分支下的 random / hybrid / temp 属于 **额外的邻居采样策略实验**，**不等价**于论文 Table X 的 **AvgSim / HeteSim / Ours（HPPR strategy）** 对比。
- **明确**：不要用 random/hybrid/temp 的结果冒充 Table X 复现。

---

## 4. 后续若要完整复现 Table X

1. **查阅论文正文 / 附录** 对 AvgSim、HeteSim、Ours 的精确定义与伪代码。
2. 在 **`Node Classification/utils.py`**（或单独模块）实现 **AvgSim**、**HeteSim** 的相似度矩阵构造，接口尽量与现有 `random_walk_sim` 输出形状兼容。
3. 在 **`main.py`** 增加 **最小 CLI**（例如 `--similarity_strategy {rw,avgsim,hetesim}`），仅在构图阶段分支，**不改变** EHGNN MLP 结构与 loss。
4. 新增 **`run_pubmed_nc_hppr_strategy_tableX.py`**（名称可调整）：固定 README PubMed NC 超参，多 seed，输出 `summary.csv` / `summary.txt`。

---

## 5. 本轮结论

- Table X **当前标记为 TODO / Not Implemented**（缺少 AvgSim、HeteSim 实现与入口开关）。
- 本轮 **未** 修改模型、loss、数据。
