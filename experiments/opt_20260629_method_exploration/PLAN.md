# EHGNN 方法级优化探索计划（工作区摘要）

> 完整计划原文：`docs/EHGNN_方法级优化探索计划.md`（已找到，1013 行）

## 今日目标

1. 在 EHGNN 项目内完成多条**方法级优化思路**原型，而非仅做超参搜索。
2. 优先低成本数据集：PubMed NC/LP；利用已有 DBLP LP 日志做瓶颈分析。
3. **不启动** DBLP LP 新完整训练、MAG240M。
4. 每条思路留证据：代码 / 命令 / 日志 / CSV / 图片 / 报告。
5. 全部归档于 `experiments/opt_20260629_method_exploration/`。

## 方法路线

| 优先级 | 方法 | 任务 | 数据集 | 今日目标 |
|--------|------|------|--------|----------|
| P0 | 已有结果与瓶颈分析 | NC/LP | 已有日志 | **必做（本轮）** |
| P1 | SeHGNN-lite | NC | PubMed | 必做（下一轮） |
| P2 | LP Pair Decoder | LP | PubMed | 必做 |
| P3 | Sampled-LP Training | LP | PubMed | 推荐 |
| P4 | EHGNN-to-MLP Distillation | NC | PubMed | 有时间做 |
| P5 | PPR-TopK-lite | 邻居选择 | PubMed 子图 | 设计/demo |

## 证据等级

- **Prototype** — 代码原型，未完整运行
- **Quick Validation** — 单 seed / 少量 epoch
- **Reliable Result** — 多 seed / 完整 epoch

今日大多数结论应标注 Prototype 或 Quick Validation。

## 禁止事项（计划原文摘要）

- 不删除正式复现日志
- 不覆盖 `server_results/`
- 不把 smoke/diagnostic 当正式 5-seed 结果
- 不把工程改动夸大为论文级创新

## 本轮范围（Step 1）

仅完成：**工作区初始化 + P0 已有结果解析**。  
不修改 `main.py`，不跑新训练，不启动 P1–P5 原型。
