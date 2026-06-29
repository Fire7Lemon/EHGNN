# 方法论文依据（P0–P5）

> 摘自 `docs/EHGNN_方法级优化探索计划.md`，供后续方法原型对照。

## P0 — 已有结果与瓶颈分析

- **依据**：EHGNN 复现日志与运行时间分解
- **边界**：仅汇总已有证据，不跑新实验

## P1 — SeHGNN-lite

- **参考**：SeHGNN / SIGN / NARS — 预计算多关系多跳聚合特征 + 轻量分类头
- **与 EHGNN 关系**：复用 HPPR/Top-K 一次聚合，导出特征后用 MLP 替代训练期图传播
- **边界**：Prototype / Quick Validation

## P2 — LP Pair Decoder

- **参考**：SEAL / BUDDY / ELPH — pair-wise 链路解码
- **改动**：点积解码 → pair MLP（拼接/双线性源-目标表征）
- **边界**：Prototype / Quick Validation

## P3 — Sampled-LP Training

- **参考**：GraphSAINT / Cluster-GCN / HGSampling
- **改动**：全图 author-node 逐步训练 → 采样式边/子图训练协议
- **边界**：Quick Validation；**不等价**于论文默认 LP 协议

## P4 — EHGNN-to-MLP Distillation

- **参考**：GLNN — GNN 教师 → MLP 学生
- **改动**：EHGNN 作 teacher，蒸馏到轻量 MLP 做推理加速探索
- **边界**：Inference optimization only

## P5 — PPR-TopK-lite

- **参考**：PPRGo / APPNP — 确定性 PPR Top-K 邻居
- **改动**：替代随机游走频率估计的 PPR Top-K 设计
- **边界**：Design / demo only

## EHGNN 基线论文

- EHGNN: Efficient Heterogeneous Graph Neural Network (IEEE TBD 2025)
- 核心：元路径随机游走 + Top-K + 一次 scatter 聚合
