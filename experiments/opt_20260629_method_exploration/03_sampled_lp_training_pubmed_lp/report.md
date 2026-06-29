# P3 Sampled-LP Training PubMed LP

> 状态：**Code Prepared** — 待服务器运行  
> 证据等级：Quick Validation（单 seed；**训练协议已改变**）

## 方法动机

原始 PubMed LP 每 epoch 对 `num_sample`（≈ disease 节点数）个训练 index 做 DataLoader 迭代，训练成本高。借鉴 GraphSAINT / Cluster-GCN 思路，**只采样一部分训练 step**，观察 AUC/AP 与 runtime 的 trade-off。

## 与原始 EHGNN LP 的区别

| 环节 | EHGNN LP | P3 Sampled-LP |
|------|----------|---------------|
| Decoder | dot-product | **同左（不变）** |
| Loss | BCELoss(sigmoid(dot)) | **同左（不变）** |
| 评估 | 全量 `evaluate_lp()` | **同左（不变）** |
| 训练 | 每 epoch `num_sample` 个 train index | 每 epoch `round(num_sample × ratio)` 个 index |

**P3 不引入 P2 PairMLPDecoder**，保证单一变量为训练采样协议。

## 代码复用审计

### 原始 PubMed LP 训练循环

见 `Link Prediction/main.py` L154–221：每 epoch 生成 `train_idx`，DataLoader 分批，dot+sigmoid+BCE，定期 `evaluate_lp()`。

### 原始训练样本组织方式

- `train_links`：`[2, num_pos_edges]` 正样本边
- `train_idx`：长度 `num_sample`，索引 `train_links[0/1][idx]`
- `batch_size`：index batch 大小（非节点 batch）
- 每 step 正负 1:1 via `neg_sample`

### 可采样对象

**每 epoch 的 train index 张量长度**（P3 采用）；`sample_ratio=1.0` 时与原版 step 数相同。

### Sampled-LP 接入点

`sampled_lp_loader.sample_train_indices()` 在 `main_pubmed_lp_sampled_training.py` 替换 L160 的 `randint` 长度。

### 为什么本方法只改变训练协议

P2 负责 decoder；P3 只改训练成本；评估不变便于对比。

### 风险与不确定点

- 与论文默认协议不等价
- evaluate 仍全量，eval 占比可能随 ratio 下降而上升
- 本地无 DGL

## 采样协议设计

- **对象**：`train_index_tensor_per_epoch`
- **默认**：ratio=0.5，`resample_each_epoch=True`
- **兼容**：ratio=1.0 → full training

## 新增文件

见 `code/`、`configs/`、`server_scripts/run_p3_sampled_lp_pubmed.sh`

## 运行方式

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p3_sampled_lp_pubmed.sh
```

## 服务器运行命令

ratio=0.50 与 0.25 各 100 epoch（脚本内可改 30）。

## 当前状态

Code Prepared；未训练

## 预期输出

`logs/*ratio*.log`、`results/*.csv`、`figs/sampled_lp_*.png`

## 结论边界

**当前仅为采样式链路预测训练协议的单 seed quick validation 准备。该方法改变训练协议，因此不能直接等价于论文默认训练设置，也不能作为正式 5-seed 结论。**
