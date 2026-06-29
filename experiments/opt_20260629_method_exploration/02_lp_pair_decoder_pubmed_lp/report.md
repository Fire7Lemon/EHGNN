# P2 LP Pair Decoder PubMed LP

> 状态：**Code Prepared** — 待服务器运行  
> 证据等级：Prototype / Quick Validation（单 seed=42）

## 方法动机

P0 显示 DBLP LP Final AUC 接近随机，PubMed LP 虽可跑通但 dot-product 解码表达力有限。借鉴 SEAL / BUDDY / ELPH，用 **pair-wise 特征 + MLP** 显式建模 `(u, v)` 关系，而非仅 `dot(h_u, h_v)`。

## 与原始 EHGNN LP 的区别

| 环节 | EHGNN LP | P2 Pair Decoder |
|------|----------|-----------------|
| 节点表征 | EHGNN encoder（不变） | 同左 |
| 链路打分 | `sum(h_u * h_v)` → sigmoid | `MLP([h_u, h_v, \|diff\|, prod])` → logit |
| 训练损失 | BCELoss(sigmoid(dot)) | **BCEWithLogitsLoss(logit)** |
| 评估 | sigmoid(dot) → `accuracy()` | sigmoid(decoder logit) → `accuracy()` |

## 代码复用审计

### 原始 PubMed LP 流程

1. `load_PubMed` → 异质图、`features` dict、正样本 `train_links[2×N]`、测试边 `test_links`、`labels`
2. 每条 meta-path：`lp_rw_seeds_all_nodes` + `random_walk_sim` → CSR 相似矩阵
3. 每 epoch：`num_sample`（= disease 节点数）步；每步采样正边索引，构造负边 `neg_sample`
4. 对 `pos_s/pos_t/neg_s/neg_t` 分别跑 EHGNN → embedding
5. dot → sigmoid → BCELoss；每 `val_epochs` 步 `evaluate_lp()`
6. `evaluate_lp`：全图 disease embedding → test 边 dot → sigmoid → AUC/AP

### 原始 decoder 位置

**文件**：`Link Prediction/main.py`

| 位置 | 代码 |
|------|------|
| 训练打分 L180–182 | `pos = (pos_s_out * pos_t_out).sum(dim=-1)`；`neg = (neg_s_out * neg_t_out).sum(dim=-1)` |
| 训练概率 L182 | `batch_out = torch.cat((pos, neg)).sigmoid()` |
| 训练损失 L184 | `BCELoss(batch_out, y_true)` |
| 评估 L142–145 | `pos = (pos_s_out * pos_t_out).sum(dim=-1)`；`test_out = pos.sigmoid()` |
| 指标 L147 | `accuracy(test_out, y_true)` → roc_auc + average_precision |

**不在** `models.py` 中；decoder 为 main.py 内联 dot-product。

### 可复用函数

| 函数 | 文件 | 用途 |
|------|------|------|
| `load_PubMed` | utils.py | 数据 |
| `random_walk_sim`, `lp_rw_seeds_all_nodes` | utils.py | RW 预计算 |
| `get_model_need` | utils.py | batch 稀疏索引 |
| `neg_sample` | utils.py | 负采样 |
| `accuracy` | utils.py | AUC/AP（输入 probability） |
| `EHGNN` | models.py | 节点 encoder |

### Pair Decoder 接入点

1. **Encoder 不变**：`encode_nodes()` 封装原 `model.forward`
2. **Decoder 替换**：`lp_pair_decoder_model.build_decoder('pair_mlp'|'dot')`
3. **训练**：`BCEWithLogitsLoss` on decoder logits
4. **评估**：`probs = decoder(..., return_logits=False)` → `accuracy(probs, labels)`
5. **DotDecoder**：logit = dot product，用于消融

### degree 特征可行性

- 原 `load_PubMed` **未导出**节点 degree
- **本轮未实现** `PairMLPDegreeDecoder`；列为后续工作

### 风险与不确定点

1. P2 用 BCEWithLogitsLoss；原版 sigmoid(dot)+BCELoss — DotDecoder 近似等价
2. PairMLP 增加参数与计算
3. `train_links` 索引逻辑与 main.py 完全一致
4. 本地无 DGL；服务器验证
5. 不解决 DBLP LP 问题

## 新增文件

见 `code/`、`configs/`、`server_scripts/run_p2_pair_decoder_pubmed_lp.sh`

## 运行方式

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p2_pair_decoder_pubmed_lp.sh
```

## 服务器运行命令

见 `server_scripts/run_p2_pair_decoder_pubmed_lp.sh`

## 当前状态

Code Prepared，未执行训练

## 预期输出

- `logs/pubmed_lp_pair_decoder_seed42_pair_mlp.log`
- `results/pubmed_lp_pair_decoder_seed42_pair_mlp.csv`
- `figs/pair_decoder_*_compare.png`

## Baseline seed=42（已找到）

Best AUC **0.5489**，AP **0.5214**，来源 `server_results/2026-06-02_/.../pubmed_lp_seed_42.log`

## 结论边界

**当前仅为链路预测解码器增强原型和单 seed quick validation 准备，不代表已完成完整 5-seed 验证，也不能说明已解决 DBLP LP 性能偏低问题。**

## 风险与修复记录

- **2026-06-03 服务器 P5 首次运行**暴露同类问题：`utils` 位于 `Link Prediction/utils.py`，从项目根运行实验脚本时需显式 `sys.path` / `PYTHONPATH`。
- **修复**：`main_pubmed_lp_pair_decoder.py` 使用 `parents[4]` 定位 EHGNN 根目录并插入 `Link Prediction`；`run_p2_pair_decoder_pubmed_lp.sh` 增加 `export PYTHONPATH="$PROJECT_ROOT/Link Prediction:..."`。
- **未修改** `Link Prediction/main.py`、`utils.py`、`models.py`。

- **2026-06-03 data path 修复**：`load_PubMed` 需 `PROJECT_ROOT/data/` 且末尾带 `/`；`main_pubmed_lp_pair_decoder.py` 已改用 `resolve_project_data_path`。

- **2026-06-03 parse 路径修复**：主实验可跑通，但 `parse_pair_decoder_results.py` 中 `log_path.relative_to(ROOT)` 因相对 log 与绝对 ROOT 混用失败。已统一 `common/path_utils.safe_relpath` + `resolve_under_root`；`run_p2_pair_decoder_pubmed_lp.sh` 支持 `PARSE_ONLY=1` 与主 CSV 存在时 skip training。

- **2026-06-03 系统审计**：移除 shell PYTHONPATH；parse/plot `\|\| [WARN]`。见 `AUDIT_FIX_REPORT.md`。

- **2026-06-03 common import path**：P4 failed because `common/path_utils.py` was not discoverable before import; all P1–P5 entry/parse/plot scripts now bootstrap `EXP_ROOT/common` via `parents[2]`; smoke test upgraded to import/`--help` checks.

