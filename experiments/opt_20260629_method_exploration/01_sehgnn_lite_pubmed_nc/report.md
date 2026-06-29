# P1 SeHGNN-lite PubMed NC

> 状态：**Code Prepared** — 待服务器运行  
> 证据等级：Prototype / Quick Validation（单 seed=42）

## 方法动机

EHGNN 在训练阶段每个 batch 仍要通过 `get_model_need` + `EHGNN.forward` 做 meta-path 邻居 scatter 聚合。SeHGNN-lite 借鉴 SeHGNN/SIGN/NARS：在训练前**一次性**预计算各 meta-path 聚合特征视图，训练期仅更新轻量 MLP / fusion head，降低训练期图传播开销。

## 与原始 EHGNN 的区别

| 环节 | EHGNN | SeHGNN-lite |
|------|-------|-------------|
| RW + Top-K | 每 epoch 复用预建 CSR | 同左，训练前对全部 labeled 节点建一次 |
| 邻居聚合 | 训练 batch 内 scatter + 可学习 M/T weight | 训练前 CSR×特征预计算，**均匀** type 权重 |
| 分类头 | 内置 MLP + α 融合自身表征 | ConcatMLP 或 MeanFusionMLP |
| 训练 loop | 需 `features` + 稀疏索引 | 仅 tensor views + 轻量 head |

## 代码复用审计

### 原始 PubMed NC 流程

1. `load_PubMed(path, 'PubMed')` → DGL 异质图 `g`、按类型分桶的 `features` dict、`labels`、`idx_train`/`idx_test`（disease 节点局部 id）
2. 6 条 `metapaths_pubmed` 逐条 `random_walk_sim(idx_train/test, g, metapath, walk_num=40, K=20)` → CSR 相似矩阵列表
3. 训练：`DataLoader(idx_train)` → `get_model_need` → `EHGNN.forward(features, s_features, s_idxs, t_idxs, ...)` → NLLLoss
4. 评估：每 `val_epochs=5` 在 `idx_test` 上算 Macro/Micro-F1（`utils.accuracy`）
5. 结果：`Node Classification/results/pubmed_nc_result.txt` 或 5-seed 批次 `pubmed_seed_42.txt`

### 可复用函数

| 函数 | 文件 | 用途 |
|------|------|------|
| `load_PubMed` | utils.py | 数据加载 |
| `random_walk_sim` | utils.py | meta-path RW + Top-K → CSR |
| `accuracy` | utils.py | Macro/Micro-F1 |
| `get_model_need` / `get_weights_sidx` | utils.py | 原 EHGNN batch 索引（P1 预计算阶段不直接使用） |

### SeHGNN-lite 接入点

1. **预计算**：对 `torch.cat([idx_train, idx_test])` 一次性跑 RW（原代码分 train/test 两次）
2. **视图构建**：`ehgnn_precompute.metapath_view` 用 `sim_csr @ features[t_type]` 代替 forward 内 scatter
3. **视图列表**：`[self_raw] + [mp_1, ..., mp_6]` → 7 视图 × 200-d
4. **分类**：`sehgnn_lite_model.ConcatMLP` / `MeanFusionMLP`

### 风险与不确定点

1. **聚合不含 EHGNN 内层 MLP**：预计算用 raw 200-d 特征；EHGNN 在 scatter 前对邻居做可训练 MLP + L2 norm — 行为不完全等价，属 intentional SIGN-style 简化
2. **无 learnable MWeight/TWeight**：预计算用均匀 type 权重；EHGNN 有可学习 softmax 权重
3. **无 α 自身融合**：EHGNN 有 `(1-α)*agg + α*mlp(self)`；SeHGNN-lite 以 self raw view 作为 concat 第一通道近似
4. **`s_type` 一致性**：各 metapath RW 返回的源类型应均为 disease（features index 1）；脚本会 warn 若不一致
5. **本地无 DGL**：仅 py_compile 非 DGL 模块；完整运行需服务器

## 新增文件

```text
code/sehgnn_lite_model.py          # ConcatMLP, MeanFusionMLP
code/ehgnn_precompute.py           # CSR 预聚合视图
code/run_pubmed_nc_sehgnn_lite.py  # 服务器入口
code/parse_sehgnn_lite_results.py
code/plot_sehgnn_lite_results.py
configs/pubmed_seed42_concat.yaml
configs/ehgnn_pubmed_baseline_seed42.json
server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh
```

## 运行方式

### 服务器（推荐）

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh
```

### 手动

```bash
cd /home/mayq/ehgnn/EHGNN
conda activate ehgnn
python -u experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/run_pubmed_nc_sehgnn_lite.py \
  --seed 42 --fusion concat \
  --root_out experiments/opt_20260629_method_exploration \
  2>&1 | tee experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/logs/pubmed_nc_sehgnn_lite_seed42_concat.log
```

## 服务器运行命令

见 `server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh`（含 parse + plot 后处理）。

## 当前状态

- **Code Prepared** — 代码与脚本已就绪
- **未在本地/服务器执行训练**（本地无 DGL）
- 语法检查：`sehgnn_lite_model.py`、`parse_*`、`plot_*` 已通过 py_compile；`run_pubmed_nc_sehgnn_lite.py` 依赖 DGL，本地跳过

## 预期输出

| 路径 | 内容 |
|------|------|
| `logs/pubmed_nc_sehgnn_lite_seed42_concat.log` | 完整训练日志 |
| `results/pubmed_nc_sehgnn_lite_seed42_concat.csv` | 指标 CSV |
| `results/pubmed_nc_sehgnn_lite_vs_ehgnn.csv` | 对比表 |
| `figs/sehgnn_lite_vs_ehgnn_*.png` | 3 张对比图 |

## 结论边界

**当前仅为方法级原型和单 seed quick validation 准备，不代表已完成完整 5-seed 验证。**

- 不能写「全面优于 EHGNN」
- 可写「构建了 SeHGNN-lite 原型并在 PubMed NC seed=42 上完成初步验证」（服务器跑完后）

## 风险与修复记录

- **2026-06-03 服务器 P5 首次运行**：`ModuleNotFoundError: No module named 'utils'` — 根因是实验脚本 `PROJECT_ROOT` 少算一层（指向 `experiments/` 而非 EHGNN 根目录），`Node Classification` 未进入 `sys.path`。
- **修复**：入口脚本改用 `Path(__file__).resolve().parents[4]`，并在 `from utils import ...` 前显式插入 `Node Classification`；对应 server 脚本增加 `export PYTHONPATH="$PROJECT_ROOT/Node Classification:..."`。
- **未修改** `Node Classification/main.py`、`utils.py`、`models.py`。

## Baseline 参考（EHGNN seed=42）

| 指标 | 值 |
|------|-----|
| Best Macro-F1 | 0.6313 |
| Best Micro-F1 | 0.6512 |
| Total time | 3.66 s |
| 来源 | `server_results/2026-06-02_/.../pubmed_seed_42.txt` |
