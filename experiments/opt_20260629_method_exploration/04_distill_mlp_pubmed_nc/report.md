# P4 EHGNN-to-MLP Distillation PubMed NC

> 状态：**Code Prepared** — 待服务器运行  
> 证据等级：Prototype / Quick Validation（推理轻量化探索）

## 方法动机

EHGNN NC 推理依赖异构图预处理 + 多 meta-path scatter。借鉴 GLNN，用 **EHGNN teacher** 的 soft labels 蒸馏 **纯 MLP student**，探索去图依赖的推理加速。

## 与原始 EHGNN NC 的区别

| 环节 | EHGNN NC | P4 Student |
|------|----------|------------|
| 输入 | 图 + RW 矩阵 + 邻居索引 | **节点特征 only** |
| 模型 | EHGNN | MLP |
| 训练 | 端到端 NLLLoss | CE + KD mixed loss |
| 推理 | 需 get_model_need + forward | 单次 MLP forward |

**P4 不降低 teacher 训练成本**；teacher 仍需完整 RW + EHGNN 训练。

## 代码复用审计

### 原始 PubMed NC teacher 流程

`Node Classification/main.py`：load_PubMed → RW sim → EHGNN → log_softmax → NLLLoss；评估 Macro/Micro-F1。

### checkpoint / logits 可用性

- 仓库内 **无** 已有 `.pt` / checkpoint（Glob 搜索为空）
- `server_results/` 仅有 `pubmed_seed_42.txt` 指标，**无** 模型权重或 logits
- **默认 `--teacher_mode train`**：P4 脚本内训练 teacher 并保存 logits
- **`--teacher_mode load`**：从 `04_distill_mlp_pubmed_nc/results/pubmed_nc_teacher_seed42_*.pt/json` 加载；缺失则报错

### student 输入选择

| 模式 | 输入 | 默认 |
|------|------|------|
| `raw` | disease 200-d 原始特征 | **是** |
| `precomputed` | P1 `ehgnn_precompute` 多 meta-path 视图 concat | 可选 |
| `raw_plus_precomputed` | raw + 预计算视图 | 可选 |

### 蒸馏目标设计

- Teacher 输出：**raw logits**（`model.forward` 后、log_softmax 前）
- 保存：`pubmed_nc_teacher_seed42_logits.pt`
- Loss：`(1-α)*CE + α*T²*KL(student_T || teacher_T)`，`T=3`, `α=0.5`

### 推理时间测量方案

| 模型 | 测量方式 | 是否含预处理 |
|------|----------|--------------|
| Teacher | 单 test batch forward 均值 ×20 | **否**（不含 RW） |
| Student | 全 test 特征 tensor forward 均值 ×20 | **否** |
| speedup | teacher_ms / student_ms | — |

### 风险与不确定点

1. Teacher 需重新训练（无现成 checkpoint）
2. KD 超参 T/α 未调优
3. Student 默认仅 raw 特征，可能弱于 teacher
4. 推理 speedup 对比口径不同（teacher 单 batch vs student 全 test tensor）
5. 本地无 DGL

## 蒸馏损失设计

见 `distillation_losses.py` — `mixed_distill_loss` with temperature softmax KL.

## 新增文件

```text
code/mlp_student_model.py
code/distillation_losses.py
code/main_pubmed_nc_distill.py
code/parse_distill_results.py
code/plot_distill_results.py
configs/ehgnn_pubmed_nc_baseline_seed42.json
server_scripts/run_p4_distill_pubmed_nc.sh
```

## 运行方式

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p4_distill_pubmed_nc.sh
```

## 服务器运行命令

见 `run_p4_distill_pubmed_nc.sh`（teacher 100 ep + student 200 ep；可改 30/50 快试）。

## 当前状态

Code Prepared；未执行训练；py_compile 待验证

## 预期输出

| 文件 | 内容 |
|------|------|
| `results/pubmed_nc_teacher_seed42_logits.pt` | train/test teacher logits |
| `results/pubmed_nc_teacher_seed42_metrics.json` | teacher F1 |
| `results/pubmed_nc_distill_student_seed42.csv` | 完整对比 |
| `figs/distill_*.png` | 4 张图 |

## Baseline seed=42（已找到）

Best Macro **0.6313**，Micro **0.6512** — `pubmed_seed_42.txt`

## 结论边界

**当前仅为 EHGNN-to-MLP 蒸馏原型和单 seed quick validation 准备。该方法主要探索推理阶段轻量化，不代表降低 EHGNN teacher 的训练成本，也不代表完整 5-seed 验证完成。**

## 风险与修复记录

- **2026-06-03**：P5 服务器首次运行 `ModuleNotFoundError: utils`；P4 同类脚本一并修复：`main_pubmed_nc_distill.py` 使用 `parents[4]` + `Node Classification` 路径；`run_p4_distill_pubmed_nc.sh` 增加 NC `PYTHONPATH`。
- **未修改** `Node Classification/` 正式主线代码。

- **2026-06-03 data path 修复**：`main_pubmed_nc_distill.py` 使用 `resolve_project_data_path(PROJECT_ROOT)`；`ehgnn_precompute.py` 无 `load_PubMed` 调用，无需改动。

- **2026-06-03 parser 路径统一**：`parse_distill_results.py` 改用 `safe_relpath`（预防性）。

- **2026-06-03 系统审计**：`TEACHER_MODE=load` 当 logits 存在、`PARSE_ONLY=1`、plot skip。见 `AUDIT_FIX_REPORT.md`。

- **2026-06-03 common import path（P4 阻塞根因）**：`main_pubmed_nc_distill.py` 启动时报 `ModuleNotFoundError: No module named 'path_utils'`。根因是 `common/` 未在 import 前加入 `sys.path`（误用 `CODE_DIR.parents[4]`）。已改为 `EXP_ROOT = parents[2]` + `COMMON_DIR` bootstrap；smoke test 从 `py_compile` 升级为 `import_smoke_checks.py`。**部署后请先跑 smoke test 再重跑 P4。**

- **2026-06-03 P4 teacher 流程对齐（t_typess None）**：import smoke 通过后，teacher 构造阶段 `AttributeError: 'NoneType' object has no attribute 'append'`（line 340）。根因：`t_typess` 被误初始化为 `None` 而非 `[]`。已提取 `build_rw_similarity_matrices()` 对齐 `Node Classification/main.py` L88–100；新增 `--dry_run_runtime_check`。

---

## P4 Runtime Bugfix: Teacher Flow Alignment

### 原始 Node Classification/main.py 中 PubMed teacher 流程

1. **数据加载**：`load_PubMed(path, 'PubMed', is_normalize)` → `g, features, labels, idx_train, idx_test`；`features` 为 `{type_idx: Tensor}` 字典；`metapaths = metapaths_pubmed`（6 条）。
2. **RW 相似矩阵**（L88–100）：
   ```python
   train_matrixs, test_matrixs, t_typess = [], [], []
   for metapath in metapaths:
       train_matrix, t_types, s_type = random_walk_sim(idx_train, g, metapath, walk_num, K, r_neighbor)
       test_matrix, _, _ = random_walk_sim(idx_test, g, metapath, walk_num, K, r_neighbor)
       train_matrixs.append(train_matrix)   # dict[t_type -> csr_matrix]
       test_matrixs.append(test_matrix)
       t_typess.append(t_types)               # list of target node type indices
   ```
   - `random_walk_sim` 返回 `(sim_matrix_dict, list(tnode_types), s_type_int)`。
   - **无 `s_typess`**；源类型仅 `s_type`（游走起点类型 index）。
3. **get_model_need**（utils.py L581–597）：`get_model_need(n_metapaths, sim_matrixs, t_typess, batch)` → `(s_idxs, t_idxs, weightss)`，对每个 meta-path `i` 遍历 `t_typess[i]` 中各 `t_type` 调 `get_weights_sidx(sim_matrixs[i][t_type], batch)`。
4. **EHGNN 初始化**（L104–115）：`in_feat=features[0].shape[1]`, `hidden`, `out_feat=labels.max()+1`, `n_layer`, `alpha`, `n_metapath=len(metapaths)`, `n_types=len(g.ntypes)`, `wo_l2/wo_mweight/wo_tweight`, `dropout`。
5. **Forward + 损失**（L136–141）：`batch_out = model(features, features[s_type][batch], s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device)` → **raw logits**；`F.log_softmax` → `NLLLoss`。
6. **评估**：同样 forward 后对输出 `log_softmax`，再 `accuracy(...)`。

### P4 当前复刻流程差异（修复前）

| 项 | 原始 main.py | P4（bug） |
|----|--------------|-----------|
| `t_typess` 初始化 | `[]` | **`None`** → append 崩溃 |
| RW 循环 | 与上表一致 | 逻辑同 main，但初始化错误 |
| `s_type` | 每次 RW 覆盖，最后用末条 meta-path 的 `s_type` | 仅首次赋值（等价若各 path 相同） |
| EHGNN 参数 | 见上 | 一致（默认 wo_*=False） |
| Teacher 训练 loss | log_softmax + NLLLoss | 一致 |
| 蒸馏保存 | N/A | `collect_teacher_logits` 保存 **raw logits**（forward 后、log_softmax 前） |

### 修复方案

1. 新增 `build_rw_similarity_matrices()`，逐行对齐 `main.py` L88–100；`t_typess = []`。
2. 新增 `create_teacher_ehgnn()` 集中 EHGNN 构造，与 main 默认超参一致。
3. 明确 `TEACHER_SIGNAL_TYPE = "raw_logits"`；训练仍用 log_softmax+NLLLoss；蒸馏 KL 对 raw logits 做 temperature softmax（见 `distillation_losses.py`）。
4. 新增 `--dry_run_runtime_check`：load → RW（32 节点 × walk_num≤5）→ `get_model_need` → 1 teacher forward → 1 student forward；**不写 results**。

### 验证结果

- 本地（无 DGL）：`import_smoke_checks` PASS；P4 dry run **SKIP**（缺 dgl/torch_scatter）。
- 服务器（ehgnn）：部署后执行：
  ```bash
  bash experiments/opt_20260629_method_exploration/server_scripts/smoke_test_method_exploration.sh
  python experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py \
    --dataset PubMed --seed 42 --teacher_epochs 1 --student_epochs 1 \
    --teacher_mode train --student_input raw --dry_run_runtime_check \
    --root_out experiments/opt_20260629_method_exploration
  ```
  期望输出含 `[P4-DRY-RUN] PASS`。通过后方可 `run_p4_distill_pubmed_nc.sh`。

