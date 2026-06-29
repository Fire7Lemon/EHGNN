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
