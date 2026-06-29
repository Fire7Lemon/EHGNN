# 方法级优化探索 — 服务器总控运行说明

统一输出根路径：

```text
experiments/opt_20260629_method_exploration/
```

**禁止**写入 `server_results/`、`Node Classification/results/`、`Link Prediction/results/`。

---

## 1. 推荐：先运行环境检查

```bash
cd /home/mayq/ehgnn/EHGNN
bash experiments/opt_20260629_method_exploration/server_scripts/check_server_before_opt_runs.sh
```

检查项：当前路径、git branch/commit、Python/Torch/DGL 版本、CUDA、nvidia-smi、实验目录是否存在。

输出日志：

```text
experiments/opt_20260629_method_exploration/server_env_check.log
```

---

## 2. 再运行总控脚本（P5 → P1 → P2 → P3 → P4）

```bash
cd /home/mayq/ehgnn/EHGNN
bash experiments/opt_20260629_method_exploration/server_scripts/run_all_method_exploration.sh
```

总控行为：

- 按顺序执行 P5、P1、P2、P3、P4 各自子脚本
- 每阶段输出 `[START]` / `[DONE]` 及 `return_code`
- **某一阶段失败时仍继续后续阶段**
- 各阶段详细日志仍写入各 P? 目录下的 `logs/`（由子脚本负责）

总控日志：

```text
experiments/opt_20260629_method_exploration/logs_run_all_method_exploration.log
```

运行状态 CSV（每阶段一行，追加更新）：

```text
experiments/opt_20260629_method_exploration/server_run_status.csv
```

字段：`stage,script,status,return_code,start_time,end_time,duration_sec,note`

---

## 3. 轻量任务优先（可选）

若只想先验证邻居选择与 NC/LP 快速实验，可单独运行：

```bash
cd /home/mayq/ehgnn/EHGNN
bash experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh
bash experiments/opt_20260629_method_exploration/server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh
bash experiments/opt_20260629_method_exploration/server_scripts/run_p2_pair_decoder_pubmed_lp.sh
```

P3（Sampled-LP，两个 ratio）与 P4（蒸馏，含 teacher 训练）耗时更长，建议环境检查通过后再跑总控或单独执行。

---

## 4. 各任务证据等级

| 阶段 | 方法 | 证据等级 | 说明 |
|------|------|----------|------|
| P5 | PPR-TopK-lite | **Prototype / Demo** | 邻居选择 design/demo，**不训练模型** |
| P1 | SeHGNN-lite | **Quick Validation** | 单 seed PubMed NC，验证预计算+轻量融合 |
| P2 | LP Pair Decoder | **Quick Validation** | 单 seed PubMed LP，验证 PairMLP 解码器 |
| P3 | Sampled-LP Training | **Quick Validation** | 改变训练协议（采样比例），非完整 5-seed |
| P4 | EHGNN-to-MLP Distillation | **Quick Validation** | 推理轻量化验证，含 teacher 训练 + student 蒸馏 |

以上结果均为探索性验证，**不能**直接作为正式复现或性能优化结论。

---

## 5. 子脚本一览

| 阶段 | 脚本 | 输出目录 |
|------|------|----------|
| P5 | `run_p5_ppr_topk_demo.sh` | `05_ppr_topk_lite_design/` |
| P1 | `run_p1_sehgnn_lite_pubmed_nc.sh` | `01_sehgnn_lite_pubmed_nc/` |
| P2 | `run_p2_pair_decoder_pubmed_lp.sh` | `02_lp_pair_decoder_pubmed_lp/` |
| P3 | `run_p3_sampled_lp_pubmed.sh` | `03_sampled_lp_training_pubmed_lp/` |
| P4 | `run_p4_distill_pubmed_nc.sh` | `04_distill_mlp_pubmed_nc/` |

---

## 6. 运行后合并回本地

服务器完成后，将整目录下载：

```text
experiments/opt_20260629_method_exploration/
```

与本地工作区合并，再运行各 P? 的 parse/plot 脚本（若子脚本未自动调用）。
