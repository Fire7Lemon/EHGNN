# 服务器运行队列 — 方法级优化探索

统一输出根：`experiments/opt_20260629_method_exploration/`

**禁止**写入 `Node Classification/results/`、`Link Prediction/results/` 或覆盖 `server_results/`。

---

## 当前建议（审计后 2026-06-03）

### 0. 运行前必做（含 common import path 修复后）

P4 曾因 `path_utils` 不可发现而失败；已修复并升级 smoke test。**部署后请先跑 smoke test。**

```bash
cd /home/mayq/ehgnn/EHGNN
bash experiments/opt_20260629_method_exploration/server_scripts/smoke_test_method_exploration.sh
bash experiments/opt_20260629_method_exploration/server_scripts/check_server_before_opt_runs.sh
```

### 1. P5 — **不需要重跑主 demo**

服务器已有核心结果：

- `05_ppr_topk_lite_design/results/pubmed_ppr_topk_demo.csv`
- `05_ppr_topk_lite_design/results/pubmed_ppr_topk_demo_meta.json`

指标（demo）：PPR 459.53s | RW 0.22s | mean Jaccard 0.0844 | overlap@20 0.1096

**仅补 parse/plot：**

```bash
PARSE_ONLY=1 bash experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh
```

结论：朴素 PPR-lite 不适合直接替代 RW Top-K，仅作 design/demo。

### 2. P3 — ratio050 **不需要重跑**

已有：`03_sampled_lp_training_pubmed_lp/results/pubmed_lp_sampled_training_seed42_ratio050.csv`

**下一步：继续 ratio025 + parse/plot**

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p3_sampled_lp_pubmed.sh
# 或仅 parse/plot：
PARSE_ONLY=1 bash experiments/opt_20260629_method_exploration/server_scripts/run_p3_sampled_lp_pubmed.sh
```

### 3. P2 — 若主 CSV 已存在

```bash
PARSE_ONLY=1 bash experiments/opt_20260629_method_exploration/server_scripts/run_p2_pair_decoder_pubmed_lp.sh
```

### 4. P1 / P4 — 仍需正式训练

P4：`path_utils` import 已修复；**`t_typess None` 已修复**。先 smoke test + dry run，再训练。

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/smoke_test_method_exploration.sh

python experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py \
  --dataset PubMed --seed 42 --teacher_epochs 1 --student_epochs 1 \
  --teacher_mode train --student_input raw --dry_run_runtime_check \
  --root_out experiments/opt_20260629_method_exploration

bash experiments/opt_20260629_method_exploration/server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh
bash experiments/opt_20260629_method_exploration/server_scripts/run_p4_distill_pubmed_nc.sh
```

### 5. 全量总控（可选，按 P5→P1→P2→P3→P4）

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_all_method_exploration.sh
```

各子脚本已支持 skip-existing / `PARSE_ONLY=1`；parse/plot 失败仅 WARN。

---

## 待运行任务队列

| 优先级 | 方法 | 目录 | 数据集 | 任务 | 状态 | 备注 |
|--------|------|------|--------|------|------|------|
| — | Smoke test | `server_scripts/` | — | 审计 | **Run first** | `smoke_test_method_exploration.sh` |
| P5 | PPR-TopK-lite | `05_ppr_topk_lite_design/` | PubMed | 邻居 demo | **Demo done — parse/plot only** | `PARSE_ONLY=1` |
| P3 | Sampled-LP | `03_sampled_lp_training_pubmed_lp/` | PubMed LP | LP | **ratio050 done — run ratio025** | skip050 自动 |
| P2 | LP Pair Decoder | `02_lp_pair_decoder_pubmed_lp/` | PubMed LP | LP | **Check CSV — parse-only?** | `PARSE_ONLY=1` |
| P1 | SeHGNN-lite | `01_sehgnn_lite_pubmed_nc/` | PubMed | NC | **Pending train** | |
| P4 | Distillation | `04_distill_mlp_pubmed_nc/` | PubMed | NC | **Pending train** (t_typess fixed) | dry_run → full train |

## PARSE_ONLY 约定

所有 `run_p1`–`run_p5` 脚本支持：

```bash
PARSE_ONLY=1 bash experiments/opt_20260629_method_exploration/server_scripts/run_pN_....sh
```

跳过训练/demo，仅 parse + plot（plot 无 matplotlib 时 exit 0）。

## 明确不在此队列（今日禁止）

- DBLP LP 新完整 5-seed 训练
- MAG240M 任何完整实验
- Yelp LP 5-seed

## 环境要求

- conda 环境：`ehgnn`
- 项目路径：`/home/mayq/ehgnn/EHGNN`
- 可选：`pip install matplotlib`（服务器无 matplotlib 时 plot 会跳过）
- 详见 `AUDIT_FIX_REPORT.md`
