# 服务器运行队列（框架）

> **本轮不执行训练。** 待 P1–P5 脚本生成后再补充具体命令。  
> 所有服务器输出必须写到项目内相对路径：

```text
experiments/opt_20260629_method_exploration/<method_dir>/
```

**禁止**写入 `Node Classification/results/`、`Link Prediction/results/` 或覆盖 `server_results/`。

## 日志规范

所有服务器运行必须使用 `tee` 写入对应方法的 `logs/` 目录，例如：

```bash
# 模板（P1 示例，命令待 P1 脚本完成后填写）
cd /path/to/EHGNN
mkdir -p experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/logs
python experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/run_sehgnn_lite.py \
  2>&1 | tee experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/logs/run_seed42.log
```

## P1 SeHGNN-lite PubMed NC

- 状态：**待服务器运行**（Code Prepared）
- 脚本：
  `experiments/opt_20260629_method_exploration/server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh`
- 输出根路径：
  `experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/`
- 预计输出：
  - `logs/pubmed_nc_sehgnn_lite_seed42_concat.log`
  - `results/pubmed_nc_sehgnn_lite_seed42_concat.csv`
  - `results/pubmed_nc_sehgnn_lite_vs_ehgnn.csv`
  - `figs/sehgnn_lite_vs_ehgnn_*.png`
  - `report.md`（运行后更新结论）
- 注意：这是单 seed quick validation，不是正式 5-seed 结论。
- 环境：`conda activate ehgnn`；项目路径 `/home/mayq/ehgnn/EHGNN`

## P2 LP Pair Decoder PubMed LP

- 状态：**待服务器运行**（Code Prepared）
- 脚本：
  `experiments/opt_20260629_method_exploration/server_scripts/run_p2_pair_decoder_pubmed_lp.sh`
- 输出根路径：
  `experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/`
- 预计输出：
  - `logs/pubmed_lp_pair_decoder_seed42_pair_mlp.log`
  - `results/pubmed_lp_pair_decoder_seed42_pair_mlp.csv`
  - `figs/pair_decoder_*_compare.png`
  - `report.md`（运行后更新）
- 注意：单 seed quick validation，不是正式 5-seed 结论。

## P3 Sampled-LP Training PubMed LP

- 状态：**待服务器运行**（Code Prepared）
- 脚本：
  `experiments/opt_20260629_method_exploration/server_scripts/run_p3_sampled_lp_pubmed.sh`
- 输出根路径：
  `experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/`
- 预计输出：
  - `logs/pubmed_lp_sampled_seed42_ratio050.log`
  - `logs/pubmed_lp_sampled_seed42_ratio025.log`
  - `results/*.csv`
  - `figs/sampled_lp_*.png`
- 默认运行：sample_ratio=0.50 与 0.25
- 注意：单 seed quick validation；**改变训练协议**，不能与论文默认设置等价比较。

## P4 EHGNN-to-MLP Distillation PubMed NC

- 状态：**待服务器运行**（Code Prepared）
- 脚本：
  `experiments/opt_20260629_method_exploration/server_scripts/run_p4_distill_pubmed_nc.sh`
- 输出根路径：
  `experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/`
- 预计输出：
  - `logs/pubmed_nc_distill_seed42_raw.log`
  - `results/pubmed_nc_teacher_seed42_logits.pt`
  - `results/pubmed_nc_teacher_seed42_metrics.json`
  - `results/pubmed_nc_distill_student_seed42.csv`
  - `figs/distill_*.png`
- 默认运行：teacher_mode=train, student_input=raw, seed=42
- 注意：单 seed quick validation；主要验证推理轻量化，非 5-seed 结论。

## 待运行任务队列

| 优先级 | 方法 | 目录 | 数据集 | 任务 | 状态 | 备注 |
|--------|------|------|--------|------|------|------|
| P1 | SeHGNN-lite | `01_sehgnn_lite_pubmed_nc/` | PubMed | NC | **Ready — 脚本已就绪** | seed=42 concat；见 `run_p1_sehgnn_lite_pubmed_nc.sh` |
| P2 | LP Pair Decoder | `02_lp_pair_decoder_pubmed_lp/` | PubMed | LP | **Ready — 脚本已就绪** | seed=42 pair_mlp；见 `run_p2_pair_decoder_pubmed_lp.sh` |
| P3 | Sampled-LP Training | `03_sampled_lp_training_pubmed_lp/` | PubMed | LP | **Ready — 脚本已就绪** | ratio=0.50 & 0.25；见 `run_p3_sampled_lp_pubmed.sh` |
| P4 | Distillation | `04_distill_mlp_pubmed_nc/` | PubMed | NC | **Ready — 脚本已就绪** | 见 `run_p4_distill_pubmed_nc.sh` |
| P5 | PPR-TopK-lite | `05_ppr_topk_lite_design/` | PubMed 子图 | 邻居选择 | **Ready — 脚本已就绪** | 见 `run_p5_ppr_topk_demo.sh` |

## P5 PPR-TopK-lite PubMed Demo

- 状态：**待服务器运行**（Code Prepared）
- 脚本：
  `experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh`
- 输出根路径：
  `experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/`
- 预计输出：
  - `logs/*.log`
  - `results/*.csv`
  - `results/*.json`
  - `figs/*.png`
  - `report.md`
- 默认运行：
  - dataset=PubMed
  - num_target_nodes=500
  - k=20
  - alpha=0.15
  - num_iters=10
- 注意：这是邻居选择替代方案的 design/demo，不训练模型，不代表完整替代 EHGNN 的 RW/HPPR 流程。
- **2026-06-03 修复**：utils 导入（`parents[4]`）+ data path 尾部 `/` + DGL `adj_external` 兼容层；修复后请重新跑 P5。

## 服务器运行前检查清单

1. 同步最新 `experiments/opt_20260629_method_exploration/`（含 `common/path_utils.py`）
2. 可选：`bash server_scripts/check_server_before_opt_runs.sh`（环境变更时建议重跑）
3. 轻量验证：`bash server_scripts/run_p5_ppr_topk_demo.sh`
4. 全量：`bash server_scripts/run_all_method_exploration.sh`

## 明确不在此队列（今日禁止）

- DBLP LP 新完整 5-seed 训练
- MAG240M 任何完整实验
- Yelp LP 5-seed（归档中未找到完整结果，但今日计划不启动）

## 下载合并说明

服务器运行完成后，用户将 `experiments/opt_20260629_method_exploration/` 整目录下载回本地，与本文档所在工作区合并。

## 环境要求（参考 2026-06-02 批次）

- conda 环境：`ehgnn`
- 项目路径示例：`/home/mayq/ehgnn/EHGNN`
- 详见 `server_results/2026-06-02_/2026-06-02_env_snapshot.txt`
