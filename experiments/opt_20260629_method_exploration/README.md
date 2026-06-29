# EHGNN 方法级优化探索实验工作区

> 统一实验根目录：`experiments/opt_20260629_method_exploration/`  
> 初始化日期：2026-06-03（本地）  
> 总方向文档：`docs/EHGNN_方法级优化探索计划.md`

## 目的

本目录集中存放 EHGNN 方法级优化探索的全部代码、日志、表格、图片与报告，与正式复现主线（`Node Classification/`、`Link Prediction/`、`server_results/` 归档）严格隔离。

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| P0 已有结果与瓶颈分析 | **In Progress → 本轮已完成解析** | 见 `00_existing_results/report.md` |
| P1 SeHGNN-lite | Planned | 待 ChatGPT 审查 P0 后启动 |
| P2 LP Pair Decoder | Planned | |
| P3 Sampled-LP Training | Planned | |
| P4 Distillation | Planned | |
| P5 PPR-TopK-lite | Planned | |

## Git 快照（初始化时）

| 检查项 | 结果 |
|--------|------|
| 当前分支 | `reproduce-baseline` |
| 当前 commit | `c231c79feeabd554b2e0522f34956ba41d2792a0` |
| 工作区 | **不干净**：未跟踪 `docs/EHGNN_方法级优化探索计划.md`、`experiments/` |
| 最近 commit | `c231c79` Merge reproduce-baseline；`37e3e81` DBLP LP 日志优化 |
| remote origin | `https://github.com/Fire7Lemon/EHGNN.git` |
| remote upstream | `https://github.com/CGCL-codes/EHGNN` |

## 本地环境摘要

见 `LOCAL_ENV_CHECK.md`。结论：**本地不适合运行 EHGNN 训练**（无 DGL、PyTorch CPU、大数据集 OOM 风险）。

## 目录索引

- `PLAN.md` — 探索计划摘要（指向 docs 原文）
- `PAPER_BASIS.md` — 各方法论文依据
- `COMMANDS.md` — 本轮执行命令记录
- `RUN_REGISTRY.csv` — 运行注册表
- `METHOD_MANIFEST.csv` — 方法清单（P0–P5）
- `SERVER_RUN_QUEUE.md` — 后续服务器任务队列
- `00_existing_results/` — P0 产出
- `01_` … `05_` — 各方法实验目录（骨架）
- `summary/` — 跨方法汇总索引

## 重要约束

1. 所有新实验输出必须写在本目录下，**不得**写入 `Node Classification/results/` 或 `Link Prediction/results/`。
2. 服务器运行后，用户将同名路径下载回本地合并。
3. **禁止**修改或覆盖 `server_results/` 已有归档。
