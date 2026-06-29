# 本地环境检查

检查时间：2026-06-03（Windows 10）

## Git 状态

| 检查项 | 结果 | 对今日任务影响 |
|--------|------|----------------|
| 当前分支 | `reproduce-baseline` | 在复现分支上初始化探索目录，符合计划 |
| 当前 commit | `c231c79feeabd554b2e0522f34956ba41d2792a0` | 含 DBLP LP 日志优化 commit |
| 工作区是否干净 | **否** — 未跟踪 `docs/EHGNN_方法级优化探索计划.md`、`experiments/` | 不影响 P0 只读解析 |
| 最近 commits | `c231c79` merge；`37e3e81/3ad3790/fe7eb0b` DBLP LP 日志优化；`b27c9ca` 忽略 server_results | — |

## Python 环境

| 检查项 | 结果 | 对今日任务影响 |
|--------|------|----------------|
| Python 版本 | **3.13.2** | 与服务器 conda 环境可能不一致 |
| 可执行文件 | `C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe` | — |
| PyTorch | **2.7.0+cpu**（可用） | 无 CUDA，无法 GPU 训练 |
| DGL | **未安装**（ModuleNotFoundError） | **无法本地运行 EHGNN 主流程** |
| matplotlib | 可用（P0 画图已验证） | P0 本地可完成 |

## 硬件（推断）

- GPU：RTX 4070 Laptop ~8GB（对话历史记录；本轮未实测 CUDA）
- 内存：~16GB — DBLP 数据加载曾本地 OOM（smoke）

## 综合判断

| 检查项 | 结果 |
|--------|------|
| 本地是否适合训练 EHGNN | **否** |
| P0 日志解析 / 画图 | **可以本地完成** |
| 哪些实验需要服务器 | P1–P5 全部训练；任何 DGL+GPU 依赖的 EHGNN 运行 |

## 本地不可运行原因

1. `dgl` 未安装
2. PyTorch 为 CPU 版
3. DBLP / 大规模异构图内存不足
4. 正式 5-seed 训练耗时长（DBLP LP 单 seed ~40–67 小时）

## 服务器环境参考（归档）

- `server_results/2026-06-02_/2026-06-02_env_snapshot.txt`
- 路径：`/home/mayq/ehgnn/EHGNN`，conda `ehgnn`
