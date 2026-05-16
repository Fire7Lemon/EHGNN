# GitHub 提交前检查清单

提交前自查清单；符合「大数据与本地实验产物不进仓库」的常见开源实践。

---

## 1. `git status` 前要检查什么

| 检查项 | 说明 |
|--------|------|
| 变更范围 | 是否仅为预期的代码 / 文档 / `.gitignore`，无误提交的 `.zip`、原始特征、`results/` 巨型 txt |
| 密钥 | 无 `.env`、token、私钥路径写入仓库 |
| IDE | `.idea/`、`.vscode/` 应由 `.gitignore` 忽略 |
| 子目录 | `Node Classification/results/`、`Link Prediction/results/`、**仓库根 `results/`**、`data/` 不应出现在暂存列表中 |
| 服务器日志 | **`logs/`**、**`*.log`** 已列入 `.gitignore`，勿强行 `-f` 提交 |
| 大文件 | `data/*.zip`、数据集目录若在仓库内且未被忽略，需移出或用 Git LFS（本仓库推荐：**保持 data 在盘、不进 Git**） |

---

## 2. 不应提交的内容

| 类别 | 示例 |
|------|------|
| 数据集 | `data/`、`*.zip`、大规模 `*.npy` / `*.npz` 等（当前 `.gitignore` 已覆盖多项格式） |
| 实验结果 | **`**/results/`**（含 **`Node Classification/results/`**、**`Link Prediction/results/`**、仓库根 **`results/`**，后者可由 **`collect_result_summaries.py`** 生成 **`ALL_SUMMARIES_INDEX.md`**） |
| Python 缓存 | `__pycache__/`、`*.pyc` |
| IDE | `.idea/`、`.vscode/` |
| 环境 | `.venv/`、`venv/`、`env/`、conda 环境目录本身 |
| 密钥与环境变量文件 | `.env` |
| 模型 checkpoint | `*.pt`、`*.pth`、`*.ckpt`、`checkpoints/`、`runs/`、`outputs/` |
| 日志 | `logs/`、`*.log` |

**注意**：根目录 `.gitignore` 含 **`*.csv`** / **`*.tsv`** 等——若在 **`results/` 之外**误放了 CSV，可能被全局忽略规则拦住或失误纳入提交；提交前务必 **`git status`**。

---

## 3. 应提交的内容

| 类别 | 示例 |
|------|------|
| 源代码 | `Node Classification/*.py`、`Link Prediction/*.py`、`MAG240M/*.py` |
| 脚本 | **`scripts/*.py`**、**`scripts/*.sh`** |
| TODO / 规划文档（可选一并纳入版本库） | **`scripts/TODO_metapath_experiments.md`**、**`scripts/TODO_hppr_strategy_study.md`** |
| 顶层说明 | **`README.md`**（原版）、**`README_MAG240M.md`** |
| 实验与结构文档 | **`PROJECT_STRUCTURE_AND_USAGE.md`**、**`EXPERIMENT_NOTES.md`**、**`REPRODUCTION_REPORT.md`**、**`GITHUB_READY_CHECKLIST.md`** |
| Git 忽略规则 | **`.gitignore`** |

**实验数值**：以 **`EXPERIMENT_NOTES.md`** / **`REPRODUCTION_REPORT.md`** 中记载的路径与快照为准；原始 **`results/`**、汇总索引 **`results/ALL_SUMMARIES_INDEX.md`** 保留在本地或服务器私有备份，**不提交**。

---

## 4. 建议提交命令（示例）

```bash
git status

git add README.md README_MAG240M.md \
  PROJECT_STRUCTURE_AND_USAGE.md EXPERIMENT_NOTES.md \
  REPRODUCTION_REPORT.md GITHUB_READY_CHECKLIST.md \
  .gitignore \
  "Node Classification/" "Link Prediction/" MAG240M/ scripts/

git commit -m "Reproduce EHGNN: scripts, docs, server runners"

git push origin <your-branch>
```

说明：

- 若使用 **`git add .`**，务必先确认 **`git status`** 中**没有** `data/`、任一 **`results/`**、`logs/`、checkpoint。  
- `git add` 路径可按实际分支调整；**不要**强行 `add -f results/` 除非明确要用 LFS 或极小摘要文件。

---

## 5. 推荐给老师的材料

| 材料 | 用途 |
|------|------|
| `REPRODUCTION_REPORT.md` | 复现目标、环境、流程、结果摘要与局限 |
| `EXPERIMENT_NOTES.md` | 具体数字、邻居策略历史、**执行状态 / smoke / 服务器计划** |
| `PROJECT_STRUCTURE_AND_USAGE.md` | 目录与命令手册（含 smoke vs 正式脚本区别） |
| `README.md` / `README_MAG240M.md` | 论文默认说明与 MAG240M 部署 |
| `scripts/TODO_*.md` | Fig.6 / Table IX / Table X 等阻塞项说明 |
