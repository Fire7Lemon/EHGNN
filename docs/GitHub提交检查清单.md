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
| Conda / venv | **`environment.yml`**（若已跟踪）会随 clone 到达本地，但 **conda 环境目录**（如 **`~/miniconda3/envs/ehgnn/`**）**不在**仓库内；clone 后须按 **`docs/环境配置说明.md`** 自行 **`conda env create`** 或等价步骤创建环境 |
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
| 环境目录与解释器缓存 | **`.venv/`**、**`venv/`**、**`env/`**；本机或服务器上的 **conda 环境目录**（含各 **`envs/`** 下具体环境）；**`miniconda3/`** / **`anaconda3/`** 等安装根目录及其 **`site-packages`**（勿将整个 conda 安装复制进仓库） |
| 密钥与环境变量文件 | `.env` |
| 模型 checkpoint | `*.pt`、`*.pth`、`*.ckpt`、`checkpoints/`、`runs/`、`outputs/` |
| 日志 | `logs/`、`*.log` |

**注意**：根目录 `.gitignore` 含 **`*.csv`** / **`*.tsv`** 等——若在 **`results/` 之外**误放了 CSV，可能被全局忽略规则拦住或失误纳入提交；提交前务必 **`git status`**。

---

## 3. 应提交的内容

| 类别 | 示例 |
|------|------|
| 源代码 | `Node Classification/*.py`、`Link Prediction/*.py`、`MAG240M/*.py` |
| 脚本与辅助说明 | **`scripts/*.py`**、**`scripts/*.sh`**、**`scripts/scripts目录说明.md`**（`scripts/` 目录索引说明） |
| TODO / 规划文档（可选一并纳入版本库） | **`docs/元路径实验待办.md`**、**`docs/HPPR策略研究待办.md`** |
| 顶层说明 | **`README.md`**（根目录原版）；其余中文说明见 **`docs/`**（含 **`MAG240M运行说明.md`** 等） |
| 环境与数据（clone 后必读） | **`docs/环境配置说明.md`**（conda / CUDA / PyTorch-DGL / **`data/`、`results/` 不同步**） |
| 实验与结构文档 | **`docs/项目结构与使用说明.md`**、**`docs/实验记录.md`**、**`docs/论文复现报告.md`**、**`docs/GitHub提交检查清单.md`** |
| 环境与依赖规格（参考） | **`environment.yml`**（conda 环境规格参考，**应提交**）；**`requirements-freeze.txt`**（本机 pip freeze 版本快照，**可提交**作对照，仍须按服务器 CUDA 选型安装 GPU 相关包） |
| Git 忽略规则 | **`.gitignore`** |

**实验数值**：以 **`docs/实验记录.md`** / **`docs/论文复现报告.md`** 中记载的路径与快照为准；原始 **`results/`**、汇总索引 **`results/ALL_SUMMARIES_INDEX.md`** 保留在本地或服务器私有备份，**不提交**。

---

## 4. 中文文档与产物提交原则（同步更新）

- **应提交**：**`docs/`** 目录下说明类 **`*.md`**（含 **`MAG240M运行说明.md`**、**`环境配置说明.md`**、**`元路径实验待办.md`**、**`HPPR策略研究待办.md`** 等）；**`scripts/scripts目录说明.md`**；根目录 **`README.md`**；**`scripts/`** 下 **`*.py`**、**`*.sh`**（见 §3）。
- **应提交**：**`environment.yml`**（conda 环境规格参考，便于他人在服务器上 **`conda env create -f environment.yml`**）。
- **可以提交**：**`requirements-freeze.txt`**（本机 **`ehgnn`** 环境的 pip 版本快照，便于排查环境差异）；**不应**视为在服务器上 **`pip install -r`** 的唯一依据，尤其 **PyTorch / DGL / torch-scatter** 须按目标机 CUDA 与官方 wheel 重新选型（详见 **`docs/环境配置说明.md`** 中 **`environment.yml` 与 requirements-freeze.txt 的作用**）。
- **不要提交**：本机或服务器上的 **conda 环境目录本身**（不在仓库内；clone 不会复制已安装环境）。
- **不要提交**：**`.venv/`**、**`venv/`**、**`env/`**，以及误拷贝进仓库的 **`miniconda3/`** / **`anaconda3/`** 安装树、任意 **`site-packages`** 路径。
- **不要提交**：**`data/`**、任一 **`results/`**、**`logs/`**、**`*.log`**。
- **不要提交**：模型权重与 checkpoint：**`*.pt`**、**`*.pth`**、**`*.ckpt`**、`checkpoints/` 等（见 §2）。

---

## 5. 建议提交命令（示例）

```bash
git status

git add README.md docs/ environment.yml requirements-freeze.txt \
  .gitignore \
  "Node Classification/" "Link Prediction/" MAG240M/ scripts/

git commit -m "Reproduce EHGNN: scripts, docs, server runners"

git push origin <your-branch>
```

说明：

- 若使用 **`git add .`**，务必先确认 **`git status`** 中**没有** `data/`、任一 **`results/`**、`logs/`、checkpoint。  
- `git add` 路径可按实际分支调整；**不要**强行 `add -f results/` 除非明确要用 LFS 或极小摘要文件。

---

## 6. 推荐给老师的材料

| 材料 | 用途 |
|------|------|
| `docs/论文复现报告.md` | 复现目标、环境、流程、结果摘要与局限 |
| `docs/实验记录.md` | 具体数字、邻居策略历史、**执行状态 / smoke / 服务器计划** |
| `docs/项目结构与使用说明.md` | 目录与命令手册（含 smoke vs 正式脚本区别） |
| `README.md` / `docs/MAG240M运行说明.md` | 论文默认说明与 MAG240M 部署 |
| `docs/环境配置说明.md` | Clone 后环境与数据准备（**非** conda/data/results 提交项） |
| `docs/元路径实验待办.md`、`docs/HPPR策略研究待办.md` | Fig.6 / Table IX / Table X 等阻塞项说明 |
