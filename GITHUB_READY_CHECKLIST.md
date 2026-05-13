# GitHub 提交前检查清单

提交前自查清单；符合「大数据与本地实验产物不进仓库」的常见开源实践。

---

## 1. `git status` 前要检查什么

| 检查项 | 说明 |
|--------|------|
| 变更范围 | 是否仅为预期的代码 / 文档 / `.gitignore`，无误提交的 `.zip`、原始特征、`results/` 巨型 txt |
| 密钥 | 无 `.env`、token、私钥路径写入仓库 |
| IDE | `.idea/`、`.vscode/` 应由 `.gitignore` 忽略 |
| 子目录 | `Node Classification/results/`、`data/` 不应出现在 `git status` 的暂存列表中（除非刻意要用 `git add -f`，一般不推荐） |
| 大文件 | `data/*.zip`、数据集目录若在仓库内且未被忽略，需移出或用 Git LFS（本仓库推荐：**保持 data 在盘、不进 Git**） |

---

## 2. 不应提交的内容

| 类别 | 示例 |
|------|------|
| 数据集 | `data/`、`*.zip`、大规模 `*.npy` / `*.npz` 等（当前 `.gitignore` 已覆盖多项格式） |
| 实验结果 | **`results/`**（规则 **`**/results/`** 会忽略各处的 `results` 目录） |
| Python 缓存 | `__pycache__/`、`*.pyc` |
| IDE | `.idea/`、`.vscode/` |
| 环境 | `.venv/`、`venv/`、`env/`、conda 环境目录本身 |
| 密钥与环境变量文件 | `.env` |
| 模型 checkpoint | `*.pt`、`*.pth`、`*.ckpt`、`checkpoints/`、`runs/`、`outputs/` |
| 日志 | `logs/`、`*.log` |

---

## 3. 应提交的内容

| 类别 | 示例 |
|------|------|
| 源代码 | `Node Classification/*.py`、`Link Prediction/*.py`、`MAG240M/*.py`、`scripts/*` |
| 顶层说明 | **`README.md`**（原版）、**`README_MAG240M.md`** |
| 本次新增说明 | **`PROJECT_STRUCTURE_AND_USAGE.md`**、**`EXPERIMENT_NOTES.md`**、**`REPRODUCTION_REPORT.md`**、**`GITHUB_READY_CHECKLIST.md`** |
| Git 忽略规则 | **`.gitignore`** |

**实验数值**：以 **`EXPERIMENT_NOTES.md`** / **`REPRODUCTION_REPORT.md`** 中的表格为准；原始 `results/` 保留在本地或私有备份。

---

## 4. 建议提交命令（示例）

```bash
git status

git add README.md README_MAG240M.md \
  PROJECT_STRUCTURE_AND_USAGE.md EXPERIMENT_NOTES.md \
  REPRODUCTION_REPORT.md GITHUB_READY_CHECKLIST.md \
  .gitignore \
  Node Classification/ Link Prediction/ MAG240M/ scripts/

git commit -m "Reproduce EHGNN PubMed experiments and add MAG240M deployment support"

git push origin <your-branch>
```

说明：

- 若使用 **`git add .`**，务必先确认 **`git status`** 中**没有** `data/`、`results/`、checkpoint。  
- `git add` 路径可按实际分支调整；**不要**强行 `add -f results/` 除非明确要用 LFS 或极小摘要文件。

---

## 5. 推荐给老师的材料

| 材料 | 用途 |
|------|------|
| `REPRODUCTION_REPORT.md` | 复现目标、环境、流程、结果摘要与局限 |
| `EXPERIMENT_NOTES.md` | PubMed 具体数字与邻居策略结论 |
| `PROJECT_STRUCTURE_AND_USAGE.md` | 目录与命令手册 |
| `README.md` / `README_MAG240M.md` | 论文默认说明与 MAG240M 部署 |
