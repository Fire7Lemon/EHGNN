# Reproduction Report — EHGNN

---

## 1. 论文名称

**Efficient Learning for Billion-scale Heterogeneous Information Networks**（仓库 `README.md` 所述实现）。

---

## 2. 复现目标

- 在 **PubMed 节点分类** 任务上，按 `README.md` 给出的默认超参运行 **`Node Classification/main.py`**，并记录测试集 Macro-F1 / Micro-F1。  
- 扩展实验：**多 seed、消融、RW 邻居选择策略（freq / random / hybrid / temp）**，结果保存在本地 `results/`（默认不提交 Git，数值见 **`EXPERIMENT_NOTES.md`**）。  
- **MAG240M**：补充服务器侧部署说明（`README_MAG240M.md`、`scripts/`），便于在具备磁盘与内存的环境拉起训练。

---

## 3. 实验环境（记录模板）

以下为本仓库运行所需的典型依赖类别（**具体版本随机器 CUDA / PyTorch 轮子而变化**，需在目标环境中确认三角兼容）：

| 组件 | 说明 |
|------|------|
| Python | 建议 3.10（参见 `README_MAG240M.md` 示例） |
| PyTorch | 与 CUDA 匹配的安装 |
| DGL | 与 PyTorch / CUDA 匹配的安装 |
| 其它 | `numpy`、`scipy`、`scikit-learn`、`ogb`；NC 模型依赖 **`torch_scatter`** |

**注意**：未在本报告中锁定某一固定 CUDA / torch / dgl 版本号；提交给老师或协作者时，建议在 PR / issue 或附录中补充 **`pip freeze` 或 `conda export`** 片段。

---

## 4. 数据集来源与放置路径

| 数据集 | 来源 | 放置 |
|--------|------|------|
| PubMed / DBLP / Yelp | 根目录 **`README.md`** 中 Google Drive 链接 | 解压至 **`EHGNN/data/<DatasetName>/`**；运行时 **`Node Classification`** 下常用 **`--path ../data/`** |
| MAG240M | OGB-LSC（见 **`README_MAG240M.md`**） | 父目录由 **`MAG240MDataset(root=...)`** 指定；数据在 **`{root}/mag240m_kddcup2021/`** |

---

## 5. PubMed Node Classification 复现流程

1. 创建并激活 Conda / venv，安装 PyTorch、DGL、`torch_scatter`（版本匹配）、`ogb` 等。  
2. 将 PubMed 数据放入 **`data/PubMed/`**（含 `node.dat`、`link.dat`、`label.dat` 等，与 `utils.load_PubMed` 一致）。  
3. 执行：

```bash
cd Node Classification
python main.py --dataset PubMed --path ../data/ --seed 42
```

4. 批量复现：`run_pubmed_5seeds.py`、`run_pubmed_ablation.py`、`run_pubmed_neighbor_strategy.py`（见 **`PROJECT_STRUCTURE_AND_USAGE.md`**）。

---

## 6. 主要结果表格（摘自本地 `results/`）

详细数值与引用路径见 **`EXPERIMENT_NOTES.md`**。摘要：

| 实验 | 要点 |
|------|------|
| 5-seed | `best_test_macro` mean ± std ≈ **0.6199 ± 0.0315**（5 个 seed） |
| 消融（3 seeds） | Full EHGNN Macro ≈ **0.6032 ± 0.0107**；Random Neighbor Macro ≈ **0.6227 ± 0.0099** |
| Neighbor sweep（3 seeds） | freq mean Macro ≈ **0.6019**；random ≈ **0.6107** |

---

## 7. 与论文结果对比

- 根目录 **`README.md`** **未嵌入** PubMed 等指标的具体数值表，仅给出超参建议。  
- **定量对比**需要对照论文 PDF / 官方补充材料中的表格，并注意：划分、预处理、随机种子、评测脚本是否与本文仓库 **完全一致**。  
- 本仓库在 PubMed 上额外观察到 **Random / 非纯频次邻居** 可能优于 **freq Top-K**（见 **`EXPERIMENT_NOTES.md`**），属于 **复现过程中的扩展发现**，**不等于论文原文结论的重述**。

---

## 8. 复现过程中已解决的问题（工程记录）

以下为项目推进中常见踩坑与对应处理方向（**不包含本轮对源码的修改记录**；具体修复以各次 commit / 本地日志为准）：

| 主题 | 说明 |
|------|------|
| DGL | 与 PyTorch / CUDA 版本对齐安装，避免 import / CUDA error |
| `torch_scatter` | NC 中 `models.py` 依赖 scatter；需安装与 torch 版本匹配的轮子 |
| NumPy ABI | 二进制扩展（如 scipy / dgl）与 NumPy 版本不匹配时的 **`numpy.dtype size changed`** 类错误；通过统一虚拟环境与版本约束缓解 |
| PubMed ID mapping | `load_PubMed` 中对全局 id 与类型块内紧凑 id 的映射（详见 **`Node Classification/utils.py`**）；缺失或错误会导致标签与特征错位 |
| SciPy sparse 索引 | 稀疏矩阵索引类型需与构建 CSR/CSC 时代码一致，避免隐式类型警告或索引错误 |
| DGL `random_walk` 与 seed | 节点 ID、`numpy`/`torch` 随机种子与 DGL 行为需在论文对齐实验中固定；注意种子参数类型与设备一致性 |

---

## 9. 当前局限

| 项目 | 说明 |
|------|------|
| MAG240M | **完整训练 / 全量数据**依赖大规模磁盘与内存；本仓库以文档与脚本支持为主，**是否在目标机器完成端到端训练需单独确认** |
| DBLP / Yelp | **未完成与 PubMed 同规格的邻居策略对比验证** |
| Link Prediction | 未在本报告中汇总系统化指标 |
| 论文逐项对齐 | 若课程或审稿要求严格对齐，需逐项核对论文附录中的实现细节与评测协议 |

---

## 10. 文档与代码变更边界

- **`REPRODUCTION_REPORT.md`、`EXPERIMENT_NOTES.md`、`PROJECT_STRUCTURE_AND_USAGE.md`、`GITHUB_READY_CHECKLIST.md`**：面向提交与老师阅读的说明材料。  
- **本轮文档撰写不改变训练逻辑**；若发现 CLI 与代码不一致等问题，仅在本报告 **`PROJECT_STRUCTURE_AND_USAGE.md`** 等处文字记录，**不在此轮修改 Python**。
