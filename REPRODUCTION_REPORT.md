# 复现报告 — EHGNN

**Git 分支：`reproduce-baseline`。** 节点分类入口 **`Node Classification/main.py`** 仅保留论文口径：**RW 后为频次 Top-K**（`Counter.most_common(K)`）与 **`--r_neighbor` 随机邻居消融**。邻居策略扩展实验代码与脚本见 **`neighbor-strategy-dev`**。

---

## 1. 论文名称

**Efficient Learning for Billion-scale Heterogeneous Information Networks**（仓库 `README.md` 所述实现）。

---

## 2. 复现目标

- 在 **PubMed / DBLP** 节点分类上按 `README.md` 或 README 对齐命令运行 **`Node Classification/main.py`**，记录测试集 Macro-F1 / Micro-F1；**Yelp** 走 **`main_yelp.py`**（多标签），selected **3-seed** 汇总见 **`EXPERIMENT_NOTES.md`**（`results/yelp_3seeds/summary.txt`）。  
- PubMed：**多 seed**（`run_pubmed_5seeds.py`）、**消融**（`run_pubmed_ablation.py`）。DBLP：**selected 3-seed**（`run_dblp_5seeds.py`）。Yelp：**selected 3-seed**（`run_yelp_3seeds.py`）；测试指标跨 seed **完全一致**等现象见 **`EXPERIMENT_NOTES.md`** §4，**不得**解读为强稳定性。  
- **MAG240M**：服务器侧部署说明（`README_MAG240M.md`、`scripts/`）。  
- **不在本分支**：hybrid / temp / `neighbor_strategy` 扫描；相关内容仅在 **`neighbor-strategy-dev`** 与 **`EXPERIMENT_NOTES.md`** 历史小节。

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

## 5. Node Classification 复现流程

### 5.1 PubMed

1. 创建并激活 Conda / venv，安装 PyTorch、DGL、`torch_scatter`（版本匹配）、`ogb` 等。  
2. 将 PubMed 数据放入 **`data/PubMed/`**（含 `node.dat`、`link.dat`、`label.dat` 等，与 `utils.load_PubMed` 一致）。  
3. 执行：

```bash
cd Node Classification
python main.py --dataset PubMed --path ../data/ --seed 42
```

4. 批量复现：`run_pubmed_5seeds.py`、`run_pubmed_ablation.py`。

### 5.2 DBLP（README 超参；不写 `pubmed_nc_result.txt`）

1. 将 DBLP 数据放入 **`data/DBLP/`**（与 `utils.load_dblp` 一致）。  
2. README 表格对齐的单次示例（训练阶段计时见 `Total training time` 行；**数据加载与 RW 预计算另计且通常远长于 PubMed**）：

```bash
cd Node Classification
python main.py --dataset DBLP --path ../data/ --seed <seed> \
  --lr 5e-4 --dropout 0.5 --hidden 512 --n_layers 5 --batch_size 3000 --alpha 0.7 --K 20
```

3. 多 seed 汇总（脚本从 stdout 写 `results/dblp_5seeds/`）：`run_dblp_5seeds.py`（支持 `--seeds`、`--skip_existing`）。当前文档记载的 **selected 3-seed** 数值见 **`EXPERIMENT_NOTES.md`** §3。

### 5.3 Yelp（README 超参；`main_yelp.py`）

1. 将 Yelp 数据放入 **`data/Yelp/`**（与 `utils.load_Yelp` 一致）。  
2. README 对齐批量脚本：`run_yelp_3seeds.py`（默认或 `--seeds`；可选 `--skip_existing`），输出 **`results/yelp_3seeds/`**。单次示例：

```bash
cd Node Classification
python main_yelp.py --dataset Yelp --path ../data/ --seed <seed> \
  --alpha 0.7 --dropout 0.5 --K 20 --lr 0.0003 --hidden 256 --n_layers 4 --batch_size 3000
```

3. **selected 3-seed** 数值与 **测试 F1 跨 seed 完全一致等现象**见 **`EXPERIMENT_NOTES.md`** §4。

---

## 6. 主要结果表格（摘自本地 `results/`）

详细数值与引用路径见 **`EXPERIMENT_NOTES.md`**。摘要：

| 实验 | 要点 |
|------|------|
| PubMed 5-seed | `best_test_macro` mean ± std ≈ **0.6199 ± 0.0315**（5 个 seed，见 `EXPERIMENT_NOTES.md`） |
| DBLP selected **3** seeds | seeds [42, 3407, 2026]；`best_test_macro` **0.150100 ± 0.010967**；`best_test_micro` **0.387500 ± 0.026121**；`final_test_macro` **0.151867 ± 0.017625**；`final_test_micro` **0.396000 ± 0.035596**；平均训练段 **10.1355 s**（`results/dblp_5seeds/summary.txt`，**EXPERIMENT_NOTES.md** §3）。 |
| Yelp selected **3** seeds | seeds [42, 3407, 2026]；`best/final_test_macro` **0.666300 ± 0.000000**；`best/final_test_micro` **0.875800 ± 0.000000**；平均训练 **964.1782 s**；best/worst seed 汇总均为 **42**（因三 seed 测试 F1 **相同**）。来源 `results/yelp_3seeds/summary.txt`；**sigmoid+BCE**、**0.5 阈值二值化**；训练过程不同而测试指标相同——**不应**将 std=0 当作强稳定性，见 **`EXPERIMENT_NOTES.md`** §4。 |
| 消融（3 seeds） | Full EHGNN Macro ≈ **0.6032 ± 0.0107**；Random Neighbor（`--r_neighbor`）Macro ≈ **0.6227 ± 0.0099** |
| 优化分支历史 sweep | 见 **`EXPERIMENT_NOTES.md`** §6（`neighbor-strategy-dev`，非本分支可运行脚本） |

---

## 7. 与论文结果对比

- 根目录 **`README.md`** **未嵌入** PubMed 等指标的具体数值表，仅给出超参建议。  
- **定量对比**需要对照论文 PDF / 官方补充材料中的表格，并注意：划分、预处理、随机种子、评测脚本是否与本文仓库 **完全一致**。  
- 消融中 **`--r_neighbor`** 相对默认频次 Top-K 的提升及优化分支下的 sweep 数值，见 **`EXPERIMENT_NOTES.md`**；**不等于论文原文对默认方法的结论重述**。

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
| DBLP | 已有 **selected 3-seed** baseline（§3）；满 5-seed 可后续补跑 |
| Yelp | 已有 **selected 3-seed** baseline（§4）；测试 **Macro/Micro 跨 seed 数值完全相同**，见 §4 **异常现象**——需区分记录事实与稳定性推断 |
| Link Prediction | 未在本报告中汇总系统化指标 |
| 论文逐项对齐 | 若课程或审稿要求严格对齐，需逐项核对论文附录中的实现细节与评测协议 |

---

## 10. 文档与代码变更边界

- **`REPRODUCTION_REPORT.md`、`EXPERIMENT_NOTES.md`、`PROJECT_STRUCTURE_AND_USAGE.md`、`GITHUB_READY_CHECKLIST.md`**：面向提交与老师阅读的说明材料。  
- **`reproduce-baseline`**：`Node Classification/main.py` 与 **`utils.py`** 以论文默认 RW–频次 Top-K 为准；模型 **`models.py`** 结构保持不变。邻居策略扩展仅在 **`neighbor-strategy-dev`**。
