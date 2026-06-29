# P5 PPR-TopK-lite Design & Demo

> 状态：**Code Prepared** — 待服务器运行  
> 证据等级：**Prototype / Design demo**（不训练模型）

## 方法动机

EHGNN 用 meta-path 随机游走 + 落点频次估计 Top-K 邻居（HPPR 近似）。P5 探索 **确定性 sparse PPR（power iteration）** 作为替代，比较邻居集合重合度与耗时，为后续接入聚合留接口。

## 与原始 EHGNN RW Top-K 的区别

| 环节 | EHGNN RW | PPR-TopK-lite |
|------|----------|---------------|
| 扩散 | 随机游走 `walk_num` 次 | 行归一化转移矩阵 + power iteration |
| Top-K | Counter.most_common(K) | PPR score Top-K |
| 确定性 | 否（随机） | 是（给定 α、迭代次数） |
| 输出 | CSR 权重矩阵 | 同格式 CSR（`build_csr_from_topk`） |

## 代码复用审计

### 原始 RW Top-K 流程

**位置**：`Node Classification/utils.py` L491–567（LP 版类似 L393+）

1. `dgl.sampling.random_walk(g, nodes, metapath)` → 游走落点 multiset
2. `_pick_neighbors_from_rw_multiset(t_nodes, K)` → Top-K 节点 + 归一化权重
3. 写入 `scipy.sparse.csr_matrix` shape `(num_source, num_target_type)`

### 原始 Top-K 输出格式

- `sim_matrix[t_type]`：CSR `(num_s, num_t)`
- 非零项 = (source_row, neighbor_col, weight)
- `get_model_need` / `get_weights_sidx` 按 batch 取行

### PPR-TopK 可接入点

替换 `random_walk_sim` 内 **Top-K 生成** 步骤：

```text
metapath_adj → row_normalize → ppr_power_iteration(seed) → topk_from_scores → build_csr_from_topk
```

输出 CSR 可直接传入 EHGNN `forward` 的 scatter 路径（需按 t_type 分桶，demo 用同型 metapath `dad_r→dad`）。

### 最小接口设计

| 函数 | 输出 |
|------|------|
| `row_normalize_csr(adj)` | 转移矩阵 P |
| `ppr_power_iteration(P, seeds, α, iters)` | scores [batch, n] |
| `topk_from_scores` | indices, weights |
| `build_csr_from_topk` | CSR 兼容 EHGNN |

### 风险与不确定点

1. **异构 meta-path**：demo 用 `dgl.metapath_reachable_graph` 构图；长 meta-path 可能稀疏/稠密差异大
2. **非同型 metapath**（源/宿类型不同）需扩展为多 CSR 块
3. PPR-lite ≠ PPRGo push 算法；精度/效率未调优
4. 高 Jaccard 不保证下游 F1 提升
5. 本地无 DGL；demo 仅服务器

## PPR-TopK 接口设计

- `alpha`：restart probability（默认 0.15）
- `num_iters`：幂迭代步数（默认 10）
- 注释见 `ppr_topk_lite.py` 模块 docstring

## Demo 设计

- 数据集：PubMed
- Meta-path index 0：`['dad_r', 'dad']`
- 采样 **500** 个 disease 源节点
- 对比：PPR-TopK vs RW-TopK（`walk_num=100`, `K=20`）
- 指标：Jaccard、overlap@K、runtime

## 新增文件

```text
code/ppr_topk_lite.py
code/demo_pubmed_ppr_topk.py
code/compare_rw_vs_ppr_neighbors.py
code/parse_ppr_topk_results.py
code/plot_ppr_topk_results.py
configs/pubmed_ppr_topk_demo.yaml
server_scripts/run_p5_ppr_topk_demo.sh
```

## 运行方式

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh
```

## 服务器运行命令

见 `run_p5_ppr_topk_demo.sh`

## 当前状态

Code Prepared；`ppr_topk_lite.py` 可本地 import；demo 需 DGL+数据

## 预期输出

| 文件 | 内容 |
|------|------|
| `results/pubmed_ppr_topk_demo.csv` | 每节点 Jaccard |
| `results/pubmed_ppr_topk_demo_meta.json` | 汇总元数据 |
| `results/rw_vs_ppr_neighbor_overlap.csv` | 聚合 overlap |
| `results/ppr_topk_summary.csv` | parse 汇总 |
| `figs/rw_vs_ppr_topk_overlap.png` | mean Jaccard |
| `figs/ppr_runtime_demo.png` | PPR vs RW 耗时 |
| `figs/ppr_topk_jaccard_distribution.png` | 分布直方图 |

## 结论边界

**当前仅为 PPR-TopK 邻居选择替代方案的设计与小规模 demo 准备。它不训练模型，不代表已经完整替代 EHGNN 的随机游走 / HPPR 近似流程，也不能作为性能优化结论。**

## 风险与修复记录

- **2026-06-03 首次服务器运行失败**：`ModuleNotFoundError: No module named 'utils'`。
- **根因**：`demo_pubmed_ppr_topk.py` 将 `PROJECT_ROOT` 算成 `experiments/`（`parents[3]`），未正确指向 EHGNN 根目录下的 `Node Classification/utils.py`。
- **修复**：改用 `Path(__file__).resolve().parents[4]`；`if str(NC_DIR) not in sys.path: sys.path.insert(0, ...)`；`run_p5_ppr_topk_demo.sh` 增加 `export PYTHONPATH="$PROJECT_ROOT/Node Classification:..."`。
- **未修改** `Node Classification/` 正式主线代码。

- **2026-06-03 第二次服务器失败**：`FileNotFoundError: .../dataPubMed/node.dat` — `load_PubMed` 使用 `data_path + dataset` 拼接，路径必须以 `/` 结尾。已修复为 `resolve_project_data_path(PROJECT_ROOT)` → 绝对路径 `PROJECT_ROOT/data/`。

- **2026-06-03 第三次服务器失败**：`TypeError: DGLGraph.adj() got an unexpected keyword argument 'scipy_fmt'`。服务器 DGL 版本不支持旧 `adj(scipy_fmt=...)` API。已在 `ppr_topk_lite.py` 新增 `dgl_graph_to_scipy_csr()`，按序尝试 `adj_external` → `old_adj` → `edges_fallback`；demo 日志与 meta JSON 记录 `dgl_version` 与 `dgl_adj_conversion_mode`。

## 后续接入 EHGNN 说明

1. 在 `random_walk_sim` 处增加 `mode=ppr|rw` 分支
2. PPR 分支调用 `build_csr_from_topk` 输出 `sim_matrix[t_type]`
3. 下游 `get_model_need` / EHGNN.forward **无需改动**
