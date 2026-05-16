# TODO：Meta-path 相关实验（Fig.6、Table IX）

**分支：`reproduce-baseline`。** 本轮 **不修改模型结构**；以下内容为后续最小补丁方案说明。

---

## 1. Fig.6：meta-path 数量分析（NC + LP）

**论文意图（概括）**：考察使用不同数量的 meta-path 时模型性能变化；可能包含随机选取子集、报告上下界等。

**当前代码能力**

- `Node Classification/main.py`：PubMed / DBLP 的 meta-path 列表在源码中 **固定**，CLI **无**「选用前 k 条 meta-path」或「随机子集」开关。
- `Node Classification/main_yelp.py`：business / phrase 两套路径固定。
- `Link Prediction/main.py`、`Link Prediction/main_yelp.py`：同理，meta-path 在文件中写死。

**是否已有现成接口**

- **无**。当前不支持在不改 Python 源码列表的情况下切换 meta-path 子集。

**下一轮最小补丁建议（不本轮实施）**

1. 增加可选 CLI，例如 `--metapath_indices 0,1,2` 或 `--num_metapaths 3`（随机种子固定时可复现随机子集）。
2. 训练脚本层循环：对每个数据集、每个 meta-path 数量配置、每个 seed 调用入口并写日志。
3. 保持随机子集规则与论文描述对齐（若论文给出具体采样协议则实现同一协议）。

---

## 2. Table IX：meta-path 权重分析（NC：PubMed / Yelp / DBLP）

**论文意图**：展示训练后各 meta-path 的 **softmax 权重**（或等价归一化权重）。

**当前代码能力**

- `EHGNN` 中存在 meta-path 可学习权重（与 `--wo_mweight` 消融相关），但 **未见** 训练结束后将 `softmax(meta_weights)` 打印到 stdout 或写入文件的统一接口。

**是否已有现成接口**

- **无** 专门导出 Table IX 的脚本或 CLI。

**下一轮最小补丁建议**

1. 在 **不改变 forward 数学形式** 前提下，在 `main.py` / `main_yelp.py` 训练结束后增加一行调试导出（例如打印 named meta-path 与对应 softmax 权重），或通过 `--dump_mweights path.json` 写入文件。
2. 批量脚本仅负责多 seed、多数据集调用与汇总。

---

## 3. 本轮结论

- Fig.6 / Table IX **无法在「零模型改动」前提下** 用统一脚本完整自动化；已用本文档标记 **TODO / Unsupported（自动化层面）**。
- 允许的手工近似：临时注释 meta-path 列表做小规模试验 **不属于** 本轮交付的规范化脚本（且易与复现口径冲突）。
