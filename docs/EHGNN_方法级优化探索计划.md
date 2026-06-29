# EHGNN 方法级优化探索计划

> 文件用途：本计划文档放入项目 `docs/` 目录，作为 Cursor 后续持续查阅和执行的方向说明。  
> 当前阶段目标不是完整 5-seed 证明某个优化一定有效，而是在一天内完成尽可能多的**方法级优化思路原型**，并统一保存代码、日志、表格、图片和分析文档，方便后续整理报告、PPT 和简历材料。

---

## 一、今日目标

本阶段工作目标：

1. 在 EHGNN 项目内完成多条**方法级优化思路**的原型尝试，而不是只做 `K / walk_num / alpha` 这类参数搜索。
2. 优先选择运行成本较低的数据集，例如：
   - PubMed Node Classification
   - PubMed Link Prediction
   - 已有 DBLP Link Prediction 日志分析
3. 不在今天启动 DBLP LP 或 MAG240M 这类高成本完整实验。
4. 每条思路都必须留下可复查证据：
   - 独立代码
   - 运行命令
   - 日志
   - CSV 表格
   - 图片
   - 简短分析报告
5. 所有今日产生的文件统一放到同一个实验根目录，禁止散落在项目各处。

---

## 二、给 Cursor 的核心要求

Cursor 在执行本计划时必须遵守以下要求：

### 必须遵守

1. **所有方法尝试代码都要保存好。**
2. **所有代码、日志、表格、图片、报告都必须保存在同一个项目内根路径下。**
3. **所有可能后续会用到的输出文件都必须集中归档，方便后续查看、复制、写报告和做 PPT。**
4. 原始复现主线代码尽量不动。
5. 如必须基于原脚本修改，应复制成实验专用脚本，不要直接覆盖正式复现脚本。
6. 每条方法必须有自己的 `code/`、`logs/`、`results/`、`figs/`、`report.md`。
7. 每次运行命令必须追加记录到 `COMMANDS.md`。
8. 每次运行结果必须写入 `RUN_REGISTRY.csv`。
9. 每条方法的论文来源、核心改动、结论边界必须写入 `METHOD_MANIFEST.csv`。
10. 所有结果必须区分：
    - 正式复现结果
    - 诊断结果
    - smoke 结果
    - 方法探索结果
    - 失败日志
11. 所有实验结论必须注明证据等级：
    - `Prototype`
    - `Quick Validation`
    - `Reliable Result`

### 禁止事项

1. 禁止删除已有正式复现日志。
2. 禁止覆盖 `server_results/` 中已有服务器归档。
3. 禁止把本地调试结果混进正式复现实验表。
4. 禁止把 smoke / diagnostic 结果当成正式 5-seed 结果。
5. 禁止把单 seed 结果写成“全面验证有效”。
6. 禁止把工程性改动夸大为论文级算法创新。
7. 禁止今天启动 DBLP LP 完整新实验或 MAG240M 完整实验，除非用户明确要求。

---

## 三、统一实验根目录

今日所有内容统一保存到：

```text
experiments/opt_20260629_method_exploration/
```

建议目录结构如下：

```text
experiments/opt_20260629_method_exploration/
  README.md
  PLAN.md
  PAPER_BASIS.md
  COMMANDS.md
  RUN_REGISTRY.csv
  METHOD_MANIFEST.csv

  00_existing_results/
    code/
    logs/
    results/
    figs/
    report.md

  01_sehgnn_lite_pubmed_nc/
    code/
    configs/
    logs/
    results/
    figs/
    report.md

  02_lp_pair_decoder_pubmed_lp/
    code/
    configs/
    logs/
    results/
    figs/
    report.md

  03_sampled_lp_training_pubmed_lp/
    code/
    configs/
    logs/
    results/
    figs/
    report.md

  04_distill_mlp_pubmed_nc/
    code/
    configs/
    logs/
    results/
    figs/
    report.md

  05_ppr_topk_lite_design/
    code/
    configs/
    logs/
    results/
    figs/
    report.md

  summary/
    all_results.csv
    all_runtime.csv
    all_figures_index.md
    optimization_day_summary.md
    optimization_findings_for_ppt.md
```

---

## 四、Git 分支建议

建议在正式复现分支之外新建探索分支：

```bash
git checkout -b opt-exploration-20260629
```

如果分支已存在，则继续使用该分支即可。

今日 commit message 使用中文，例如：

```bash
git add experiments/opt_20260629_method_exploration
git commit -m "新增EHGNN方法级优化探索实验"
```

注意：是否提交由用户决定。Cursor 不要未经确认自动提交。

---

## 五、今日方法路线总览

| 优先级 | 方法 | 任务 | 数据集 | 改变类型 | 今日目标 |
|---|---|---|---|---|---|
| P0 | 已有结果与瓶颈分析 | NC / LP | 已有日志 | 结果整理 | 必做 |
| P1 | SeHGNN-lite | 节点分类 | PubMed | 预计算特征 + 轻量分类头 | 必做 |
| P2 | LP Pair Decoder | 链路预测 | PubMed | 链路解码器增强 | 必做 |
| P3 | Sampled-LP Training | 链路预测 | PubMed | 采样式训练协议 | 推荐 |
| P4 | EHGNN-to-MLP Distillation | 节点分类 | PubMed | 蒸馏轻量学生模型 | 有时间做 |
| P5 | PPR-TopK-lite | 邻居选择 | PubMed 子图 | 确定性 PPR 替代 RW 设计 | 有时间至少做骨架 |

---

## 六、P0：已有结果与瓶颈分析

### 目标

先整理已有正式复现和 DBLP LP 日志，不跑新实验，形成优化动机证据。

### 输入来源

重点读取：

```text
server_results/
Link Prediction/results/dblp_lp_5seeds_val1000/logs/
已有上传或本地归档的 DBLP LP seed 日志
```

### 需要产出

保存到：

```text
experiments/opt_20260629_method_exploration/00_existing_results/
```

产物：

```text
results/dblp_lp_5seed_summary.csv
results/dblp_lp_runtime_breakdown.csv
results/core_reproduction_summary.csv

figs/dblp_lp_final_auc_by_seed.png
figs/dblp_lp_best_vs_final_auc.png
figs/dblp_lp_runtime_by_seed.png
figs/dblp_lp_runtime_breakdown.png

report.md
```

### 分析重点

1. DBLP LP 5-seed 是否完整。
2. 每个 seed 的：
   - Best AUC
   - Best AP
   - Final AUC
   - Final AP
   - Total training time
   - Done Load Data time
   - Done my sim time
3. DBLP LP 的主要瓶颈：
   - 训练 step 多
   - `evaluate_lp()` 开销大
   - random walk / sim 预计算开销大
4. 当前 DBLP LP 性能是否接近随机。
5. 为什么后续需要方法级优化探索。

---

## 七、P1：SeHGNN-lite，预计算特征 + 轻量分类头

### 思路来源

借鉴 SeHGNN / SIGN / NARS 一类方法的思想：  
先预计算多关系、多元路径、多跳邻居聚合特征，再用轻量分类器或融合头进行训练，降低训练阶段图传播负担。

### 与 EHGNN 的关系

EHGNN 本身已有“一次聚合”的思想。本尝试进一步将其改造成：

```text
原始 EHGNN：
元路径随机游走 / Top-K 邻居选择
→ 一次聚合
→ EHGNN 模型训练

SeHGNN-lite：
复用或导出聚合特征
→ 拼接 / 加权融合
→ 轻量 MLP 分类头
```

### 数据集

```text
PubMed Node Classification
seed = 42
```

时间允许时可补：

```text
seed = 3407, 2026
```

### 代码保存位置

```text
experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/
```

建议文件：

```text
run_pubmed_nc_sehgnn_lite.py
sehgnn_lite_model.py
extract_precomputed_features.py
train_fusion_mlp.py
parse_sehgnn_lite_log.py
```

### 输出保存位置

```text
experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/
```

产物：

```text
logs/sehgnn_lite_pubmed_seed42.log

results/pubmed_nc_sehgnn_lite_seed42.csv
results/pubmed_nc_sehgnn_lite_vs_ehgnn.csv

figs/sehgnn_lite_vs_ehgnn_macro_f1.png
figs/sehgnn_lite_vs_ehgnn_micro_f1.png
figs/sehgnn_lite_vs_ehgnn_runtime.png

report.md
```

### 记录指标

1. Macro-F1
2. Micro-F1
3. best epoch
4. total runtime
5. preprocess time
6. train time
7. parameter count
8. 与原 EHGNN PubMed NC seed=42 的对比

### 结论边界

可以写：

```text
借鉴 SeHGNN 的预计算聚合思想，构建了 EHGNN-SeHGNN-lite 原型，并在 PubMed 节点分类上完成单 seed 初步验证。
```

不能写：

```text
该方法已全面优于 EHGNN。
```

---

## 八、P2：LP Pair Decoder，链路预测解码器增强

### 思路来源

借鉴 SEAL / BUDDY / ELPH 等链路预测研究的启发：  
链路预测不一定只依赖两个节点 embedding 的点积，也可以显式建模节点对之间的 pair-wise 表征。

### 与 EHGNN 的关系

原始 EHGNN 链路预测可能类似：

```text
score(u, v) = dot(h_u, h_v)
```

尝试改成：

```text
score(u, v) = MLP([h_u, h_v, |h_u - h_v|, h_u * h_v])
```

可选增强：

```text
score(u, v) = MLP([h_u, h_v, |h_u - h_v|, h_u * h_v, degree_u, degree_v])
```

### 数据集

```text
PubMed Link Prediction
seed = 42
```

### 对比组

| 方法 | 说明 |
|---|---|
| dot decoder | 原始点积解码器 |
| pair-mlp decoder | 拼接、差值、Hadamard 乘积 |
| pair-mlp-degree decoder | 加入简单结构特征，如 degree |

### 代码保存位置

```text
experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/
```

建议文件：

```text
main_pubmed_lp_pair_decoder.py
lp_pair_decoder_model.py
run_pair_decoder_pubmed.py
parse_pair_decoder_results.py
```

### 输出保存位置

```text
experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/
```

产物：

```text
logs/pubmed_lp_pair_decoder_seed42.log

results/pubmed_lp_pair_decoder_compare.csv

figs/pair_decoder_auc_compare.png
figs/pair_decoder_ap_compare.png
figs/pair_decoder_runtime_compare.png

report.md
```

### 记录指标

1. AUC
2. AP
3. best epoch
4. final AUC / AP
5. runtime
6. 与原 PubMed LP seed=42 的对比

### 结论边界

可以写：

```text
针对 EHGNN 链路预测中点积解码器表达力有限的问题，尝试引入 pair-wise MLP decoder，对源节点和目标节点 embedding 的拼接、差值与 Hadamard 乘积进行建模。
```

不能写：

```text
该方法已经解决 DBLP LP 性能偏低问题。
```

---

## 九、P3：Sampled-LP Training，采样式链路预测训练协议

### 思路来源

借鉴 GraphSAINT / Cluster-GCN / HGSampling 类方法的训练策略变化：  
不要在每个 epoch 中完整覆盖所有训练样本，而是通过子图或边采样降低每轮训练成本。

### 与 EHGNN 的关系

原始链路预测训练：

```text
每个 epoch 遍历较大规模训练节点 / 边
```

尝试版：

```text
每个 epoch 只采样一部分正负边训练
观察速度与 AUC/AP 的 trade-off
```

### 数据集

```text
PubMed Link Prediction
seed = 42
```

### 对比组

```text
full training
sample_ratio = 0.25
sample_ratio = 0.50
sample_ratio = 0.75
```

如时间不足，可只做：

```text
sample_ratio = 0.25
sample_ratio = 0.50
```

也可以先用较少 epoch 快速验证，例如：

```text
epochs = 30 或 50
```

但必须在 report 中写明这是 quick validation，不是正式对比。

### 代码保存位置

```text
experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/
```

建议文件：

```text
main_pubmed_lp_sampled_training.py
sampled_edge_loader.py
run_sampled_lp_pubmed.py
parse_sampled_lp_results.py
```

### 输出保存位置

```text
experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/
```

产物：

```text
logs/pubmed_lp_sampled_training_seed42.log

results/pubmed_lp_sampled_training.csv

figs/sampled_lp_auc_tradeoff.png
figs/sampled_lp_ap_tradeoff.png
figs/sampled_lp_runtime_tradeoff.png
figs/sampled_lp_pareto_auc_runtime.png

report.md
```

### 结论边界

可以写：

```text
尝试将链路预测训练从全量遍历改为采样式训练，初步观察速度与 AUC/AP 的 trade-off。
```

不能写：

```text
采样式训练与论文默认训练协议完全等价。
```

---

## 十、P4：EHGNN-to-MLP Distillation，蒸馏轻量学生模型

### 思路来源

借鉴 GLNN 一类 GNN-to-MLP 蒸馏方法：  
用 GNN teacher 的 logits 或 soft labels 训练 MLP student，使推理阶段摆脱图结构依赖。

### 与 EHGNN 的关系

```text
EHGNN teacher
→ soft labels / logits
→ MLP student
→ 快速推理
```

这个方向主要优化推理成本，不一定降低 teacher 训练成本。

### 数据集

```text
PubMed Node Classification
seed = 42
```

### 代码保存位置

```text
experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/
```

建议文件：

```text
save_teacher_logits_pubmed.py
train_mlp_student.py
distillation_losses.py
run_distill_pubmed.py
parse_distill_results.py
```

### 输出保存位置

```text
experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/
```

产物：

```text
logs/mlp_student_seed42.log

results/pubmed_nc_distill_mlp.csv

figs/distill_teacher_student_f1.png
figs/distill_inference_time.png
figs/distill_train_time.png

report.md
```

### 结论边界

可以写：

```text
借鉴 GNN-to-MLP 蒸馏思想，探索将 EHGNN 的图依赖知识迁移到轻量 MLP 学生模型中，以降低推理复杂度。
```

不能写：

```text
学生模型已经完全替代 EHGNN。
```

---

## 十一、P5：PPR-TopK-lite，确定性 PPR 替代随机游走设计

### 思路来源

借鉴 PPRGo / APPNP 的近似 PPR 扩散思想。  
EHGNN 当前代码中没有显式 `HPPR()` 函数，真实实现更像：

```text
meta-path random walk
→ 落点频次统计
→ Top-K 邻居
```

本方向尝试设计：

```text
meta-path adjacency
→ sparse transition matrix
→ approximate PPR / power iteration
→ Top-K 邻居
```

### 今日目标

今天不强求完整替换 EHGNN 的随机游走流程，至少完成：

1. 设计文档；
2. 小规模代码骨架；
3. PubMed 子图 demo；
4. RW Top-K 与 PPR Top-K 的邻居重合度比较。

### 代码保存位置

```text
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/
```

建议文件：

```text
ppr_topk_lite.py
demo_pubmed_ppr_topk.py
compare_rw_vs_ppr_neighbors.py
```

### 输出保存位置

```text
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/
```

产物：

```text
results/rw_vs_ppr_neighbor_overlap.csv

figs/rw_vs_ppr_topk_overlap.png
figs/ppr_runtime_demo.png

report.md
```

### 结论边界

可以写：

```text
设计了确定性 PPR-TopK 替代随机游走邻居估计的原型方案，并在小规模样本上比较 RW Top-K 与 PPR Top-K 的邻居重合度。
```

不能写：

```text
PPR-TopK 已经完整替代 EHGNN 的 HPPR / RW 流程。
```

---

## 十二、今日时间安排建议

### 第 0 阶段：0～1 小时

建立统一实验根目录和记录文件。

必须完成：

```text
README.md
PLAN.md
PAPER_BASIS.md
COMMANDS.md
RUN_REGISTRY.csv
METHOD_MANIFEST.csv
```

同时记录当前 Git 状态：

```bash
git status
git branch --show-current
git rev-parse HEAD
```

写入根目录 `README.md`。

---

### 第 1 阶段：1～2 小时

完成 P0：

```text
00_existing_results/
```

目标：

1. 解析已有 DBLP LP 5-seed 日志；
2. 生成 summary CSV；
3. 生成 3～4 张图；
4. 写 `report.md`。

---

### 第 2 阶段：2～5 小时

完成 P1：

```text
01_sehgnn_lite_pubmed_nc/
```

目标：

1. 保存实验代码；
2. 跑 PubMed NC seed=42；
3. 生成 CSV；
4. 生成 F1 / runtime 对比图；
5. 写 `report.md`。

---

### 第 3 阶段：5～8 小时

完成 P2：

```text
02_lp_pair_decoder_pubmed_lp/
```

目标：

1. 保存实验代码；
2. 跑 PubMed LP seed=42；
3. 生成 AUC / AP 对比表；
4. 生成图片；
5. 写 `report.md`。

---

### 第 4 阶段：8～10 小时

尽量完成 P3：

```text
03_sampled_lp_training_pubmed_lp/
```

目标：

1. 完成 sampled training 代码；
2. 至少跑 `sample_ratio=0.25` 和 `0.50`；
3. 生成速度-性能 trade-off 表；
4. 生成图片；
5. 写 `report.md`。

---

### 第 5 阶段：10～12 小时

整理所有成果。

必须生成：

```text
summary/all_results.csv
summary/all_runtime.csv
summary/all_figures_index.md
summary/optimization_day_summary.md
summary/optimization_findings_for_ppt.md
```

如果仍有时间，再做 P4 或 P5 的代码骨架和 report。

---

## 十三、统一记录文件格式

### RUN_REGISTRY.csv

字段：

```text
run_id
method
task
dataset
seed
script
command
status
start_time
end_time
runtime_sec
metric_primary
metric_secondary
log_path
result_csv
figure_dir
note
```

### METHOD_MANIFEST.csv

字段：

```text
method_id
method_name
paper_basis
core_change
dataset
task
code_dir
result_dir
figure_dir
status
claim_boundary
```

### summary/all_results.csv

字段：

```text
method
task
dataset
seed
macro_f1
micro_f1
auc
ap
best_epoch
final_metric
runtime_sec
preprocess_time_sec
train_time_sec
eval_time_sec
status
note
```

### summary/all_figures_index.md

每张图记录：

```text
- 图名：
- 路径：
- 对应方法：
- 数据集：
- 图中指标：
- 可用于 PPT 哪一页：
- 注意事项：
```

---

## 十四、今日结果的证据等级

每个结果必须标注证据等级：

| 等级 | 定义 | 可写法 |
|---|---|---|
| Prototype | 代码原型完成，未完整运行 | “实现了原型” |
| Quick Validation | 单 seed 或少量 epoch 跑通 | “完成初步验证” |
| Reliable Result | 多 seed / 完整 epoch 验证 | “结果表明” |

今日大多数实验预计属于：

```text
Prototype 或 Quick Validation
```

不能写成最终结论。

---

## 十五、今日不做内容

| 内容 | 原因 |
|---|---|
| DBLP LP 新方法完整训练 | 单 seed 成本过高，不适合一天内多尝试 |
| MAG240M 完整复现或优化 | 数据、内存、磁盘和时间风险高 |
| DDP / 多 GPU 重构 | 工程量过大，偏系统重构 |
| 完整 PPRGo 级别替换 | 改动深，今天只做设计或 demo |
| 多 seed 大规模验证 | 与今日“多思路探索”目标冲突 |

---

## 十六、今日最终成果验收清单

一天结束时，至少应有：

```text
experiments/opt_20260629_method_exploration/
  README.md
  PLAN.md
  PAPER_BASIS.md
  COMMANDS.md
  RUN_REGISTRY.csv
  METHOD_MANIFEST.csv

  00_existing_results/
    results/*.csv
    figs/*.png
    report.md

  01_sehgnn_lite_pubmed_nc/
    code/*.py
    logs/*.log
    results/*.csv
    figs/*.png
    report.md

  02_lp_pair_decoder_pubmed_lp/
    code/*.py
    logs/*.log
    results/*.csv
    figs/*.png
    report.md

  summary/
    all_results.csv
    all_runtime.csv
    optimization_day_summary.md
    optimization_findings_for_ppt.md
    all_figures_index.md
```

更理想的完整成果包括：

```text
03_sampled_lp_training_pubmed_lp/
04_distill_mlp_pubmed_nc/
05_ppr_topk_lite_design/
```

---

## 十七、面向后续报告的表述边界

可以写：

```text
在完成基础复现后，我们进一步围绕 EHGNN 的训练效率与链路预测表达力进行了方法级优化探索。具体包括：基于预计算聚合特征的 SeHGNN-lite 节点分类原型、基于 pair-wise 表征的链路预测解码器增强、采样式链路预测训练协议、EHGNN-to-MLP 蒸馏设计，以及确定性 PPR-TopK 替代随机游走的方案设计。相关代码、日志、表格、图片和分析报告统一保存在 experiments/opt_20260629_method_exploration/ 下。
```

不能写：

```text
所有优化方法均已在全数据集 5-seed 上验证有效。
```

更严谨的说法是：

```text
当前结果属于方法探索与小规模初步验证，后续仍需在多 seed 与更多数据集上进一步验证。
```

---

## 十八、给 Cursor 的最后提醒

执行本计划时，请始终记住：

1. **今天的关键不是跑最大实验，而是把多条方法级优化思路做成可保存、可复查、可继续扩展的实验资产。**
2. **所有代码都要保存到统一实验根目录，不要散落在原项目各任务目录。**
3. **所有日志、表格、图片、报告都要放到统一根路径下。**
4. **每条方法都要有独立代码、独立结果、独立 report。**
5. **任何结果都要注明证据等级，不得夸大。**
6. **不要污染正式复现结果，不要混用 server_results、本地调试 results 和方法探索 results。**
