# 命令记录

## 2026-06-03 本地初始化 — Git 检查

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log --oneline -n 5
git remote -v
```

用途：记录实验初始化时的 Git 快照。  
结果：分支 `reproduce-baseline`，commit `c231c79`；未跟踪 `docs/EHGNN_方法级优化探索计划.md` 与 `experiments/`。

---

## 2026-06-03 本地初始化 — Python 环境检查

```bash
python --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import dgl; print('dgl', dgl.__version__)"
```

用途：判断本地能否运行 EHGNN。  
结果：Python 3.13.2；torch 2.7.0+cpu；**dgl 未安装** → 本地不可训练。

---

## 2026-06-03 P0 — DBLP LP 日志解析与画图

```bash
python experiments/opt_20260629_method_exploration/00_existing_results/code/p0_scan_and_parse.py
```

用途：扫描 `server_results/` 等路径，解析 DBLP LP 5-seed 日志，生成 CSV 与 matplotlib 图。  
结果：5/5 seed 完整；输出 `dblp_lp_5seed_summary.csv`、`dblp_lp_5seed_stats.csv`、4 张 PNG。

---

## 2026-06-03 P0 — 日志路径列举（PowerShell）

```powershell
cmd /c "dir /b server_results\2026-06-02_\Link Prediction\results\dblp_lp_5seeds_val1000\logs"
```

用途：确认 DBLP LP val1000 正式日志文件名。  
结果：5 个 `*_e100.log` + 2 个 diagnostic `*_e2.log`。

---

## 2026-06-03 P0 — 单日志字段抽样（Python）

```bash
python -c "..."  # 见 p0_scan_and_parse.py 内 parse_dblp_lp_log
```

用途：验证 Final Epoch / Total training time / eval time 正则。  
结果：seed42 — Final Epoch 99，Total 242316 s，mean eval ~477 s。

---

## 2026-06-03 P1 — 只读代码审计

```bash
# 阅读 Node Classification/main.py, models.py, utils.py（只读，无命令输出）
```

用途：确定 PubMed NC 数据流、RW 预计算、EHGNN.forward 输入格式与可复用函数。  
结果：见 `01_sehgnn_lite_pubmed_nc/report.md` §代码复用审计。

---

## 2026-06-03 P1 — 语法检查（本地，无 DGL）

```bash
python -m py_compile experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/sehgnn_lite_model.py
python -m py_compile experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/ehgnn_precompute.py
python -m py_compile experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/parse_sehgnn_lite_results.py
python -m py_compile experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/plot_sehgnn_lite_results.py
# run_pubmed_nc_sehgnn_lite.py — 跳过（import utils/dgl，本地无 DGL）
```

用途：验证 P1 新增 Python 文件语法。  
结果：**全部通过**（含 `run_pubmed_nc_sehgnn_lite.py` 语法检查；运行时仍需 DGL）。

---

## 2026-06-03 P1 — 服务器脚本路径

```text
experiments/opt_20260629_method_exploration/server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh
```

用途：服务器一键运行 P1（train + parse + plot）。  
结果：脚本已创建，**未执行**。

---

## 2026-06-03 P2 — 只读代码审计

```bash
# 阅读 Link Prediction/main.py, main_yelp.py, models.py, utils.py（只读）
```

用途：定位 dot decoder、evaluate_lp、PubMed LP 数据流。  
结果：见 `02_lp_pair_decoder_pubmed_lp/report.md` §代码复用审计。

---

## 2026-06-03 P2 — 语法检查（本地，无 DGL）

```bash
python -m py_compile experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/lp_pair_decoder_model.py
python -m py_compile experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/parse_pair_decoder_results.py
python -m py_compile experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/plot_pair_decoder_results.py
python -m py_compile experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/main_pubmed_lp_pair_decoder.py
```

用途：验证 P2 新增 Python 文件语法。  
结果：**全部通过**（运行时仍需 DGL）。

---

## 2026-06-03 P2 — 服务器脚本路径

```text
experiments/opt_20260629_method_exploration/server_scripts/run_p2_pair_decoder_pubmed_lp.sh
```

用途：服务器一键运行 P2（train + parse + plot）。  
结果：脚本已创建，**未执行**。

---

## 2026-06-03 P3 — 只读代码审计

```bash
# 阅读 Link Prediction/main.py, utils.py, models.py（训练循环与 evaluate_lp）
```

用途：确定训练 index 组织、采样接入点。  
结果：见 `03_sampled_lp_training_pubmed_lp/report.md` §代码复用审计。

---

## 2026-06-03 P3 — 语法检查（本地，无 DGL）

```bash
python -m py_compile experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/sampled_lp_loader.py
python -m py_compile experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/parse_sampled_lp_results.py
python -m py_compile experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/plot_sampled_lp_results.py
python -m py_compile experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/main_pubmed_lp_sampled_training.py
```

用途：验证 P3 Python 语法。  
结果：**全部通过**（运行时仍需 DGL）。

---

## 2026-06-03 P3 — 服务器脚本路径

```text
experiments/opt_20260629_method_exploration/server_scripts/run_p3_sampled_lp_pubmed.sh
```

用途：服务器运行 ratio=0.50 与 0.25 两次 + parse + plot。  
结果：脚本已创建，**未执行**。

---

## 2026-06-03 P4 — 只读代码审计

```bash
# 阅读 Node Classification/main.py, models.py, utils.py; 检查 server_results checkpoint
```

用途：确认 teacher 流程、checkpoint/logits 可用性、student 输入。  
结果：无 checkpoint；baseline 在 pubmed_seed_42.txt；见 `04_distill_mlp_pubmed_nc/report.md`。

---

## 2026-06-03 P4 — 语法检查（本地，无 DGL）

```bash
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/mlp_student_model.py
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/distillation_losses.py
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/parse_distill_results.py
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/plot_distill_results.py
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py
```

用途：验证 P4 Python 语法。  
结果：**全部通过**（运行时仍需 DGL）。

---

## 2026-06-03 P4 — 服务器脚本路径

```text
experiments/opt_20260629_method_exploration/server_scripts/run_p4_distill_pubmed_nc.sh
```

用途：服务器 train teacher + distill student + parse + plot。  
结果：脚本已创建，**未执行**。

---

## 2026-06-03 P5 — 代码审计（只读）

```powershell
# RW Top-K 流程
rg "random_walk_sim|_pick_neighbors_from_rw_multiset|get_model_need" "Node Classification/utils.py"
rg "random_walk_sim" "Link Prediction/utils.py"
```

用途：确认 PubMed 加载、meta-path RW、Top-K→CSR 接口。  
结果：NC `utils.py` L491–567 `_pick_neighbors_from_rw_multiset` + `random_walk_sim`；LP 版 L393+ 类似。

---

## 2026-06-03 P5 — 文件创建

```text
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/ppr_topk_lite.py
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/compare_rw_vs_ppr_neighbors.py
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/parse_ppr_topk_results.py
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/plot_ppr_topk_results.py
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/configs/pubmed_ppr_topk_demo.yaml
experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/report.md
experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh
```

---

## 2026-06-03 P5 — 本地 py_compile

```powershell
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/ppr_topk_lite.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/compare_rw_vs_ppr_neighbors.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/parse_ppr_topk_results.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/plot_ppr_topk_results.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py
```

用途：验证 P5 Python 语法。  
结果：**全部通过**（demo 运行时仍需 DGL + PubMed 数据，不在本地执行）。

---

## 2026-06-03 P5 — 服务器脚本路径

```text
experiments/opt_20260629_method_exploration/server_scripts/run_p5_ppr_topk_demo.sh
```

用途：服务器 PPR-TopK demo + compare + parse + plot。  
结果：脚本已创建，**未执行**。

---

## 2026-06-03 — utils 导入路径修复（P1–P5）

**问题**：服务器运行 P5 报 `ModuleNotFoundError: No module named 'utils'`。  
**根因**：实验入口脚本 `PROJECT_ROOT = EXP_ROOT.parent` 少算一层，指向 `experiments/` 而非 EHGNN 根目录。

**修复文件**：

```text
01_sehgnn_lite_pubmed_nc/code/run_pubmed_nc_sehgnn_lite.py
02_lp_pair_decoder_pubmed_lp/code/main_pubmed_lp_pair_decoder.py
03_sampled_lp_training_pubmed_lp/code/main_pubmed_lp_sampled_training.py
04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py
05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py
server_scripts/run_p1_sehgnn_lite_pubmed_nc.sh  (+ PYTHONPATH NC)
server_scripts/run_p2_pair_decoder_pubmed_lp.sh (+ PYTHONPATH LP)
server_scripts/run_p3_sampled_lp_pubmed.sh      (+ PYTHONPATH LP)
server_scripts/run_p4_distill_pubmed_nc.sh      (+ PYTHONPATH NC)
server_scripts/run_p5_ppr_topk_demo.sh          (+ PYTHONPATH NC)
```

**py_compile（本地）**：

```powershell
python -m py_compile experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/run_pubmed_nc_sehgnn_lite.py
python -m py_compile experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/main_pubmed_lp_pair_decoder.py
python -m py_compile experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/main_pubmed_lp_sampled_training.py
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/main_pubmed_nc_distill.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py
# + 各 P? parse/plot 工具脚本（见下方结果）
```

**未修改**：`Node Classification/`、`Link Prediction/` 下 main/utils/models。

**py_compile 结果**：上述 5 个入口脚本 + 各 P1–P5 parse/plot/工具脚本共 23 个文件 **全部通过**（2026-06-03 本地）。

---

## 2026-06-03 — data path trailing slash 修复（P1–P5）

**问题**：P5 第二次服务器运行报 `FileNotFoundError: .../dataPubMed/node.dat`。  
**根因**：`utils.load_PubMed` 使用 `path = data_path + data_name + '/'`；实验脚本传入 `str(Path(...))` 无尾部 `/`，且曾错误解析为 `Node Classification/../data`。

**修复**：

```text
common/path_utils.py                    # ensure_trailing_slash, resolve_project_data_path
01_.../run_pubmed_nc_sehgnn_lite.py
02_.../main_pubmed_lp_pair_decoder.py
03_.../main_pubmed_lp_sampled_training.py
04_.../main_pubmed_nc_distill.py
05_.../demo_pubmed_ppr_topk.py
```

默认 `data_path = ensure_trailing_slash(PROJECT_ROOT / "data")`；`--path` 仅绝对路径可覆盖。

**未修改**正式主线 `utils.py`。

---

## 2026-06-03 — P5 DGL adj API 兼容修复

**问题**：`DGLGraph.adj() got an unexpected keyword argument 'scipy_fmt'`。  
**修复**：`ppr_topk_lite.dgl_graph_to_scipy_csr()` — `adj_external` / `old_adj` / `edges_fallback`；`demo_pubmed_ppr_topk.py` 调用并写入 meta。

```powershell
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/ppr_topk_lite.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/demo_pubmed_ppr_topk.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/parse_ppr_topk_results.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/plot_ppr_topk_results.py
```

**未修改**正式主线代码。

---

## 2026-06-03 — P2/P3 parse 相对路径修复

**问题**：主实验成功，parse 阶段 `ValueError: ... is not in the subpath of ROOT`（相对 `log_path` vs 绝对 `ROOT`）。  
**修复**：`common/path_utils.py` 新增 `safe_relpath`、`resolve_under_root`；P1–P5 parse + P0 扫描脚本全部替换 `relative_to(ROOT)`。

**Server 容错**：

- `run_p2_pair_decoder_pubmed_lp.sh`：`PARSE_ONLY=1`；主 CSV 存在 skip training；parse/plot 失败仅 WARN
- `run_p3_sampled_lp_pubmed.sh`：ratio050 CSV 存在 skip training；`PARSE_ONLY=1`

```powershell
python -m py_compile experiments/opt_20260629_method_exploration/common/path_utils.py
python -m py_compile experiments/opt_20260629_method_exploration/01_sehgnn_lite_pubmed_nc/code/parse_sehgnn_lite_results.py
python -m py_compile experiments/opt_20260629_method_exploration/02_lp_pair_decoder_pubmed_lp/code/parse_pair_decoder_results.py
python -m py_compile experiments/opt_20260629_method_exploration/03_sampled_lp_training_pubmed_lp/code/parse_sampled_lp_results.py
python -m py_compile experiments/opt_20260629_method_exploration/04_distill_mlp_pubmed_nc/code/parse_distill_results.py
python -m py_compile experiments/opt_20260629_method_exploration/05_ppr_topk_lite_design/code/parse_ppr_topk_results.py
```

---

## 2026-06-03 — 系统性工程审计

见 `AUDIT_FIX_REPORT.md` 与 `server_scripts/smoke_test_method_exploration.sh`。

**修复摘要**：`bootstrap_paths` / `add_source_dir(nc|lp)`、`plot_utils` matplotlib skip、全 stage `PARSE_ONLY=1` + skip-existing、移除 shell `PYTHONPATH` 混用。

**本地**：Windows 无 bash 时运行 `python experiments/opt_20260629_method_exploration/common/import_smoke_checks.py`；完整 smoke test 在服务器执行。

---

## 2026-06-03 — common import path 专项修复（P4 path_utils）

**问题**：P4 服务器报错 `ModuleNotFoundError: No module named 'path_utils'`。各 P1–P5 脚本在 `from path_utils import ...` 前未把 `common/` 加入 `sys.path`（部分脚本误用 `CODE_DIR.parents[4]` 定位 common）。`py_compile` 只检查语法，不执行 import，未能发现。

**修复**：所有导入 `path_utils` / `plot_utils` 的脚本在 import 前加入：

```python
EXP_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = EXP_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
```

**Smoke test 升级**：`common/import_smoke_checks.py` + `smoke_test_method_exploration.sh` 从纯 `py_compile` 升级为真实 import / `--help` 检查（可发现 `path_utils`、`plot_utils`、NC/LP `utils` 路径、data path 尾部 `/`）。

```bash
# 服务器
bash experiments/opt_20260629_method_exploration/server_scripts/smoke_test_method_exploration.sh

# 本地 / 任意环境
python experiments/opt_20260629_method_exploration/common/import_smoke_checks.py
```

**P4 重跑**（smoke 通过后）：

```bash
bash experiments/opt_20260629_method_exploration/server_scripts/run_p4_distill_pubmed_nc.sh
```

**未修改** `Node Classification/`、`Link Prediction/` 正式主线代码。
