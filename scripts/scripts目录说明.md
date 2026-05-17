# scripts 目录说明

> **分支**：`reproduce-baseline`  
> **说明对象**：本仓库 `scripts/` 目录下**当前真实存在**的 `.py`、`.sh` 文件。  
> **说明**：论文相关的 **Fig.6 / Table IX / Table X TODO** 文档当前位于 **`docs/`**，不在 `scripts/` 内；见下文 §5。

---

## 1. 目录定位

`scripts/` 主要用于：

- **服务器运行辅助**（核心复现、扩展实验的一键串联）；
- **MAG240M** 数据校验与服务器侧运行流水线；
- **资源监控**（包裹长命令并周期性记录 GPU/内存/磁盘）；
- **实验结果汇总索引**（扫描各 `summary.*`，不写回原始结果）；
- **Linux 环境创建 / conda-pack / 解压**（与根目录 `environment.yml`、`requirements-server-cu121.txt` 配合）；
- **环境自检**（import torch、dgl、ogb 等）。

**不存放**具体模型训练的「业务主入口」：`main.py`、各类 `run_*_5seeds.py` 等仍在 **`Node Classification/`**、**`Link Prediction/`**、**`MAG240M/`**。

---

## 2. 文件总览表

| 文件 | 类型 | 作用 | 是否会直接运行实验 | 推荐运行位置 |
|------|------|------|-------------------|--------------|
| `check_mag240m_data.py` | Python 工具 | 校验 `MAG240MDataset(root)` 路径并实例化数据集 | **否**（校验为主；数据缺失时可能触发 OGB **下载/预处理**，耗时与磁盘占用大） | 仓库根：`python scripts/check_mag240m_data.py ...` |
| `collect_result_summaries.py` | Python 工具 | 扫描全仓库 `summary.csv` / `summary.txt`，生成根目录 `results/ALL_SUMMARIES_INDEX.md` | **否** | 仓库根 |
| `check_env.py` | Python 工具 | 只读检查 Python/torch/CUDA、dgl、torch_scatter、numpy、sklearn、ogb 等 | **否** | 任意（建议在已激活的 conda 环境中） |
| `run_mag240m_server.sh` | Shell 脚本 | 磁盘门禁 → `conda activate` → `check_mag240m_data.py` → **`MAG240M/main.py` 训练** | **是**（MAG240M 训练） | **Linux**；仓库根执行 |
| `run_server_resource_monitor.sh` | Shell 脚本 | 后台每 60s 采样资源，**包裹**一条用户命令 | **否**（本身不训练；被包裹命令可以是训练） | **Linux**；仓库根 |
| `server_run_core_reproduction.sh` | Shell 脚本 | 串联 NC/LP **5-seed 主结果** + MAG240M 校验 + **`run_mag240m_server.sh`** | **是**（含完整 MAG240M 流水线） | **Linux**；仓库根 |
| `server_run_extended_experiments.sh` | Shell 脚本 | 串联 NC/LP **消融 + K/α/walk_num 敏感性** | **是** | **Linux**；仓库根 |
| `setup_env_linux_cu121.sh` | Shell 脚本 | Linux 上创建/复用 conda 环境 `ehgnn` 并安装 PyTorch 2.3.1+cu121、DGL、torch-scatter、`requirements-server-cu121.txt` | **否** | **Linux**；仓库根 |
| `pack_env_linux.sh` | Shell 脚本 | 对已配置好的 `ehgnn` 执行 `conda pack`，生成根目录 `ehgnn-linux-py310-cu121-torch231.tar.gz` | **否** | **Linux**；仓库根 |
| `unpack_env_linux.sh` | Shell 脚本 | 解压 conda-pack 包并执行 `conda-unpack`，再运行 `check_env.py` | **否** | **Linux**；仓库根 |

---

## 3. Python 工具说明

### 3.1 `check_mag240m_data.py`

- **用途**：检查 **`MAG240MDataset(root=...)`** 父目录是否就绪；打印解析后的根路径、`mag240m_kddcup2021/`、关键文件（如 `processed/paper/node_feat.npy`）是否存在；然后 **构造 `MAG240MDataset`** 并输出论文数、类别数、`split_dict` 规模等。
- **典型命令**（在仓库根）：
  ```bash
  python scripts/check_mag240m_data.py --root "$HOME/data"
  ```
- **参数**：**`--root`**（必填）：`MAG240MDataset` 的**父目录**（其下应为 `mag240m_kddcup2021/`）；支持 `~/` 展开。
- **与 `MAG240MDataset(root)` 的关系**：脚本内部对该路径调用 **`MAG240MDataset(root=root)`**，与 **`MAG240M/main.py`** 中数据根语义一致（详见 **`../docs/MAG240M运行说明.md`**）。
- **下载 / 大数据**：若数据缺失，**可能触发 OGB 下载与预处理**；脚本文档字符串注明耗时可达数小时至约一天，视磁盘与网络而定。
- **运行前**：预留足够磁盘（MAG240M 规模极大）；服务器脚本 `run_mag240m_server.sh` 默认要求 **`MAG240_DATA_ROOT` 所在文件系统可用空间 ≥ 500 GB**（可绕过但不推荐）。

### 3.2 `collect_result_summaries.py`

- **用途**：从仓库根递归查找（跳过 `.git`）所有 **`summary.csv`**、**`summary.txt`**，写入 **`results/ALL_SUMMARIES_INDEX.md`**（UTC 时间戳 + 路径列表）。
- **不修改**已有 summary 文件。
- **适合**：服务器批量跑完后统一做结果路径索引；该 **`results/`** 目录通常被 `.gitignore` 忽略，**勿提交**索引文件到 Git。
- **典型命令**：
  ```bash
  python scripts/collect_result_summaries.py
  ```

### 3.3 `check_env.py`

- **用途**：只读环境探测：解释器、`sys.prefix`、`torch`/`CUDA`/GPU、`dgl`、`torch_scatter`、`numpy`、`scipy`、`sklearn`、`pandas`、`tqdm`、`ogb` 等；可选探测 `joblib`、`dask`。
- **退出码**：任一**必选**依赖 import 失败则 **非 0**。
- **典型命令**：
  ```bash
  python scripts/check_env.py
  ```

---

## 4. Shell 脚本说明

### 4.1 `run_mag240m_server.sh`

- **用途**：服务器 MAG240M **流水线**：仓库布局检查 → **`conda`** → （可选）磁盘空间门禁 → **`check_mag240m_data.py`** → 进入 **`MAG240M/`** 执行 **`python main.py --data_root "$MAG240_DATA_ROOT"`**（可附加 **`main.py` 的参数**，脚本末尾 `"$@"` 透传）。
- **运行前**：建议 **`chmod +x scripts/run_mag240m_server.sh`**；在仓库根执行 **`./scripts/run_mag240m_server.sh`** 或 **`bash scripts/run_mag240m_server.sh`**。
- **环境变量**：**`MAG240_DATA_ROOT`**（默认 **`$HOME/data`**）；**`MAG240M_CONDA_ENV`**（默认 **`ehgnn`**）；磁盘不足时可设 **`MAG240_SKIP_DISK_CHECK=1`**（不推荐）。
- **日志**：标准输出与错误会 **tee** 到 **`logs/mag240m_run.log`**。
- **建议**：长时间训练放在 **tmux / screen** 中运行；依赖 **`conda`** 在 PATH 且已 **`conda init bash`** 一类配置。

### 4.2 `run_server_resource_monitor.sh`

- **用途**：在**后台**每 **60 秒**追加写入一次 **`nvidia-smi`**、**`free -h`**、**`df -h`**（仓库根）、**`ps`**（内存排序前几行）；同时 **前台执行用户给定的一条命令**（通过 **`eval`**）。
- **日志**：**`logs/resource_monitor_<时间戳>.log`**。
- **典型用法**（仓库根）：
  ```bash
  bash scripts/run_server_resource_monitor.sh 'cd "Node Classification" && python run_pubmed_5seeds.py'
  ```
- **说明**：不负责解析实验指标，**只做资源采样**；被包裹命令的退出码会原样返回。

### 4.3 `server_run_core_reproduction.sh`

- **用途**：**服务器核心复现**一键串联：
  - **Node Classification**：`run_pubmed_5seeds.py`、`run_dblp_5seeds.py`、`run_yelp_5seeds.py`（均为 **5 seeds**，含 `--skip_existing` 的后两者）；
  - **Link Prediction**：`run_*_lp_5seeds.py` 三组数据集；
  - **MAG240M**：`python scripts/check_mag240m_data.py --root "${MAG240_DATA_ROOT:-$HOME/data}"`，再 **`chmod +x`** 并执行 **`./scripts/run_mag240m_server.sh`**（即包含 MAG240M **训练**）。
- **日志**：**`logs/server_core_reproduction.log`**（脚本整体 **`tee`**）。
- **失败行为**：任一步 **`run_step`** 非零退出码则 **打印 `[FAIL]` 并立即 `exit`**（**`set -e`** 语义下的封装）。
- **典型命令**：
  ```bash
  chmod +x scripts/server_run_core_reproduction.sh
  bash scripts/server_run_core_reproduction.sh
  ```
- **注意**：脚本**不会**安装依赖、**不会**切换 Git 分支；需事先 **`conda activate`**（或与 `run_mag240m_server.sh` 内 conda 环境名一致）。

### 4.4 `server_run_extended_experiments.sh`

- **用途**：**扩展实验**：NC/LP 全数据集消融 **`run_nc_ablation_all_datasets.py`** / **`run_lp_ablation_all_datasets.py`**；以及 **`run_nc_sensitivity_{k,alpha,walk_num}.py`**、**`run_lp_sensitivity_{k,alpha,walk_num}.py`**（均带 **`--seeds 42 3407 2026 6666 8888 --skip_existing`**）。
- **日志**：**`logs/server_extended_experiments.log`**。
- **建议**：在 **`server_run_core_reproduction.sh`** 完成且资源允许后再跑。
- **失败行为**：任一步失败即退出（同 core 脚本的 **`run_step`**）。
- **典型命令**：
  ```bash
  chmod +x scripts/server_run_extended_experiments.sh
  bash scripts/server_run_extended_experiments.sh
  ```

### 4.5 `setup_env_linux_cu121.sh`

- **用途**：仅在 **Linux** 上：检查 **`conda`** → 根据根目录 **`environment.yml`** 创建或复用 **`ehgnn`** → 安装 **PyTorch 2.3.1+cu121**、**DGL**（脚本内注释说明与历史 Windows **dgl 1.1.2** 方案区分）、**torch-scatter**、**`requirements-server-cu121.txt`** → 运行 **`check_env.py`**。
- **典型命令**：**`bash scripts/setup_env_linux_cu121.sh`**（仓库根）。

### 4.6 `pack_env_linux.sh`

- **用途**：在 **Linux** 上对 conda 环境 **`ehgnn`** 执行 **`conda pack`**，输出 **`ehgnn-linux-py310-cu121-torch231.tar.gz`** 到仓库根；会先 **`conda install -n base conda-pack`**（若需）。
- **提醒**：**勿将 tar.gz 提交 Git**；宜上传 Release / 对象存储；解压后需 **`conda-unpack`**。

### 4.7 `unpack_env_linux.sh`

- **用途**：解压 conda-pack 产物并运行 **`<target_dir>/bin/conda-unpack`**，再用解压环境的 Python 运行仓库内 **`scripts/check_env.py`**。
- **用法**：
  ```bash
  bash scripts/unpack_env_linux.sh <env_tar_gz> <target_dir>
  ```
- **注意**：仅适用于 **Linux 上打出来的** conda-pack 包，**不适用于 Windows** 环境包。

---

## 5. TODO 文档说明

下列两份 **TODO** 为 Markdown **规划文档**，**不在 `scripts/` 目录内**，路径位于 **`docs/`**：

### 5.1 `HPPR策略研究待办.md`

- **路径**：**`../docs/HPPR策略研究待办.md`**
- **对应**：论文 **Table X**（PubMed NC 上 **AvgSim / HeteSim / Ours** 等 HPPR / 相似度策略对比）。
- **现状**：文档结论为 **AvgSim / HeteSim 未实现**；random / hybrid / temp 属于 **`neighbor-strategy-dev`** 一类邻居策略探索，**不能替代 Table X**。
- **后续**：需在构图阶段实现策略分支与批量脚本等（见该文档 §4）。

### 5.2 `元路径实验待办.md`

- **路径**：**`../docs/元路径实验待办.md`**
- **对应**：**Fig.6**（meta-path 数量）、**Table IX**（meta-path 权重导出）。
- **现状**：meta-path 列表在源码中写死，**无** CLI 子集 / 权重导出接口；故先以 TODO 文档记录最小补丁方案。
- **后续**：需加 CLI 或导出逻辑及批量脚本（见该文档）。

---

## 6. 推荐服务器使用顺序

1. **环境与数据**：按 **`../docs/环境配置说明.md`** 配置 conda / CUDA；准备好 **`data/`**（及 MAG240M 父目录）。
2. **（可选）环境自检**：**`python scripts/check_env.py`**
3. **MAG240M 数据**：**`python scripts/check_mag240m_data.py --root "$HOME/data"`**（确认磁盘与时间成本）。
4. **核心复现**：**`bash scripts/server_run_core_reproduction.sh`**
5. **扩展实验**：**`bash scripts/server_run_extended_experiments.sh`**
6. **结果索引**：**`python scripts/collect_result_summaries.py`**
7. **超长任务**：用 **`run_server_resource_monitor.sh`** 包裹单条长命令。

示例：

```bash
cd ~/EHGNN
conda activate ehgnn

python scripts/check_mag240m_data.py --root "$HOME/data"

chmod +x scripts/server_run_core_reproduction.sh
bash scripts/server_run_core_reproduction.sh

chmod +x scripts/server_run_extended_experiments.sh
bash scripts/server_run_extended_experiments.sh

python scripts/collect_result_summaries.py
```

---

## 7. 注意事项

- **`scripts/` 内 `.sh` 面向 Linux 服务器**；**Windows PowerShell 不能直接跑 `.sh`**，请用 **Git Bash / WSL / Linux**。
- **长实验**建议使用 **tmux / screen**。
- **`data/`**、**`results/`**、**`logs/`**、checkpoint **不应提交 GitHub**（见 **`../docs/GitHub提交检查清单.md`**）。
- 服务器上同步回本地的结果，可自行归档到 **`server_results/`** 等**未被误提交**的路径（勿与仓库忽略规则冲突）。
- **`Link Prediction/`** 下 **`run_*_lp_smoke.py`** 等为 **smoke**，**不等于**论文主表指标。
- **`server_run_core_reproduction.sh`** 与 **`server_run_extended_experiments.sh`** 可能**耗时极长**，请预留 GPU 与时间窗口。

---

## 8. 与其它文档的关系

| 需求 | 文档（相对本文件路径） |
|------|------------------------|
| 环境配置 | **`../docs/环境配置说明.md`** |
| 项目整体结构与子目录命令 | **`../docs/项目结构与使用说明.md`** |
| 实验进度与 smoke | **`../docs/实验记录.md`** |
| 复现状态 | **`../docs/论文复现报告.md`** |
| Git 提交规则 | **`../docs/GitHub提交检查清单.md`** |
| MAG240M 部署细节 | **`../docs/MAG240M运行说明.md`** |

---

## 9. 文档修订说明（仓库维护）

- 本文件路径：**`scripts/scripts目录说明.md`**（文件名 **`scripts目录说明.md`**，位于 **`scripts/`** 目录）。
- 若 `scripts/` 增删脚本，请同步更新本文件的 **§2 总览表**与各小节。
