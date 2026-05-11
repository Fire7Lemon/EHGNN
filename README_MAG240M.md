# MAG240M（EHGNN）服务器运行指南

本说明面向 **Linux 服务器**：clone 本仓库后安装依赖、下载 **OGB-LSC MAG240M** 数据并运行 `MAG240M/main.py`。  
数据体量极大，**请勿在本地笔记本下载全量数据**。

---

## 一、代码里数据路径说明（与 `main.py` 一致）

| 参数 | 含义 |
|------|------|
| `--path` | 传给 `ogb.lsc.MAG240MDataset(root=...)` 的 **父目录**。下载完成后其中会出现 `mag240m_kddcup2021/`。 |
| `--data_root` | 与 `--path` 等价别名（便于脚本/export）；若指定则 **覆盖 `--path`**。 |
| `--feature_path` | 可选；默认 `<root>/mag240m_kddcup2021/processed/paper/`（含 `node_feat.npy`）。 |
| `--other_feature_path` | 可选；默认 `<root>/mag240m_kddcup2021/`（含 `author.npy`、`institution.npy`）。 |

图与划分来自 **`MAG240MDataset`**（见 `MAG240M/utils.py` 中 `load_240m`），**不是** `ogbn-mag` 小规模数据集。

示例（Linux）：

```bash
export MAG240_DATA_ROOT="${HOME}/data"
python MAG240M/main.py --data_root "${MAG240_DATA_ROOT}"
```

---

## 二、推荐服务器配置

| 项目 | 建议 |
|------|------|
| OS | **Linux x86_64**（Ubuntu 20.04/22.04 LTS 等） |
| 内存 | **≥ 256 GB RAM**（全图与特征 mmap 极度吃内存） |
| 磁盘 | **≥ 500 GB SSD**，推荐 **1 TB+**（原始数据 + 解压 + 缓存） |
| GPU | 论文设置见下；显存需按 batch 与实现评估 **≥ 16 GB** 起步 |

### 论文实验环境（摘自 README / 论文常用描述，可作对齐参考）

- CPU：Intel Xeon E5-2680  
- GPU：NVIDIA Tesla **P100 16GB**  
- 内存：**256 GB RAM**  
- OS：**Ubuntu 18.04**  
- CUDA：**11.4**  

实际部署时 CUDA / PyTorch / DGL 版本需 **三角兼容**（见下文常见问题）。

---

## 三、Conda 环境与依赖安装

```bash
conda create -n ehgnn python=3.10 -y
conda activate ehgnn
```

按服务器 CUDA 版本安装 **PyTorch**（示例：CUDA 12.1 轮子，请替换为官方命令）：

```bash
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装与 PyTorch 匹配的 **DGL**（示例 CUDA 12.1；请以 [DGL 安装页](https://www.dgl.ai/pages/start.html) 为准）：

```bash
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

其它 Python 依赖：

```bash
pip install ogb numpy scipy scikit-learn
```

若训练入口使用 `torch-scatter`，请安装与当前 **torch + CUDA** 匹配的轮子（参考 [PyG wheels](https://data.pyg.org/whl/)），例如：

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

具体 URL 需与你的 `torch` 版本一致。

---

## 四、MAG240M 数据下载

在目标 **数据父目录**（例如 `$HOME/data`）下由 OGB 创建 `mag240m_kddcup2021/`：

```bash
export MAG240_DATA_ROOT="${HOME}/data"
mkdir -p "${MAG240_DATA_ROOT}"
python -c "from ogb.lsc import MAG240MDataset; MAG240MDataset(root='${MAG240_DATA_ROOT}')"
```

首次运行会下载并解压，耗时与磁盘 IO 较长；建议在 `tmux`/`screen` 中执行。

---

## 五、数据可读性校验（不训练）

```bash
export MAG240_DATA_ROOT="${HOME}/data"
python scripts/check_mag240m_data.py --root "${MAG240_DATA_ROOT}"
```

---

## 六、训练入口

在仓库根目录（或自行设置 `PYTHONPATH`）从 **`MAG240M` 目录**运行（保证 `utils`/`models` 导入正确）：

```bash
export MAG240_DATA_ROOT="${HOME}/data"
cd MAG240M
python main.py --data_root "${MAG240_DATA_ROOT}" --gpu 0
```

或使用一键脚本（见 `scripts/run_mag240m_server.sh`）。

---

## 七、一键脚本

```bash
export MAG240_DATA_ROOT="${HOME}/data"
export MAG240M_CONDA_ENV="${MAG240M_CONDA_ENV:-ehgnn}"
chmod +x scripts/run_mag240m_server.sh   # 仅需一次
./scripts/run_mag240m_server.sh
```

日志：`logs/mag240m_run.log`（相对 **仓库根目录**）。

---

## 八、常见问题

| 现象 | 可能原因 | 处理 |
|------|-----------|------|
| `No space left on device` | 磁盘不足 | 扩容或使用更大挂载；清理中间文件 |
| Killed / OOM | 内存不足 | 增大 RAM；减小 `batch_size`（属训练超参，需谨慎） |
| 下载中断 | 网络不稳 | 删除不完整缓存后重跑同一 `MAG240MDataset(root=...)`；必要时换镜像网络 |
| CUDA error / DGL import error | **Torch 与 DGL CUDA 版本不匹配** | 按同一 CUDA 标签重装 PyTorch + DGL |
| `FileNotFoundError` for `*.npy` | `feature_path` / `other_feature_path` 与真实目录不一致 | 确认 `mag240m_kddcup2021` 子目录完整；可用 `--feature_path` / `--other_feature_path` 显式指定 |

---

## 九、与 Node Classification / Link Prediction 的关系

本指南 **仅** 涉及 **MAG240M** 子目录与仓库根目录下的 `README_MAG240M.md`、`scripts/`。  
PubMed 等已在 **Node Classification** 中稳定的脚本 **不在此修改范围内**。
