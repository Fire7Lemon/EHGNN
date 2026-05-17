# EHGNN

An implementation for the paper--Efficient Learning for Billion-scale Heterogeneous Information Networks.

### Dataset

The three datasets used in the paper (PubMed, Yelp and DBLP) can be downloaded from [here](https://drive.google.com/drive/folders/186u90Y0gzmdI-R6gQEOis1Nip6G759rv?usp=drive_link). In addition, the OGB-MAG240M dataset can be found [here](https://ogb.stanford.edu/docs/lsc/mag240m/). Please place the downloaded datasets in the `../data`.

### Usage

To conduct the experiments, please execute `main.py` in each folder (Node Classification, Link Prediction, and MAG240M). Hyperparameters can be explored within the `main.py`, and here are the ones we used.

| Task | Dataset | $\alpha$ | K   | learning rate | dropout | hidden dimension | layers | batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Node Classification | PubMed | 0.7 | 20  | 1e-3 | 0.4 | 256 | 4   | 3000 |
|     | Yelp | 0.7 | 20  | 3e-4 | 0.5 | 256 | 4   | 3000 |
|     | DBLP | 0.7 | 20  | 5e-4 | 0.5 | 512 | 5   | 3000 |
|     | OGB-MAG240M | 0.7 | 25  | 3e-4 | 0.4 | 512 | 5   | 5000 |
| Link Prediction | PubMed | 0.1 | 20  | 3e-4 | 0.5 | 256 | 4   | 40  |
|     | Yelp | 0.1 | 20  | 3e-4 | 0.5 | 256 | 4   | 100 |
|     | DBLP | 0.7 | 20  | 5e-4 | 0.5 | 512 | 5   | 1000 |

## 项目文档导航

面向中文读者：说明类中文 Markdown 集中在 **`docs/`** 目录（**`README.md`** 仍为 GitHub 默认首页，保留在仓库根目录）。

| 文档 | 说明 |
|------|------|
| [项目结构与使用说明](docs/项目结构与使用说明.md) | 说明项目目录、任务入口、运行脚本和结果目录 |
| [实验记录](docs/实验记录.md) | 记录本地 smoke test、selected seeds、服务器待跑实验和异常现象 |
| [论文复现报告](docs/论文复现报告.md) | 面向论文复现进度的正式说明 |
| [环境配置说明](docs/环境配置说明.md) | 服务器环境重建、`environment.yml` / `requirements-freeze.txt` 的定位、PyTorch / DGL / torch-scatter 与 CUDA、`data`/`results` 同步问题 |
| [GitHub提交检查清单](docs/GitHub提交检查清单.md) | 提交前检查哪些文件应该提交、哪些不应提交（含 `environment.yml`、`requirements-freeze.txt`） |
| [MAG240M运行说明](docs/MAG240M运行说明.md) | MAG240M 数据下载、校验和服务器运行说明 |
| [scripts 目录说明](scripts/scripts目录说明.md) | `scripts/` 下校验、串联实验、资源监控、环境与 conda-pack 脚本说明 |

根目录 **`environment.yml`** 为 conda **规格参考**（可用 **`conda env create -f environment.yml`** 创建同名环境）；**`requirements-freeze.txt`** 为本机 pip **版本快照**，便于对照差异。**二者都不是已安装的环境本体**，clone 后仍需自行安装依赖，GPU 相关包须按目标机 CUDA 与官方说明选型（详见 **`docs/环境配置说明.md`**）。
