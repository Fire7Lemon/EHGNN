# server_scripts

本目录用于存放后续在 GPU 服务器上运行的 shell 脚本。

## 约定

1. 输出根路径固定为 `experiments/opt_20260629_method_exploration/`。
2. 每个脚本对应 `METHOD_MANIFEST.csv` 中的一条方法。
3. 日志必须 `tee` 到 `<method_dir>/logs/`。
4. **本轮（P0）未生成可执行训练脚本。**

## 待添加（P1 起）

- `run_p1_sehgnn_lite_pubmed_nc.sh` — **已创建**
- `run_p2_pair_decoder_pubmed_lp.sh` — **已创建**
- `run_p3_sampled_lp_pubmed.sh` — **已创建**
- `run_p4_distill_pubmed_nc.sh` — **已创建**
- `run_p5_ppr_topk_demo.sh` — **已创建**
- `check_server_before_opt_runs.sh` — **已创建**（运行前环境检查）
- `run_all_method_exploration.sh` — **已创建**（P5→P1→P2→P3→P4 总控）
- 总控说明见 `README_RUN_ALL.md`

## TODO

- [ ] P1 脚本骨架
- [ ] 统一 `SEED` / `OUT_DIR` 环境变量
- [ ] 与 `RUN_REGISTRY.csv` 字段对齐
