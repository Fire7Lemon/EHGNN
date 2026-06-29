# 全部图片索引

## P0 — DBLP LP 已有结果

- 图名：DBLP LP Final Test AUC by Seed
- 路径：`00_existing_results/figs/dblp_lp_final_auc_by_seed.png`
- 对应方法：P0
- 数据集：DBLP Link Prediction (val1000 config)
- 图中指标：Final Test AUC（5 seeds）；虚线 random 0.5
- 可用于 PPT 哪一页：问题背景 —「DBLP LP 运行完整但性能接近随机」
- 注意事项：非论文 Table VI 对标；engineering config (val1000)

---

- 图名：DBLP LP Best vs Final Test AUC
- 路径：`00_existing_results/figs/dblp_lp_best_vs_final_auc.png`
- 对应方法：P0
- 数据集：DBLP LP
- 图中指标：Best AUC vs Final AUC 对比
- 可用于 PPT 哪一页：动机 —「训练后期指标回落/接近随机」
- 注意事项：Best 多出现在 epoch 0–20

---

- 图名：DBLP LP Total Training Time by Seed
- 路径：`00_existing_results/figs/dblp_lp_runtime_by_seed.png`
- 对应方法：P0
- 数据集：DBLP LP
- 图中指标：Total training time (hours)
- 可用于 PPT 哪一页：瓶颈分析 —「单 seed 40–67 小时」
- 注意事项：含 load + sim + train + eval

---

- 图名：DBLP LP Mean Runtime Breakdown
- 路径：`00_existing_results/figs/dblp_lp_runtime_breakdown.png`
- 对应方法：P0
- 数据集：DBLP LP
- 图中指标：Load Data / RW Sim / Training Loop 均值（小时）
- 可用于 PPT 哪一页：瓶颈分析 —「训练 loop 占主导」
- 注意事项：Training Loop = total − load − sim；eval 时间含在 loop 内

## P1–P5

（待后续方法运行后补充）
