# PPT 要点草稿

> 证据等级：P0 = 已有日志汇总（Reliable for runtime；**非**性能复现成功）

## Slide：为何做方法级优化

- DBLP LP 5-seed **运行完整**（100 epoch），但 Final AUC **0.5008 ± 0.0039** ≈ 随机
- 单 seed 训练 **~52 h 均值** → 不适合快速迭代；PubMed 更适合原型
- 瓶颈：**~1.77M author steps/epoch** + **~411 s/eval**

## Slide：已有正式复现覆盖

- NC：PubMed / DBLP / Yelp 5-seed ✓（2026-06-02）
- LP：PubMed 5-seed ✓；DBLP 5-seed ✓（日志）；Yelp LP ✗

## Slide：今日方法路线

SeHGNN-lite → Pair Decoder → Sampled-LP → Distill → PPR-TopK-lite

## TODO

- [ ] P1–P5 结果填入
- [ ] 每张图见 `all_figures_index.md`
