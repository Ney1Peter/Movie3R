# Movie3R-Hybrid V14

V14 是正式的 one-shot latent coarse-to-fine streaming版本。仓库中
`versions/v12/experiments/v14_*` 是历史实验编号，不属于本目录的正式 V14 release。

当前状态（2026-07-27）：

```text
V14.1 event-only architecture implemented
single-event overfit completed
10-event pilot completed
```

当前第一阶段文档：

- `docs/V14_1_ONE_SHOT_SHADOW_CORRECTION_TRAINING.md`
- `docs/V14_1_INITIAL_PILOT_RESULTS_20260727.md`

总体方法规范的仓库内归档：

```text
versions/v14/docs/Movie3R-V14.MD
```

当前可用单样本 checkpoint：

```text
output/v14_1/v14_1_cut_event_single_lbn1_1192/checkpoint-best.pth
```

冻结依赖：

```text
V9: full correct-token decoder refinement and corrected pose/human heads
V12: pre-decode hard reset and fixed shot Boundary
V13: mean_raw_t uniform multi-human consensus
```

V14.1 使用两帧 pre-cut context 和一张显式标记的 post-cut event frame，只在 event
frame 插入 semantic/alignment correct tokens并启用 pose/human correction。它暂不接
identity、V13 residual、shot Boundary propagation或 automatic cut detector。
