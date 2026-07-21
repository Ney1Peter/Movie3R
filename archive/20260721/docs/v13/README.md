# V13 文档入口

V13 验证严格流式 World-Coordinate Memory 的几何可行性，并在 Stage-1 硬门控失败后停止 matcher、memory policy、soft constraint 和 reliability 训练。

主报告：

- [V13 Causal World-Coordinate Memory Feasibility Ladder](V13_CAUSAL_WORLD_COORDINATE_MEMORY_FEASIBILITY_LADDER_20260719.md)

核心结论：

- AvatarReX 和 THuman 的 fresh pointmap 在 pseudo correct correspondence 下可以达到约 0.2-0.5 度旋转误差；
- MVHuman100/200 的 local pointmap/depth consistency 明显不足，导致全量三帧 pseudo 上限仍为 `0.2087 m / 2.898 deg`；
- Sim(3) 整体更差，单一尺度修正不是主解；
- 三帧、空间均匀采样和高历史覆盖有明确收益，但仍未达到 matcher 训练门槛；
- 下一步应先改进 MVHuman 多帧 local geometry 和 depth，而不是训练 World-Anchor Matcher。
