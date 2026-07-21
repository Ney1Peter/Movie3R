# V14 文档入口

V14 测试无 GT depth、无 GT correspondence 的选择性 World-Memory 重定位。

主报告：

- [V14 Depth-Free Selective World-Memory Relocalization](V14_DEPTH_FREE_SELECTIVE_WORLD_MEMORY_RELOCALIZATION_20260719.md)

核心结论：

- World-Memory 与 Fixed Explicit 在 Oracle 层面存在有限互补性；
- 互补性主要集中在 THuman、高重叠和高纹理子集；
- always-use World-Memory 明显更差；
- Geometry、Token、Human/Gravity Gate 均不能跨数据源控制 false accept；
- 完整 Accept/Wait/Fallback 策略差于 Fixed Explicit；
- 当前 Selective World-Memory 主线应停止，World-Memory 只保留为诊断候选。
