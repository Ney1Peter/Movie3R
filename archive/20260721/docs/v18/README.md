# V18 Human-Calibrated Metric Translation

当前结论与完整实验记录见：

- [V18_HUMAN_CALIBRATED_METRIC_TRANSLATION_20260720.md](V18_HUMAN_CALIBRATED_METRIC_TRANSLATION_20260720.md)

核心结果：人体投影候选将 translation mean 从 Fixed Explicit 的 `1.715 m` 降至 `0.872 m`；进一步引入冻结 `DA3Metric-Large` 后降至 `0.518 m`，translation catastrophic 从 `65.6%` 降至 `15.0%`。四个数据源均同方向改善，说明外部 metric depth 可以补足 Human3R 缺失的绝对尺度。
