# V10 文档索引

这个目录保留当前 V10 主线文档。旧探索记录放在 `archive/`，避免和当前路线混在一起。

## 当前主线

先读：

```text
AGENT_BRIEFING_V10_20260715.md
```

设计文档：

```text
V10_CAUSAL_STREAMING_MODEL_DESIGN_20260713.md
```

核心 probe 和最小验证：

```text
V10_BEDLAM_MOTION_INTEGRATOR_PROBE_20260713.md
V10_MINIMAL_BEDLAM21_VALIDATION_20260714.md
V10_ORACLE_STATE_VS_GAUGE_PROBE_20260715.md
```

## 归档记录

`archive/` 里是早期 V10 探索，主要用于追溯，不作为当前默认方案：

```text
archive/V10_END_TO_END_EVAL_PROTOCOL_20260708.md
archive/V10_HUMAN3R_SPECIFIC_GLOBAL_ALIGNMENT_PLAN_20260713.md
archive/V10_LARGE_4SOURCE_DATASET_SETUP_20260708.md
archive/V10_STATIC_ALIGNMENT_PROBE_RESULTS_20260708.md
archive/V10_STREAMING_GLOBAL_ALIGNMENT.md
```

当前主线以严格流式 global-state integrator 为核心，优先围绕 Human3R output-domain、oracle boundary、history-current direct + residual integrator 继续推进。
