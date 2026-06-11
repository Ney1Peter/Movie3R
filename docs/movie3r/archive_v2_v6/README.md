# V2-V6 Historical Archive

该目录保存 Movie3R 在当前分水岭之前的历史探索文档。

## 归档原因

近期测试显示，Human3R 在 RICH / AvatarReX 等纹理较丰富的数据上通常表现稳定，而明显偏移更多出现在简单、低纹理、背景特征弱的视频中。

这意味着 V2-V6 中围绕 ShotToken、背景特征匹配、AnchorToken 和 pose-only adapter 的探索，没有完全命中当前最重要的失败场景。旧方向仍有诊断价值，但不再作为当前主线继续推进。

## 归档内容

| 文档 | 内容 |
|---|---|
| `shot_token_generation_explanation.md` | 早期 ShotToken 生成逻辑说明 |
| `shot_token_recent_training_report.md` | ShotToken 近期训练记录 |
| `shot_token_v4_design.md` | V4 pose alignment / ShotToken 设计 |
| `shot_token_v5_plan.md` | V5 layerwise pose-only 规划 |
| `shot_token_v6_plan.md` | V6 local scene re-anchor token 规划 |
| `V6 AnchorPoseAdapter.md` | V6 AnchorPoseAdapter 设计和实现记录 |
| `recent_training_runtime_summary.md` | 近期训练运行记录与实验日志 |
| `model.md` | V1/V2 Shot-Aware model 设计旧文档 |
| `training.md` | V2-V6 训练配置旧文档 |
| `ppt_training_config_and_experiment_setup.md` | V4/V5 训练汇报材料 |
| `ANCHOR_TOKEN_V6_CONTEXT.md` | V6 AnchorToken handoff 上下文 |
| `anchor_token_report_v1/` | AnchorToken V6 图文报告和可视化材料 |

相关历史脚本已归档到：

```text
scripts/archive_v2_v6/
```

## 使用方式

这些文档用于追溯和复盘，不应再被理解为当前主线设计。当前主线见：

```text
docs/movie3r/v9/README.md
```
