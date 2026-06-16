# Movie3R V9

V9 是当前主线，从已经验证的 V8.9 implicit human-pose correction branch 开始。

新对话或新实验请先读：

| 文档 | 用途 |
|---|---|
| [AGENT_BRIEFING.md](AGENT_BRIEFING.md) | 给 AI 工具看的接手说明：项目背景、历史尝试、当前状态、下一步 |
| [METHOD_OVERVIEW.md](METHOD_OVERVIEW.md) | 给我们自己看的通俗技术说明：Human3R、UniCon3R、V9 设计、loss |
| [MODEL_ARCHITECTURE_DETAILS.md](MODEL_ARCHITECTURE_DETAILS.md) | 模型结构细节：`A_corr,t` 构造、decoder 拼接、pose/human correction heads、PPT 符号 |
| [GUARDRAILS.md](GUARDRAILS.md) | 易错细节和固定规则：坐标系、dataloader、可视化、训练、commit |
| [FORMAL_60H_TRAINING_PLAN.md](FORMAL_60H_TRAINING_PLAN.md) | 下一轮正式 60h 混合训练方案：数据、配置、指标、checkpoint、训练前检查 |

Current base commit:

```text
a79eb18 feat: add v8.9 implicit human-pose correction
```

Current preserved checkpoint:

```text
output/v9_saved_weights/v9_implicit_avatarrex_single_checkpoint-best.pth
output/v9_saved_weights/v9_implicit_avatarrex_single_checkpoint-final.pth
```

Starting point:

```text
A_corr_t enters the decoder as a UniCon-style streaming relation token.
The refined correction token predicts both pose-token and human-token residuals.
Camera correction is applied before the original pose head.
Human correction is applied before the original Human3R human head.
GT camera and SMPL are only used for loss, metrics, and visualization overlays.
```

历史 V8 文档已经归档到：

```text
docs/movie3r/archive_v8/
```

其中 [v8_9_implicit_human_pose_token.md](../archive_v8/v8_9_implicit_human_pose_token.md)
是 V9 的直接来源。
