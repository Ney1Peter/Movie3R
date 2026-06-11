# Movie3R 文档入口

Movie3R 当前主线是 **V9 implicit human-pose correction**。V8.9 已验证的 UniCon-style streaming correction token 是 V9 的起点；V8 及更早方案只作为历史复盘保留。

## 当前判断

当前最有效的方向不是后处理 teacher 或显式背景 anchor，而是在 Human3R 前馈流式框架内加入 `A_corr_t` correction token。该 token 进入 decoder 后同时产生 pose-token residual 和 human-token residual，分别修正 camera pose 和人体 latent；GT 只用于 loss、metric 和可视化 overlay，不参与 inference。

V9 下一步重点是清理复现实验入口、扩大数据验证、完善 camera / SMPL / MPJPE 等指标，并继续检查 AvatarReX / THUman 坐标系统一问题。

## 当前文档

| 文档 | 内容 |
|---|---|
| [V9 当前入口](v9/README.md) | 当前主线、保留权重和 V9 起点 |
| [V8 历史归档](archive_v8/README.md) | V8.1-V8.9 设计、坐标系、训练和失败复盘 |
| [V7 历史归档](archive_v7/README.md) | V7 后处理式 correction / implicit token adapter 记录 |
| [V2-V6 历史归档](archive_v2_v6/README.md) | 旧 ShotToken / AnchorToken / V6 记录 |
| [训练代码入口](train_code.md) | 训练代码说明入口 |

## 历史分水岭

V2-V6 文档和报告已归档到：

```text
docs/movie3r/archive_v2_v6/
```

V7 文档和运行脚本已归档到：

```text
docs/movie3r/archive_v7/
scripts/archive_v7/
```

V8 文档已归档到：

```text
docs/movie3r/archive_v8/
```

归档内容保留用于复盘，不再代表当前主线。

## 当前代码状态

模型、训练、推理代码仍保留历史 V2-V8 实验路径以便复盘和兼容旧 checkpoint。新的整理和复现实验从 V9 文档入口继续。
