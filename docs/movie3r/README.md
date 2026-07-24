# Movie3R 研究文档入口

正式版本已经从散落的实验编号中分离。用户运行版本时只选择 V9、V12 或 V13；
`V10/V11/V14/V20` 继续作为历史实验编号保留在对应版本内部。

## 正式版本

| 入口 | 内容 |
|---|---|
| [版本总目录](../../versions/README.md) | 三个正式版本的定位、tag、权重和命令 |
| [V9 已训练版](../../versions/v9/README.md) | 学习式 AABB correction、LoRA 权重与训练复现 |
| [V12 单人版](../../versions/v12/README.md) | 当前 short-shot camera-human alignment 主版 |
| [V13 多人版](../../versions/v13/README.md) | GT-ID 多人 shared-Boundary 研究版 |
| [编号规则](BOUNDARY_VERSIONING.md) | 正式版本与历史实验编号的映射 |

## 版本内文档

- V12 的架构、组件消融、连续性、coupled-root 和最终审计均在
  [`versions/v12/docs/`](../../versions/v12/docs/README.md)。
- V12 的历史实验脚本均在
  [`versions/v12/experiments/`](../../versions/v12/experiments/README.md)。
- V13 的 GT-ID 多人结果、EgoBody 与 MultiHuman 数据说明均在
  [`versions/v13/docs/`](../../versions/v13/docs/README.md)。
- V9 的训练计划、方法和训练记录均在
  [`versions/v9/docs/`](../../versions/v9/docs/README.md)。

## 历史归档

- 2026-07-21 近期实验：[../../archive/20260721/README.md](../../archive/20260721/README.md)
- V8 历史：[archive_v8/README.md](archive_v8/README.md)
- V7 历史：[archive_v7/README.md](archive_v7/README.md)
- V2-V6 历史：[archive_v2_v6/README.md](archive_v2_v6/README.md)

归档内容没有删除。它们用于追溯研究过程，不应再被理解为当前可选版本。
