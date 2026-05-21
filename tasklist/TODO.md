# Movie3R TODO

更新时间：2026-05-19

## 当前主线：V7 调研阶段

当前项目已经从 V2-V6 的 ShotToken / background AnchorToken 路线切换到 V7 调研阶段。

V7 目前不急于定义新模型或新训练路线。当前 TODO 只记录调研和整理工作。

## 当前待办

- [ ] 收集低纹理、弱背景特征、简单场景中的 Human3R shot-boundary 失败案例。
- [ ] 记录每个失败案例的原视频、boundary 位置、原版 Human3R 输出和可视化结论。
- [ ] 对比 RICH / AvatarReX 稳定样本和低纹理失败样本的差异。
- [ ] 梳理哪些现象是第一帧偏移，哪些是后续累计漂移，哪些是人体/相机/背景分支不一致。
- [ ] 验证 camera-frame SMPL joints / human token / pointmap confidence 在失败帧中是否仍有可用信号。
- [ ] 评估 V7 候选方向：online human-scene pose correction。
- [ ] 在有足够失败案例和可信 cue 诊断前，不新增复杂 V7 模型实现。

## 历史归档

旧 V6 AnchorToken TODO 已归档到：

```text
tasklist/archive/TODO_anchor_v6_20260519.md
```

更早 TODO 和工作记录保留在：

```text
tasklist/archive/
```
