# Movie3R TODO

更新时间：2026-05-27

## 当前主线：V8 调研准备阶段

当前项目已经从 V2-V7 的 ShotToken / background AnchorToken / 后处理式 correction 路线切换到 V8 调研准备阶段。

V7 的 offline human-scene geometry teacher -> causal implicit token student 路线已归档。后续不再默认推进 post-processing correction、stable-window pseudo labels 或 V7 token student 训练闭环。

## 当前待办

- [x] 完成 H36M 两个 clip 的 offline teacher pseudo label 和 implicit token student 单 clip overfit。
- [x] 导出 viewer-ready corrected output，用 corrected 点云/人体叠加 raw camera 检查效果。
- [x] 整理 MS-AIST `shot2` 99 个 clip 的 staged pilot manifest。
- [x] Stage A：前 12 个 `shot2` 候选已完成 raw / teacher / token pipeline 和质量门控。
- [x] 新增 Stage-A quality gate，当前接受 2 / 12 个 pseudo labels。
- [x] 显式过滤疑似无跳变样本和多人样本：候选默认 `score >= 0.2`，quality gate 默认要求 teacher 使用窗口内单人 SMPL。
- [x] Stage-A runner 默认跳过已标记 failed 的 case，避免扩大候选池时反复重跑坏样本。
- [x] 归档 V7 文档和运行脚本。
- [ ] 明确 V8 的问题定义、输入约束和不使用的 V7 假设。
- [ ] 收集 V8 第一批代表性 failure / stable cases，区分低纹理失败、纹理丰富稳定和检测失败。
- [ ] 设计不依赖后处理 teacher 的最小实验。
- [ ] 确定 V8 是否需要改模型、改数据、改训练目标或只做诊断。

## 存储待办

- [ ] V7 训练主数据只保留必要 manifest、summary / metrics json 和少量 debug 样本。
- [ ] 完整 Human3R saved-output 只保留近期对比需要的 inspection 样本。
- [ ] 新增 V8 输出目录规范，避免继续混用 V7 stage-a / training / inspection 命名。

## 历史归档

V7 文档和脚本已归档到：

```text
docs/movie3r/archive_v7/
scripts/archive_v7/
```

旧 V6 AnchorToken TODO 已归档到：

```text
tasklist/archive/TODO_anchor_v6_20260519.md
```

更早 TODO 和工作记录保留在：

```text
tasklist/archive/
```
