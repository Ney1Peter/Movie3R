# Movie3R TODO

更新时间：2026-05-25

## 当前主线：V7 调研阶段

当前项目已经从 V2-V6 的 ShotToken / background AnchorToken 路线切换到 V7 调研阶段。

V7 当前候选主线已经收敛到 offline human-scene geometry teacher -> causal implicit token student。单 clip overfit sanity check 已通过，下一步重点是 multi-clip held-out validation。

## 当前待办

- [x] 完成 H36M 两个 clip 的 offline teacher pseudo label 和 implicit token student 单 clip overfit。
- [x] 导出 viewer-ready corrected output，用 corrected 点云/人体叠加 raw camera 检查效果。
- [x] 整理 MS-AIST `shot2` 99 个 clip 的 staged pilot manifest。
- [x] Stage A：前 12 个 `shot2` 候选已完成 raw / teacher / token pipeline 和质量门控。
- [x] 新增 Stage-A quality gate，当前接受 2 / 12 个 pseudo labels。
- [x] 显式过滤疑似无跳变样本和多人样本：候选默认 `score >= 0.2`，quality gate 默认要求全程单人 SMPL。
- [ ] 扩大候选池或改进 teacher，先获得足够 accepted pseudo labels。
- [ ] Stage B：使用筛选后的 20 train + 5 val clips 训练 multi-clip token adapter，验证 held-out 泛化。
- [ ] 继续比较 `human_scene` / `human` / `scene` / `pose` / `all` ablation。
- [ ] 加入正常帧 no-op 约束，防止 adapter 在非 boundary 帧乱修。
- [ ] 在 held-out viewer 中人工检查 corrected camera / pointcloud / human mesh 是否比 raw 更自然。
- [ ] 如果 20/5 正向，再扩展到完整 shot2：80 train + 19 val。
- [ ] 根据 multi-clip 结果决定是否接入正式 Human3R forward 训练。

## 存储待办

- [ ] 训练主数据只保留 `v7_tokens.npz`、`pseudo_gt_labels.npz`、summary / metrics json。
- [ ] 完整 Human3R saved-output 只保留少量 debug / viewer 样本。
- [ ] corrected viewer output 使用 hardlink / symlink 复用 `color/depth/conf/smpl`，只新写 `camera/*.npz`。

## 历史归档

旧 V6 AnchorToken TODO 已归档到：

```text
tasklist/archive/TODO_anchor_v6_20260519.md
```

更早 TODO 和工作记录保留在：

```text
tasklist/archive/
```
