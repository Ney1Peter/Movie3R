# Movie3R V7

## 阶段定位

V7 当前不是一个已经确定的模型方案，而是新的调研阶段。当前候选主线是 online human-scene pose correction：冻结或基本冻结 Human3R 主干，在每帧 forward 末端增加轻量 correction head，预测一个小的 SE(3) camera pose residual。

V2-V6 的主要工作围绕 ShotToken、background feature anchor、AnchorToken 和 pose-only camera adapter 展开。近期测试表明，这些方向没有完全命中当前最重要的失败场景：低纹理、弱背景特征、简单场景中的 shot boundary 偏移。

## 当前目标

V7 当前先做四件事：

1. 收集 Human3R 在低纹理 shot change 场景中的失败案例。
2. 区分低纹理失败和 RICH / AvatarReX 高纹理稳定场景之间的差异。
3. 验证 Human3R 输出中哪些 cue 在失败帧仍然可信，包括 human token、SMPL camera-frame joints、pointmap confidence 和 camera pose prior。
4. 在充分调研前，不急于实现复杂模型结构。

2026-05-25 更新：V7 已完成第一版 implicit human-scene token adapter 单 clip overfit 验证。结果显示，Human3R internal pose / human / scene / memory tokens 中存在可复现 offline teacher correction 的信号。当前还不能证明泛化，下一步应转向 MS-AIST `shot2` 多 clip train / val 验证。

## 当前约束

- 暂不继续把 V2-V6 作为主线扩展。
- 暂不默认背景特征匹配一定可靠。
- 暂不做 offline post-processing、chunk stitching、pose graph optimization 或 bundle adjustment。
- 当前先整理文档、数据、现象和失败案例，再实现最小 correction head。

## 相关文档

```text
docs/movie3r/current_research_context.md
docs/movie3r/v7/online_human_scene_pose_correction_plan.md
docs/movie3r/v7/human_scene_pose_correction_experiment_log.md
docs/movie3r/v7/implicit_token_adapter_validation.md
docs/movie3r/archive_v2_v6/README.md
tasklist/TODO.md
```

## 当前验证状态

当前阶段结论：

```text
单 clip overfit 已通过，证明 token 中有 correction 信号；
下一步必须做 multi-clip held-out validation，验证泛化性和 no-op 稳定性。
```
