# Movie3R V7 Archive

## 阶段定位

V7 已归档。该阶段不是一个最终模型方案，而是围绕 online human-scene pose correction、offline teacher pseudo labels、implicit token adapter 的调研分支。

V2-V6 的主要工作围绕 ShotToken、background feature anchor、AnchorToken 和 pose-only camera adapter 展开。近期测试表明，这些方向没有完全命中当前最重要的失败场景：低纹理、弱背景特征、简单场景中的 shot boundary 偏移。

## 原阶段目标

V7 当前先做四件事：

1. 收集 Human3R 在低纹理 shot change 场景中的失败案例。
2. 区分低纹理失败和 RICH / AvatarReX 高纹理稳定场景之间的差异。
3. 验证 Human3R 输出中哪些 cue 在失败帧仍然可信，包括 human token、SMPL camera-frame joints、pointmap confidence 和 camera pose prior。
4. 在充分调研前，不急于实现复杂模型结构。

2026-05-25 更新：V7 已完成第一版 implicit human-scene token adapter 单 clip overfit 验证。结果显示，Human3R internal pose / human / scene / memory tokens 中存在可复现 offline teacher correction 的信号。当前还不能证明泛化，下一步应转向 MS-AIST `shot2` 多 clip train / val 验证。

2026-05-25 追加更新：MS-AIST `shot2` Stage-A 初轮跑了前 12 个候选，11 个完成 pipeline，质量门控接受 2 个。accepted labels 上 student overfit 仍通过，但 teacher pseudo label 可用率偏低；手动检查还确认候选中混有无明显跳变和多人样本，已补充 score filter 和 single-person filter。因此下一步应先扩大筛选或改进 teacher，再进入 20/5 held-out 训练。

## 归档结论

2026-05-27 更新：V7 不再作为当前主线推进。近期实验表明，后处理式 floor / human / scene correction 可以作为诊断工具，但在真实视频上容易受 SMPL 检测、floor/background 可靠性和参考帧选择影响，不适合作为下一阶段主方案。后续从 V8 重新定义问题和方法，不再沿用 V7 的后处理式改进路线。

## 原阶段约束

- 暂不继续把 V2-V6 作为主线扩展。
- 暂不默认背景特征匹配一定可靠。
- 原计划暂不做 offline post-processing、chunk stitching、pose graph optimization 或 bundle adjustment；实际 V7 后续补充了若干 offline teacher / correction 诊断脚本，现一并归档。
- 当前先整理文档、数据、现象和失败案例，再实现最小 correction head。

## 相关文档

```text
docs/movie3r/current_research_context.md
docs/movie3r/archive_v7/online_human_scene_pose_correction_plan.md
docs/movie3r/archive_v7/human_scene_pose_correction_experiment_log.md
docs/movie3r/archive_v7/implicit_token_adapter_validation.md
docs/movie3r/archive_v2_v6/README.md
tasklist/TODO.md
```

## 当前验证状态

当前阶段结论：

```text
单 clip overfit 已通过，证明 token 中有 correction 信号；
MS-AIST Stage-A 初轮显示 teacher label 质量是当前瓶颈；
下一步必须先得到足够 accepted pseudo labels，再做 multi-clip held-out validation。
```
