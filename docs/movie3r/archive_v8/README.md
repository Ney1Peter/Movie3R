# Movie3R V8 Archive

本目录是 V8 历史归档。当前主线已经切到 [V9](../v9/README.md)，V8 文档只用于复盘 V8.1-V8.9 的实验路径、坐标系问题和最终的 implicit human-pose correction 起点。

如果需要回看 V8 的完整上下文，先读：

```text
docs/movie3r/archive_v8/START_HERE.md
```

## 阶段定位

V8 是新的调研入口。V7 的后处理式 floor / human / scene correction 与 implicit token adapter 路线已经归档，不再作为当前主线继续扩展。

V8 的核心背景是：Human3R 在纹理丰富数据上通常稳定，但在低纹理、弱背景特征、简单场景的 shot boundary 后第一帧容易出现相机/world gauge 偏移。我们要针对这个具体失败模式重新设计实验。

## 当前起点

V8 需要重新定义 shot-change 场景下的目标、约束和最小实验，不默认沿用 V7 的 offline teacher、post-processing correction、stable window 或 pseudo-label 生成流程。

2026-05-30 更新：V8.1 已经跑通一个关键 sanity check。使用 raw calibration camera pose 作为监督 target 后，UniCon-style decoder-in pose prompt 可以在一个 AvatarReX AABB 样本上 overfit，并把后两帧 B-camera pose 修到正确方向。这个结果证明当前 decoder-in prompt / pose-token residual / original pose head 链路是通的。

2026-06-02 更新：V8.2 设计已整理为新的主线文档，并已加入第一版训练前置代码。核心变化是把 `A_corr_t` 定义为 human-centric current-history pose relation prompt，而不是简单的四个固定人体部位 token。它更接近 UniCon3R 的 contact relation prompt：decoder-in relation token + 显式 drift/alignment 监督 + residual pose latent refinement。

2026-06-11 更新：V8.9 当前最佳版本已从显式 `smpl_transl` 诊断分支推进到 implicit human latent correction。`A_corr_t` 仍按 UniCon-style 在 decoder 前由 image / pose / human tokens、state memory、pose memory、上一帧 corr/delta/gate 构造；decoder 后同时产生 pose token residual 和 human token residual。pose residual 加到 pose token 后走原 pose head，human residual 加到 decoder human token 后再走原 Human3R human head。GT camera / SMPL 只用于 loss、metric 和红色 overlay，不参与 inference。

第一阶段建议继续做三件事：

1. 建立低纹理 boundary failure set 和高纹理 stable control set。
2. 明确不依赖后处理 teacher 的评价指标和可视化协议。
3. 再决定是否需要模型结构、训练目标或数据构造上的修改。

## V8.1 AvatarReX 坐标系规则

使用 AvatarReX 做 V8 pose correction 时：

- 当前统一训练目录是 `/data/wangzheng/iJCV-CODE/data/Training`；旧的 `/data/wangzheng/iJCV-CODE/data/training` 和 `Avatarrex_output/Training` 只是兼容 symlink。
- pose supervision 使用 `raw_camera_pose`，来自 raw AvatarReX calibration：
  `/data/wangzheng/iJCV-CODE/data/avatarrex_{lbn1,zxc,zzr}/calibration_full.json`。
- 正确 target 是相对第 0 帧的 raw calibration camera：
  `T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i`。
- 不要把 `/data/wangzheng/iJCV-CODE/data/Training/<group>/<seq>/cam/*.npz` 的 processed `camera_pose` 当作最终监督 target 或 GT camera 可视化；它是给 SMPL/depth 预处理和数据组织用的坐标。
- 指标和 viewer 中的 GT camera 也必须用 raw calibration target，再对齐到 saved Human3R output 的第 0 帧 viewer gauge。
- V8.1 pose-only 训练使用 `load_da3_depth=False`。
- `Avatarrex_output/depth/*.npy` 是 DA3 pseudo-depth，不是跨相机 metric GT depth，不能用来验证世界坐标是否对齐。

## 初始约束

- 不以 offline 后处理 correction 作为主方案。
- 不默认依赖 post-shot stable window、BA、pose graph 或显式 floor/SMPL anchor。
- 先明确新的失败模式、可用输入和训练目标，再新增模型结构。
- V7 归档内容只作为诊断经验和负例参考。

## 相关归档

```text
docs/movie3r/archive_v7/
scripts/archive_v7/
```

## 当前 V8 文档

```text
docs/movie3r/archive_v8/v8_2_pose_relation_prompt_plan.md
docs/movie3r/archive_v8/v8_9_implicit_human_pose_token.md
docs/movie3r/archive_v8/v8_7_head_lora_finetune_plan.md
docs/movie3r/archive_v8/v8_1_unicon_style_implementation_plan.md
docs/movie3r/archive_v8/v8_1_token_extraction_validation_plan.md
docs/movie3r/archive_v8/v8_1_large_scale_training_plan.md
docs/movie3r/archive_v8/report_human3r_unicon3r_pose_prompt_intro.md
```

## 历史速览

V2-V6 尝试 ShotToken / background AnchorToken / pose adapter，但低纹理场景缺少可靠背景 anchor，方向不稳健。

V7 尝试 offline floor / human / scene correction teacher 和 implicit token adapter。它能作为诊断工具，但依赖 SMPL、floor/background 点云、参考帧和 post-shot 信息，真实视频上不够可靠，因此归档。
