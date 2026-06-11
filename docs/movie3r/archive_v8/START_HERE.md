# V8 Start Here

## 一句话目标

Movie3R 的目标是改善 Human3R 在低纹理、弱背景特征、简单场景的 shot boundary 附近出现的相机位姿跳变/漂移问题，重点是跳变后的第一帧到前几帧。

## 背景动机

Human3R 在很多纹理丰富的视频上表现稳定，例如 RICH / AvatarReX 这类背景、人体纹理和相机信息较充分的数据。但在低纹理、弱背景、简单室内或电影裁剪片段中，镜头硬切后第一帧经常出现世界坐标/相机位姿明显偏移。

这说明问题不是“所有 shot change 都会失败”，而是更接近：

```text
当 Human3R 在 shot boundary 处缺少可靠场景匹配证据时，内部时序状态或相机估计可能被新镜头首帧拉偏。
```

V8 要重新从这个失败模式出发，不再默认沿用之前的 post-processing correction 或 pseudo-label teacher。

## 2026-05-30 关键更新：V8.1 raw-camera overfit 已跑通

V8.1 已经完成一个成功的 UniCon-style decoder-in pose prompt 单样本 overfit：

```text
sample:
  AvatarReX AABB 22010710 -> 22053923, start_frame=0

result:
  corrected trans err = 0.0075
  corrected rot err   = 0.0617 deg

viewer:
  http://127.0.0.1:8112
```

这次成功的关键不是换模型结构，而是修正训练 target 的坐标系。

以后使用 AvatarReX 做 V8.1 pose correction 时必须注意：

1. V8 当前统一训练目录是 `/data/wangzheng/iJCV-CODE/data/Training`；旧的 `/data/wangzheng/iJCV-CODE/data/training` 和 `Avatarrex_output/Training` 只作为兼容 symlink 保留。
2. 不要用 `/data/wangzheng/iJCV-CODE/data/Training/<group>/<seq>/cam/*.npz` 里的 processed `camera_pose` 作为最终监督 target。它是给 SMPL/depth 预处理和数据组织用的相机坐标，不是 V8 pose correction 的监督坐标；直接拿它算 loss 或画 GT camera 会导致坐标系错位。
3. 正确 target 来自 raw calibration：

```text
/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/calibration_full.json
/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zxc/calibration_full.json
/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr/calibration_full.json
X_cam = R_w2c @ X_world + T_w2c
R_c2w = R_w2c.T
t_c2w = -R_w2c.T @ T_w2c
T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i
```

4. 训练、指标和 GT camera 可视化都必须使用同一个 target：

```text
raw_camera_pose_i = raw calibration c2w for camera i
T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i
```

如果看到 B 视角 `y-axis ~= -1`，说明又用了错误的 processed pose。

5. `Avatarrex_output/depth/*.npy` 是 DA3 / monocular pseudo-depth，不是 metric GT depth。V8.1 pose-only 训练和坐标 sanity check 必须使用：

```text
load_da3_depth=False
```

Human3R 自己预测的 pointmap/depth 可以作为模型输出或可视化 cue 看，但 DA3 depth 不能用来验证跨相机世界坐标是否正确。

## 2026-05-30 关键更新：V8.1 小批量 C 版本效果可用

在单样本 overfit 之后，已经跑通 10 个 AvatarReX AABB clip 的小批量 C 版本训练：

```text
C version:
  A_corr_t 进入 decoder
  + V8.1 prompt / residual / gate branch
  + 微调原 pose head
  + 冻结 backbone / decoder / scene head / human head
```

训练设置：

```text
train:
  10 fixed AABB clips
  1000 steps
  batch size = 1
  raw calibration pose target
  load_da3_depth=False
```

测试结果：

```text
same-pair held-out:
  raw B-frame: 178.4 deg / 3.56
  C B-frame:     1.70 deg / 0.043

new-pair held-out:
  raw B-frame: 156.4 deg / 3.31
  C B-frame:    15.0 deg / 0.39
```

可视化检查：

```text
same-pair viewer:
  http://127.0.0.1:8115

new-pair viewer:
  http://127.0.0.1:8116
```

经验结论：

- C 版本已经足够作为下一步扩大数据量训练的 baseline。
- same-pair 泛化很好，new-pair 也明显优于 raw Human3R，但还有剩余误差。
- 当前 `gate_mean` 接近 0，说明这轮成功主要可能由 pose head 微调承担；后续要继续监控 prompt/residual/gate 是否真正起作用。
- 下一步应该扩大到更多 AABB clips 和更多 camera pairs，同时保留 same-pair / new-pair held-out 测试。

大规模训练计划见：

```text
docs/movie3r/archive_v8/v8_1_large_scale_training_plan.md
```

## 2026-05-31 关键更新：Stage A 10k 已完成

第一轮 Stage A 大规模训练已经跑完：

```text
checkpoint:
  /tmp/movie3r_v8_1_pose_prompt_posehead_stage_a_10k_nodepth_rawpose_gpu4/checkpoint-final.pth

eval outputs:
  output/v8_1_stage_a_eval/
```

核心观察：

- 训练正常完成，没有 NaN、OOM 或 traceback。
- same-pair held-out 结果可用。
- new-pair held-out 明显更难，B 段误差更大。
- `gate_mean` 约为 `0.0002`，基本塌到 0。
- 因此这轮更像验证了训练闭环和 pose head 微调，而不是证明 UniCon-style residual/gate branch 已经真正起作用。

详细训练指标、四组评估指标和下一步 ablation 见：

```text
docs/movie3r/archive_v8/v8_1_large_scale_training_plan.md
```

## 2026-06-01 关键更新：Prompt-only ablation 已完成

这次实验冻结原版 Human3R pose head，只训练：

```text
v8_pose_prompt.*
v8_pose_residual_head.*
```

对应 checkpoint：

```text
output/v8_1_train_runs/v8_1_pose_prompt_stage_a_10k_nodepth_rawpose/checkpoint-final.pth
```

核心观察：

- 训练集上 corrected pose 明显优于 raw pose，说明 correct pose token 分支本身可以学到东西。
- `test_new` 前 20 个样本上，原版 Human3R 的 B 段误差是 `2.864m / 126.50deg`，prompt-only V8.1 是 `0.867m / 27.10deg`。
- 但是 `gate_mean` 仍然很小，`delta_norm` 很大，说明目前是“大 residual × 小 gate”的不健康形式。
- 下一步应该优先做 gate supervision、residual norm regularization、fixed-gate/no-gate ablation，而不是只扩大训练。

## 2026-06-02 关键更新：V8.2 Pose Relation Prompt

V8.1 的实验说明 decoder-in pose prompt 链路能工作，但也暴露出一个设计问题：当前的四个 body queries 本质上只是 learnable queries，并没有被显式监督成 pelvis / torso / left foot / right foot。因此不能简单把它们解释成稳定的人体部位 token。

V8.2 的新主线是把 `A_corr_t` 定义为 human-centric current-history pose relation prompt：

```text
当前 human / image / pose tokens
+ recurrent state memory
+ 上一帧 corrected pose / human anchors / correction token
+ drift / reliability cue
  -> A_corr_t

A_corr_t 进入 decoder
  -> refined A_corr_t
  -> drift score / gate head
  -> pose latent residual head
  -> corrected pose token
  -> pose head
  -> corrected camera pose
```

这更接近 UniCon3R 的范式：不是手工指定某个 token 一定等于脚或骨盆，而是构造一个专门学习“当前帧和历史世界是否对齐”的 relation prompt，并用显式 pose / drift 监督训练它。

详细设计见：

```text
docs/movie3r/archive_v8/v8_2_pose_relation_prompt_plan.md
```

当前第一版训练前置代码也已加入：

```text
src/dust3r/v8_pose_prompt.py
  V82PoseRelationPrompt
  V82PoseRelationResidualHead

src/dust3r/losses.py
  V82PoseRelationLoss

config/train_v8_2_pose_relation_small.yaml
```

当前 V8.2 只使用 3 个 relation tokens：

```text
semantic pose-scene context
explicit latent alignment cue
temporal correction momentum
```

第一版训练只启用 `L_pose_gt + L_drift_score/gate + L_improvement_margin + L_residual_small`，暂时不启用 pointmap/floor/contact/body-part auxiliary。

## 2026-06-03 关键更新：V8.3 Image-Only Pose Prompt

V8.2 Stage-B 在 AvatarReX dataloader eval 上有效，但严格改成普通 RGB 图片文件夹推理后，泛化明显变差。排查后发现，训练/测试 dataloader 路径中仍然保留了一些不适合真实 image-only 推理的输入信号：

```text
ray_map / ray_mask
shot_label
GT-like camera / geometry fields
```

这会让模型容易学到：

```text
第 2/3 帧要强修
```

而不是真正学到：

```text
根据当前人体、历史人体/pose memory 和图像 token 判断该怎么修。
```

V8.3 的目标是：

```text
4-frame image-only pose correction
模型 forward 只看 RGB 产生的 tokens 和内部 memory
GT 只用于 loss / evaluation
```

当前仍然只做 4 帧训练，先不扩展到长序列，因为 Human3R 原训练本身也是 4 帧视频式训练。V8.3 的重点是先把短序列 image-only correction 跑通。

详细计划见：

```text
docs/movie3r/archive_v8/v8_3_image_only_pose_prompt_plan.md
```

## 当前代码状态

V7 已归档，V8 当前主线从 V8.1 UniCon-style decoder-in pose prompt 推进到 V8.2 pose relation prompt。当前原版 Human3R 推理仍可正常运行，V8.1 训练代码用于 pose correction 实验；V8.2 已有第一版训练前置代码，但还没有正式开始训练。

保留历史代码的原因是兼容旧 checkpoint 和复现实验，不代表当前主线：

```text
src/dust3r/model.py
src/dust3r/v7_pose_adapter.py
scripts/archive_v7/
docs/movie3r/archive_v7/
docs/movie3r/archive_v2_v6/
```

新工作优先从 V8 文档和新的实验脚本开始，不要默认继续扩展 `scripts/archive_v7/`。

## V2-V6 做过什么，为什么不是当前主线

V2-V6 主要围绕 ShotToken、background AnchorToken、pose-only adapter、decoder token 注入和 LoRA/residual adapter 展开。

主要结论：

- Global ShotToken 不足以稳定解决 boundary pose drift。
- 把 shot / anchor 信号注入 decoder 容易影响 dense reconstruction，不一定只修 camera pose。
- Background AnchorToken / feature matching 在 RICH、AvatarReX 这类高纹理场景可以提供线索，但这些场景中原版 Human3R 往往已经稳定，收益不明显。
- 真正容易失败的低纹理场景，恰好缺少可靠背景特征，导致背景 anchor 假设不稳健。
- V2-V6 提供了很多诊断经验，但没有命中当前最关键的低纹理失败模式。

历史记录在：

```text
docs/movie3r/archive_v2_v6/
```

## V7 做过什么

V7 转向 human/scene pose correction，核心尝试是用 offline teacher 生成 camera correction，再训练轻量 implicit token adapter 学习该 correction。

做过的主要工作：

- 构建 MS-AIST shot-change clips，包含 refined `shot2/shot3/shot4` 30-frame clips。
- 用 Human3R saved outputs 估计 floor normal，并对 post-shot 帧做 floor leveling。
- 使用 SMPL stable joints / foot joints 做 human-anchor yaw + translation alignment。
- 尝试加入 background scene Chamfer，形成 floor + human + scene hybrid correction。
- 生成 V7 pseudo labels：`target_delta_t`、`target_delta_rotvec`、`target_alpha`、`r_human`、`r_scene`。
- dump Human3R internal pose / scene / human / memory tokens，训练 implicit token adapter。
- 测试 token-only adapter，即不显式输入 raw camera pose，只用 tokens 预测 correction。
- 对 PKUHuman、H36M、MS-AIST、电影 clip 做原版 Human3R 与 corrected output 可视化对比。

归档位置：

```text
docs/movie3r/archive_v7/
scripts/archive_v7/
```

## V7 为什么归档

V7 的诊断价值明确，但不适合作为下一阶段主线。

主要原因：

- Correction 是后处理式的，依赖 Human3R saved output、SMPL 检测、floor/background 点云估计和参考帧选择。
- Offline teacher 常依赖 post-shot 信息或稳定窗口，这不符合最终在线/因果推理目标。
- Floor normal 在低纹理、遮挡或人物占画面较大时会不稳定，可能产生过大的旋转。
- Human anchor 依赖 SMPL 检测；真实电影 clip 中经常出现参考帧或目标帧无人/漏检，teacher 直接失败。
- Scene/background Chamfer 在弱背景或低置信点不足时不可用，或者会给出不可靠约束。
- MS-AIST Stage-A pseudo label 可用率和质量受筛选影响很大，teacher label 质量成为瓶颈。
- Adapter 单 clip overfit 可以证明 tokens 中有 correction 信号，但 held-out 泛化不稳定；例如训练内样本可拟合，未见样本仍会出现较大平移/旋转误差。
- 继续扩大 V7 会把精力放在修 teacher、修筛选、修后处理上，而不是解决原始模型在低纹理 boundary 的根因。

因此 V8 不应继续默认采用：

```text
offline post-processing correction
post-shot stable window
explicit floor / SMPL / background anchor teacher
BA / pose graph / chunk stitching
V7 pseudo-label training loop
```

## 最近有代表性的测试观察

- RICH / AvatarReX：纹理丰富，原版 Human3R 通常较稳定，不能充分暴露目标失败模式。
- MS-AIST refined clips：可构造大量 shot boundary clips，但 pseudo-label teacher 成功率、单人筛选和质量门控是瓶颈。
- H36M `h36_new.mp4`：可以看到 shot-change 三帧对比；human+scene hybrid 比 pure human correction 更保守，但仍属于后处理诊断。
- PKUHuman temporal stitching：用于验证“时间拼接 shot-change”，不是左右拼接；原版 Human3R 可用于快速观察 boundary 行为。
- 电影 clip `clip02/clip03`：原版 Human3R 可跑全帧；在部分边界帧 corrected teacher 会因无人体检测或背景点不足而失败，说明 V7 teacher 不够可靠。

## V8 应该从哪里开始

V8 的第一步不是写新 adapter，而是重新明确问题定义和最小实验：

- 目标帧：shot boundary 前一帧、跳变后第一帧、跳变后第二帧。
- 目标现象：相机/world gauge 在跳变后首帧发生不合理偏移。
- 目标数据：低纹理、弱背景、简单场景，以及一小组高纹理稳定样本作为对照。
- 目标约束：最终方法应尽量在线/因果，不依赖未来帧、stable window、BA 或显式 decoded floor/SMPL anchor。
- 初始输出：先建立 V8 failure set、stable set、评价方式和 baseline，而不是直接训练复杂模型。

## 新对话建议先读

```text
docs/movie3r/archive_v8/START_HERE.md
docs/movie3r/archive_v8/README.md
docs/movie3r/archive_v8/current_research_context.md
docs/movie3r/archive_v7/README.md
docs/movie3r/archive_v2_v6/README.md
tasklist/TODO.md
```
