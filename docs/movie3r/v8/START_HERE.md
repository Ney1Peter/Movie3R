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

## 当前代码状态

V7 已归档，V8 尚未确定新的模型结构或训练方案。当前原版 Human3R 推理仍可正常运行。

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
docs/movie3r/v8/START_HERE.md
docs/movie3r/v8/README.md
docs/movie3r/current_research_context.md
docs/movie3r/archive_v7/README.md
docs/movie3r/archive_v2_v6/README.md
tasklist/TODO.md
```
