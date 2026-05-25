# Current Research Context: Low-Texture Shot Changes

## 1. 当前观察

近期测试表明，Human3R 在一些简单视频或低纹理场景中，遇到镜头不连续时会出现明显偏移。偏移通常集中在镜头变化后的第一帧，后续帧可能逐步恢复，但 boundary frame 的错误已经足以影响整体可视化和轨迹连续性。

与此相对，在 RICH / AvatarReX 这类纹理更丰富、相机和人体信息更充分的数据上，Human3R 表现明显更稳定。我们在 Guitar、Juggle 等 RICH 子集上测试低 overlap 和较大相机跳变样本时，原版 Human3R 仍然经常能保持较好的连续性，几乎不出现我们在简单低纹理视频中观察到的明显偏移。

这说明当前失败模式不是“所有镜头跳变都会导致 Human3R 失败”，而更接近：

```text
当场景纹理不足、背景特征弱、可稳定匹配的环境锚点少时，Human3R 在 shot boundary 处更容易发生相机/世界对齐偏移。
```

## 2. 对旧假设的修正

之前 V2-V6 的多轮探索主要围绕 ShotToken、背景特征匹配、AnchorToken 和 pose-only adapter 展开。这个方向隐含了一个重要假设：

```text
镜头变化处可以从场景/背景中提取足够可靠的局部特征锚点，用这些锚点辅助相机重定位。
```

新的测试结果表明，这个假设并不稳健。真正容易出问题的低纹理场景，往往恰好缺少可靠的背景特征；反而是 RICH 这类背景和人体信息都较丰富的数据，Human3R 本身已经足够稳，外部背景 anchor 的收益不明显。

因此，不能继续默认“只靠背景特征匹配就能解决 shot boundary 偏移”。XFeat / patch matching / local scene anchor 可以作为诊断或补充信息，但不应该再被视为唯一或主要依据。

## 3. 当前更清晰的问题边界

Movie3R 当前需要重新聚焦的问题是：

```text
面向低纹理、弱背景特征、简单场景中的镜头不连续，改善 Human3R 在 shot boundary 附近的稳定性。
```

这个问题边界比早期设想更窄，也更符合实际观测：

- RICH 等高纹理场景不是主要失败场景。
- 低纹理场景才是当前需要重点解释和改进的场景。
- 背景特征 anchor 在这类场景中经常不可用或不可靠。
- 人体本身是视频中持续存在的稳定对象，未来调研需要重新评估“人作为锚点”的价值。

## 4. V2-V6 的定位

V2-V6 的工作仍然有记录价值，但它们已经不再代表当前主线。

这些阶段主要回答了以下问题：

- global ShotToken 是否足够稳定。
- ShotToken 进入 decoder 是否会污染 dense reconstruction。
- local background AnchorToken 是否能作为重定位证据。
- pose-only / camera-only adapter 是否比 full-decoder token 更安全。
- RICH / AvatarReX 上的 overlap、anchor 和 shot boundary 样本是否能暴露明显失败。

这些探索的结论是：旧方向在工程上提供了很多诊断经验，但没有直接命中当前更重要的低纹理失败场景。因此，V2-V6 文档已作为历史分支归档到：

```text
docs/movie3r/archive_v2_v6/
```

归档不代表删除，也不代表这些结果无效。它们作为分水岭之前的调研记录保留，用于避免重复走同样的路线。

## 5. 当前状态

当前项目处于重新调研阶段。这个阶段在文档中记为 V7。

2026-05-25 更新：V7 已完成第一版 implicit human-scene token adapter overfit 验证。该验证显示，在两个 H36M shot-change clip 上，只读取 Human3R internal tokens 的轻量 adapter 可以复现 offline human-scene teacher 生成的 camera pose correction。这说明 token 中存在可用 correction 信号，但目前仍只是单 clip overfit，下一步必须做 MS-AIST `shot2` multi-clip held-out validation 来验证泛化性。

2026-05-25 追加更新：MS-AIST `shot2` Stage-A 初轮已跑前 12 个候选，11 个完成 raw / teacher / token pipeline，1 个因 stable window 内 SMPL 漏检导致 teacher 失败。质量门控后只有 2 个 pseudo labels 被接受，accepted clip 的 pooled-token student overfit 仍然通过。手动检查确认 `shot2` 候选中混有无明显跳变和多人样本，现已补充 detection score 过滤和 SMPL 单人检测过滤。这说明当前瓶颈主要是 offline teacher label 质量和筛选命中率，不应在未筛选的 pseudo labels 上直接做 20/5 multi-clip 训练。

本文记录目前观察到的问题变化和方向边界。V7 当前候选思路见：

```text
docs/movie3r/v7/online_human_scene_pose_correction_plan.md
docs/movie3r/v7/implicit_token_adapter_validation.md
```

后续需要继续通过更多低纹理样本、简单背景样本和真实失败案例来确认 Human3R 的具体失效机制，再决定是否实现该方向的最小 correction head。
