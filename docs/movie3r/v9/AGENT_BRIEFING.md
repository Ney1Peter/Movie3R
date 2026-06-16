# V9 Agent Briefing

这份文档给新的 AI 对话或新的协作者使用。目标是读完后能知道 Movie3R 当前在做什么、哪些方向已经失败、当前最可信的版本是什么、下一步应该怎么接。

## 一句话目标

Movie3R 现在要解决的是：在 Human3R 的前馈流式框架内，改善镜头跳变或跨视角组合时 camera pose 和人体 SMPL 在统一 world 里的错位问题。

核心约束：

- 推理时只能使用当前帧、上一帧 recurrent state、上一帧 correction memory；不能看未来帧，也不能用全局 GT。
- GT camera / SMPL 只能用于训练 loss、测试指标和可视化 overlay。
- 当前重点是让模型自己通过 correction token 学会修正，而不是靠后处理脚本手动对齐。

## 当前主线

当前最好版本来自 V8.9，V9 以它为起点：

```text
A_corr_t enters the decoder as a UniCon-style streaming relation token.
The refined correction token predicts both pose-token and human-token residuals.
Camera correction is applied before the original pose head.
Human correction is applied before the original Human3R human head.
```

更通俗地说：

```text
输入当前帧
  -> Human3R 正常提取 image / pose / human tokens 和 recurrent state
  -> 新增 A_corr_t correction token
  -> A_corr_t 和原 token 一起进 decoder
  -> decoder 后得到 refined A_corr_t
  -> pose residual head 修正 pose token
  -> human latent residual head 修正 human token
  -> 原版 pose head / human head 输出 corrected camera 和 corrected SMPL
```

当前不是显式去找 pelvis / torso / left foot / right foot，也不是直接手工移动 `smpl_transl`。早期显式人体平移分支只用于诊断，证明“人体也必须修”；V9 目标是隐式修正 human latent。

## 关键代码位置

| 文件 | 作用 |
|---|---|
| `src/dust3r/v8_pose_prompt.py` | `V82PoseRelationPrompt`、pose residual head、human latent correction head |
| `src/dust3r/model.py` | 把 `A_corr_t` 接入 decoder，并在 pose head / human head 前应用 residual |
| `src/dust3r/losses.py` | `V82PoseRelationLoss`，包含 pose、gate、improvement、human translation 等监督 |
| `src/dust3r/datasets/avatarrex.py` | AvatarReX / THUman dataloader、raw camera pose、resize/no-crop、SMPL 坐标处理 |
| `config/train_v8_9_avatarrex_lbn1_single_aabb_no_crop_from_human3r.yaml` | 以原版 Human3R 为初始化的 AvatarReX 单 clip 配置参考 |
| `docs/movie3r/v9/GUARDRAILS.md` | 坐标系和可视化规则，修改前必须看 |

## 已经尝试过什么

V2-V6：ShotToken / background AnchorToken / local feature anchor。

结论：背景 anchor 在高纹理数据上能提供线索，但高纹理数据里原版 Human3R 本身往往已经稳定；真正困难的低纹理或弱背景场景反而没有可靠背景 anchor。因此这条路线归档。

V7：offline teacher、post-processing floor / human / scene correction、implicit adapter。

结论：后处理能作为诊断工具，但依赖参考帧、SMPL 检测、floor/background 可靠性，不满足最终前馈流式目标。归档。

V8.1：decoder-in pose prompt，先只修 camera pose。

结论：在修正 AvatarReX raw camera 坐标系后，单样本 overfit 成功，证明 pose token residual 链路可行。重要踩坑：不能用 processed `cam/*.npz` 当 AvatarReX pose loss 的最终 GT。

V8.2-V8.4：更像 UniCon3R 的 relation prompt、gate、mixed AABB/AAAA benchmark。

结论：prompt-only 能降低部分 camera error，但一度受 dataloader 泄漏、错误 resize/crop、坐标系和 raw viewer 不一致影响。后续测试必须严格使用真实 inference 可用的信息。

V8.6：显式 `smpl_transl` correction。

结论：camera pose 修正后，人仍可能在深度/高度上错位；说明只修相机不够，还需要修 human branch。显式平移分支在单 clip 上有效，但不是最终形式。

V8.9：implicit human latent correction。

结论：在 AvatarReX 单 clip 上，从原版 Human3R 权重训练，camera 和 human 都能被修到接近 GT。该版本是 V9 的起点。

## 当前最重要的成功经验

1. 单修 camera pose 不够。Human3R 的人体输出里有自己的 `smpl_transl` / human latent，camera 对了但人仍可能浮起来或前后错位。
2. 显式修 `smpl_transl` 能证明问题可解，但最终更合理的是在 human latent 进入原 human head 前加 residual。
3. 坐标系是第一优先级。AvatarReX / THUman / viewer / Human3R raw output 必须使用同一套规则，否则 loss 很低也可能可视化完全错误。
4. 新训练必须从原版 Human3R checkpoint 初始化，除非明确做 continuation。不要从某个旧 V8/V9 实验权重继续训练后再声称是新方法能力。
5. 推理过程不能依赖 GT。可以保存 raw/corrected/GT 三套结果用于对比，但 GT 只能在 loss、metric、viewer overlay 中出现。
6. 如果快速运动人物被错误拉回历史位置，优先考虑 Trophies-style human-aware attention：保留原图给 human branch，看 scene/camera/memory 时对人体 patch 降权；不要简单把人从输入图像里抹掉。

## 当前数据状态和建议

当前可靠主线仍然优先使用：

- AvatarReX 预处理训练/测试数据。
- THUman 预处理训练/测试数据。

ASIT/AIST 相关数据可以保留为后续扩展，但不要默认混入 V9 大训练，除非先完成：

- SMPL vs SMPL-X 兼容检查。
- camera / SMPL 投影可视化检查。
- 与 AvatarReX / THUman 的 world gauge 统一检查。

## 下一步建议

1. 先用 V9 文档和 guardrails 清理复现入口。
2. 用原版 Human3R 权重复现 V8.9 AvatarReX 单 clip overfit。
3. 用 5 clip 版本验证 implicit human latent correction 是否稳定。
4. 再扩大到 AvatarReX + THUman，必须显式划分 train / test，并同时包含 AABB 和 AAAA。
5. 指标不要只看 loss。至少要看 corrected vs raw 的 camera trans/rot、human trans error、gate、delta norm，并配合正确 viewer。
6. LoRA 微调 pose head / human head 可以作为后续 ablation，但不要做全量 head 解冻作为主线。

## 新对话接手 Checklist

开始任何训练或可视化前，先确认：

- 已读 `docs/movie3r/v9/GUARDRAILS.md`。
- 可视化 raw Human3R 时，先单独跑原版 demo 或正确 raw saved output，确认坐标系一致。
- AvatarReX pose loss 用 `raw_camera_pose`，THUman 用官方 `cam/*.npz` c2w 作为 raw pose。
- `load_da3_depth=False`，不要把 DA3 pseudo-depth 当 metric depth。
- AvatarReX 全身可视化优先使用 `resize_mode='resize_only_16'` 或等价 no-crop 路径。
- 新实验从原版 Human3R checkpoint 初始化。
- checkpoint 只长期保留 best/final 或明确命名的里程碑。
