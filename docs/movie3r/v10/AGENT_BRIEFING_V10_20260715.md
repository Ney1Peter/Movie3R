# V10 新会话总览

日期：2026-07-15

这份文档给新的 AI 会话使用。目标是让新会话快速知道 Movie3R 当前在做什么，为什么从 V9 转到 V10，现在已经验证到哪里，接下来应该优先做什么。

## 1. 当前项目要解决的问题

Movie3R 面向的是单目、多分镜、镜头不连续的人体三维重建。

原版 Human3R/CUT3R 对连续镜头内的重建通常不错，比如同一个镜头连续几帧中，相机、人体、点云一般比较稳定。但是在分镜跳变后，新镜头段往往会落在另一个局部坐标系里，导致：

- A 段和 B 段的人体不在同一个世界位置；
- 相机位姿的全局关系不连续；
- 点云/背景也无法自然拼接；
- 如果对所有帧都做 correction，原本稳定的同镜头帧反而会被过度修正。

最终想做的是：

```text
输入单目 RGB 多分镜视频
逐帧、流式、前馈处理
不使用未来帧
不做全局 BA
输出同一个世界坐标系下的 camera + human + point cloud
```

严格流式的含义是：第 t 帧只能使用历史信息和当前帧，不能先跑完整段再回头修过去的帧。

## 2. 为什么 V9 暂时停下

V9 主要在 Human3R decoder 内部做 correct token / correction head / gate / loss 消融。

已经做过的结论：

- correct token 确实有用，no-correction-token 对照证明了 token 设计不是完全没贡献；
- semantic/alignment/momentum/human token 等组合都试过；
- token 层面能带来一定改善，但上限有限；
- 对稳定帧容易过度纠正，比如 AA 两帧原版 Human3R 本来对齐，V9 correction 后反而错开；
- 对 A/B 段的全局 gauge 关系仍然不够稳定。

关键问题是：token-level correction 在跳变发生前就试图修正内部 latent，它很优雅，但很难构造足够真实、足够大、带 GT 的训练数据。并且 token 信息已经被压缩，不一定包含足够强的几何约束。

所以当前主线从 V9 的 token correction 转到 V10 的 global-state alignment。

## 3. V10 当前核心思路

V10 不再让模型每一帧都强行修 Human3R 输出，而是把任务拆成两层：

```text
Human3R 负责每个连续 shot 内的 local reconstruction
V10 负责在 shot boundary 处把新 local segment 接回历史 global state
```

当前流程：

```text
单目 RGB 视频逐帧输入
  ↓
shot detector 判断当前帧是否是新镜头段起点
  ↓
如果不是新段：沿用 Human3R local state
  ↓
如果是新段：reset/fork Human3R local state
  ↓
Human3R 输出当前 local camera / SMPL-X / point cloud
  ↓
V10 streaming integrator 根据历史 global state + 当前 local 输出预测 segment-to-global transform
  ↓
这个 transform 缓存给当前 segment 后续帧使用
  ↓
输出 global camera + human + point cloud
```

当前先不重点验证 detector。训练和验证 alignment 时使用 oracle boundary，也就是直接告诉模型哪里发生跳变。

## 4. 当前最重要模块

当前最有潜力的模块是：

```text
history_direct_residual_integrator
```

它不是直接一次性预测最终完整 SE(3)，而是两步：

```text
history_current_integrator:
  看历史 global state + 当前 local boundary frame
  预测 coarse segment-to-global SE3

history_direct_residual_integrator:
  在 coarse 对齐后的结果上
  再预测一个小 residual SE3
```

组合方式：

```text
R_final = R_residual @ R_direct
t_final = R_residual @ t_direct + t_residual
```

这个设计的意义：

- current-only MLP 基本学不好，说明只看当前帧不够；
- history-current integrator 明显更好，说明历史 global state 是必要信息；
- direct + residual 比单纯 direct 更好，说明“先粗接，再小修”比暴力预测完整变换更稳定；
- 它仍然是严格流式的，boundary 当前帧只看历史和当前，不看未来。

## 5. 当前数据和文件

BEDLAM 第 21 序列已经检查过对应关系。

原始数据：

```text
/data/wangzheng/iJCV-CODE/data/BEDLAM/21
```

过滤后的 manifest：

```text
config/manifests/bedlam_seq000021_good_6fps/metadata.json
```

有效帧：

```text
0000, 0005, 0010, ..., 0140
```

共 29 帧，每帧 4 个人。`0145` 被排除，因为 NPZ 中是 3 个人，但 mask 中有 4 个 body instance，其中一个是边缘截断的人。

已确认：

- images 是 `seq_000021_xxxx.png`；
- camera CSV 和有效帧一一对应；
- NPZ 的 `imgname` 对应 `seq_000021/seq_000021_xxxx.png`；
- 有姿态标注的有效帧都有 body mask；
- 有效帧中 NPZ 人数和 body mask 实例数一致；
- NPZ 人物顺序和 mask instance 顺序可通过 bbox center 对应。

## 6. 当前关键脚本

训练和 probe：

```text
scripts/v10_bedlam_motion_integrator_probe.py
```

把训练好的 integrator 应用到 Human3R saved output：

```text
scripts/v10_apply_integrator_to_human3r_saved_output.py
```

把 BEDLAM GT/probe 导出成 Human3R viewer 能看的 payload：

```text
scripts/v10_export_bedlam_probe_human3r_payload.py
```

轻量可视化：

```text
scripts/v10_visualize_bedlam_motion_integrator_probe.py
```

Human3R 风格可视化仍然优先使用：

```text
scripts/view_human3r_saved_output.py
```

打开 viewer 时尽量使用 CPU，避免占 GPU：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/view_human3r_saved_output.py ... --device cpu
```

## 7. 当前主要结果

最小验证文档：

```text
docs/movie3r/v10/V10_MINIMAL_BEDLAM21_VALIDATION_20260714.md
```

Human3R output-domain 的当前主线结果目录：

```text
output/v10_bedlam21_minimal_validation/human3r_domain_integrator_fresh_full
```

当前较好的指标：

```text
history_direct_residual_integrator
root 0.4731
rot 9.42
cam 0.3956
boundary 0.6459
velocity 0.0814
non-boundary 0.0380
```

旧一版 full run 类似：

```text
root 0.4919
cam 0.3998
boundary 0.6663
```

这个结果说明：

- raw perturbed 很差；
- current-only 很差；
- history-current 明显有效；
- history-direct-residual 是当前主线；
- 手工 explicit SE3 在干净 GT 域很强，但在 Human3R 输出域不稳定，不适合作为主线；
- reverse/bidirectional 分支暂时没有带来稳定收益。

## 8. 已试过但暂不作为主线的分支

### 8.1 显式人体 SE3 粗对齐

直接用 SMPL-X/root/anchor 做显式对齐，在静止人物上看起来很强，但问题是：

- 对多人顺序、Human3R 局部坐标噪声敏感；
- 可能出现上下颠倒、roll/pitch 错误；
- 对运动人物会把真实运动抹掉；
- 容易被认为只是后处理。

因此它可以作为 strong heuristic baseline 或辅助 cue，但不是当前主线贡献。

### 8.2 Bidirectional / reverse guidance

参考 Cycle-World 的双向反馈试过两个版本：

```text
human3r_domain_integrator_fresh_full_bidir
human3r_domain_integrator_fresh_full_feature_cycle_guided
```

结论：

- bidir teacher 略微改善 camera/non-boundary，但 root/boundary 变差；
- reverse SE3 predictor 没学好；
- reverse feature predictor 能学到特征预测，但 runtime transform guidance 反而让主对齐变差；
- 暂时不继续推进，除非重新设计成训练时 regularizer，而不是推理时强行修 transform。

### 8.3 V9 token 分支

V9 结果可以作为论文中“为什么需要 state-level segment alignment”的动机：

- token correction 有用，但不足以处理 shot gauge；
- 稳定帧不应该被每帧 correction；
- 分镜问题更像 state/gauge 管理，而不是普通 per-frame refinement。

## 9. 当前建议的下一步

优先方向：

```text
V10.1 = history-current direct + residual + compact scene cue
```

具体建议：

1. 继续使用 oracle boundary，先不要让 detector 干扰 integrator 判断。
2. 训练域优先使用 Human3R saved-output，而不是直接 BEDLAM GT，因为推理时输入就是 Human3R 输出。
3. 在当前 root/camera 特征基础上加入 compact point-cloud / scene cue，例如 background centroid、scale、PCA axes、camera-to-scene vector。
4. loss 继续保持 output-domain 对齐，不要只监督显式 SE3 参数。
5. 保持 strict streaming：新 segment 第一帧预测一次 transform，后续帧缓存复用或因果平滑，不能回头改历史。
6. detector 可以单独训练，最后组合；alignment 训练阶段继续用 oracle boundary。

## 10. 给新 AI 的工作提醒

不要优先做这些事：

- 不要再大规模消融 V9 correct token，除非用户明确要求；
- 不要把 viewer 开在 GPU 上；
- 不要删除 `output/v10_bedlam21_minimal_validation/*`，这些是当前 V10 关键结果；
- 不要把 large output/cache 提交进 git；
- 不要把显式 SMPL-X 对齐包装成最终主线，它目前更适合当 baseline 或辅助 cue。

优先读这些文件：

```text
docs/movie3r/v10/AGENT_BRIEFING_V10_20260715.md
docs/movie3r/v10/V10_CAUSAL_STREAMING_MODEL_DESIGN_20260713.md
docs/movie3r/v10/V10_MINIMAL_BEDLAM21_VALIDATION_20260714.md
docs/movie3r/v10/V10_BEDLAM_MOTION_INTEGRATOR_PROBE_20260713.md
```

如果要复现实验，先检查：

```text
config/manifests/bedlam_seq000021_good_6fps/metadata.json
output/v10_bedlam21_minimal_validation/original_human3r_demo_fresh
output/v10_bedlam21_minimal_validation/human3r_domain_integrator_fresh_full
```

一句话总结当前路线：

```text
Human3R 做局部重建，V10 在严格流式条件下维护跨分镜 global state，并用 history-current direct + residual integrator 把新镜头段接回同一个世界坐标系。
```
