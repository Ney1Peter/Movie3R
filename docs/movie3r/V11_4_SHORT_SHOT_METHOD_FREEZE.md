# V11.4 Short-Shot Method Record

> 状态：保留为当前效果最好的 short-shot / short-horizon alignment 方法。
>
> 重要限制：该方法用于一次或少量 camera cuts 后的 shot re-anchoring，不适用于无限长度视频，也未证明长期 multi-shot world mapping 稳定。

## 方法定义

冻结推理流程为：

```text
normal frame
  -> original frozen Human3R recurrent path

camera cut
  -> cut trigger
  -> reset Human3R state before decoding the first post-cut frame
  -> Fixed Explicit coarse Boundary
  -> V16 torso-motion rotation with a fixed 20 deg bound
  -> Conditional VGGT only for difficult rotation tails
  -> V11.4 one uniform shot scale
  -> solve one translation from the predicted pre-cut human anchor
     and current post-cut human root
  -> apply one fixed shot-level Boundary to the complete new shot
  -> optional V14.2 continuity after alignment
  -> Align-Then-Commit
```

## 每个模型何时使用

| 模型/模块 | 使用时机 | 职责 |
|---|---|---|
| Human3R | 每一帧 | 生成 shot-local camera、pointmap 和 SMPL-X；cut 后第一帧 decode 前 hard reset |
| Fixed Explicit | 每个 cut 一次 | 生成可解释的 coarse Boundary 初始化 |
| V16 torso motion | 每个 cut 一次 | 用人体 torso temporal motion 修正 rotation，统一限制在 20 deg 内 |
| VGGT-1B | 只在 frozen trigger 命中的 cut | 处理困难 rotation tail；不逐帧运行，不负责 scale 或 translation |
| Keypoint R-CNN | cut-time scale/depth cue | 提供人体 2D keypoints，不直接预测 SE(3) |
| DA3Metric-Large | 每个 cut 的 pre/post Boundary 图像 | 提供人体/背景 metric cue，估计统一 shot scale |
| V11.4 uniform similarity | 每个新 shot 一次 | 用同一个 scale 缩放 camera translation、pointmap、human root、body offsets、joints 和 vertices |
| V14.2 continuity | alignment 完成后，可选 | 只稳定 shape、body scale 和 local pose，不参与 Boundary 求解 |

## 统一几何规则

新 shot 只允许一组：

```text
scale      s
rotation   R
translation t
```

它们共同作用于：

- camera translation；
- pointmap；
- SMPL-X root；
- root-centered body offsets；
- joints；
- vertices。

禁止：

- scene 和 human 使用不同 deployable scale；
- 只缩放 camera，不缩放 body；
- camera 使用 calibrated root、human 使用 raw root；
- 每帧重估 scale 或 Boundary；
- 在最终结果后增加独立 foot translation；
- 用 GT camera、GT depth 或 source ID 生成 candidate。

## 使用范围

推荐使用范围：

- 单次 camera cut；
- 1-2 次相邻 cuts 的 short shot；
- 需要快速恢复 camera/human world placement 的短时视频；
- 每个 shot 内 Boundary 固定复用。

不应使用或不应宣称已解决：

- 无限长度视频；
- 任意次数 camera cuts；
- 长期无漂移 world mapping；
- camera、human、scene 三者同时最优；
- 8-cut 以上稳定闭环；
- scene pointmap 的 view-dependent/non-rigid depth error。

## 目前效果

### 180-cut 统一协议

| 指标 | Fixed | V11.4 + Conditional VGGT |
|---|---:|---:|
| camera translation | `0.712 m` | `0.403 m` |
| camera rotation | `24.20 deg` | `12.09 deg` |
| human root | `0.234 m` | `0.163 m` |
| human joints | `0.290 m` | `0.216 m` |
| human vertices | `0.285 m` | `0.210 m` |
| scene discontinuity | `0.483 m` | `0.532 m` |
| camera success | `41.1%` | `70.0%` |

### Untouched 60-cut holdout

| 指标 | Fixed | V11.4 + Conditional VGGT |
|---|---:|---:|
| camera translation | `0.663 m` | `0.450 m` |
| camera rotation | `23.05 deg` | `14.08 deg` |
| human root | `0.234 m` | `0.195 m` |
| human joints | `0.291 m` | `0.241 m` |
| human vertices | `0.285 m` | `0.236 m` |
| scene discontinuity | `0.475 m` | `0.546 m` |
| camera success | `41.7%` | `66.7%` |

结论是稳定的：

- camera 和 human 在独立 holdout 上复现改善；
- scene 存在约 5-7 cm 的显著 trade-off；
- V11.4 scale 是 camera-oriented，不是 scene-optimal scale。

## 为什么限定为 short shot

真实 recurrent rollout 使用前一次 predicted world 作为下一次 anchor，不在每个 cut 重新回到 GT gauge。四源平均结果为：

| cuts | camera drift | rotation drift | human root drift |
|---:|---:|---:|---:|
| 1 | `0.229 m` | `7.81 deg` | `0.093 m` |
| 2 | `0.326 m` | `23.97 deg` | `0.094 m` |
| 4 | `0.698 m` | `37.99 deg` | `0.134 m` |
| 8 | `0.946 m` | `59.03 deg` | `0.193 m` |

1-2 cuts 时仍属于可用的短期 re-anchoring 范围；4-8 cuts 后 rotation 和 camera error 明显累积。当前方法没有 loop closure、BA、global trajectory optimization 或长期 gauge correction，因此没有机制保证误差不会随 cut 数增加。

准确表述为：

> V11.4 performs causal, fixed-boundary re-anchoring for short shots and sparse camera cuts. It is not an unlimited-horizon mapping system.

## 可视化入口

真实 recurrent 1/2/4/8-cut 三维对比使用：

```bash
TMPDIR=output/v14_5_final_audit/tmp \
.venv/bin/python scripts/v14_5_multicut_interactive_viewer.py \
  --device cuda:5 \
  --port 8107
```

可视化固定包含四类序列：

| Source | 观察重点 |
|---|---|
| AvatarReX | 1-2 cuts 的短期改善，以及 8-cut 的中等累计漂移 |
| THuman | camera 改善，但多 cut 后 human root 可能变差 |
| MVHuman100 | 明确的 8-cut rotation/camera failure |
| MVHuman200 | camera 改善与 scene trade-off 同时出现 |

界面中的 `1 cut`、`2 cuts` 是方法主要适用区间；`4 cuts`、`8 cuts` 是累计误差压力测试，不是方法适用于无限长度的证据。

三维视角默认使用 `Current human`，以当前人体为中心并保留固定第三方观察方向；`Current shot` 用于查看当前 shot 内相机、人体和局部轨迹，`Full rollout` 用于查看完整累计漂移。切换序列、cut 前缀或主方法后会自动重新居中，也可以点击 `Center 3D view` 手动恢复视角。
