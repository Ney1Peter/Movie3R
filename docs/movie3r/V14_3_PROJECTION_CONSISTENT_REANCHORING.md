# V14.3 Projection-Consistent Human-Camera Re-anchoring

## Executive Conclusion

V14.3 证明了题目中的核心因果关系：V18/DA3 camera-only 版本之所以经常出现
“相机对了、人反而不对”，主要不是相机公式错误，而是 camera translation 使用了
校准后 root，最终 SMPL-X 仍使用 raw Human3R root。让 camera 和完整人体共同使用
同一个 camera-frame root 后，camera 指标完全保留，人体 root、joints 和 vertices
同时明显改善；数值闭环误差最大仅 `2.73e-7 m`。

但本轮也发现第二个独立问题：DA3 能恢复更好的 metric camera/human gauge，却不能
自动修复 Human3R raw pointmap 的尺度。DA3 coupled 的 camera/human 指标最好，但
RGB 重投影、脚底-场景距离和跨 shot scene continuity 明显弱于 V18 coupled。因此：

```text
accepted core:
  calibrated root -> camera solve + complete SMPL-X translation

current geometry-safe candidate:
  V16 rotation + V18 Human Projection Coupled + V14.2 Continuity

strong metric diagnostic, not yet a complete scene solution:
  DA3 Coupled (one cut-time inference)
```

DA3 仍值得保留，因为它把 MVHuman 的 camera/human metric error 大幅降低；下一步必须
解决 pointmap metric scale 或明确的 scene similarity，而不能只继续移动 camera/human。

## Protocol

主实验使用与 V18 相同的 180 个真实 cross-camera cuts：AvatarReX 48、THuman 48、
MVHuman100 48、MVHuman200 36。所有版本固定：

- frozen Human3R，cut 后解码前 hard reset；
- GT cut index，仅 cut 时触发新逻辑；
- V16 torso-motion rotation 和统一 `20 deg` bound；
- frozen 2D joints、相同 intrinsics 和相同 pre-cut motion anchor；
- 每个 post shot 只有一个固定 Boundary transform；
- camera、pointmap、SMPL-X、joints、vertices 使用同一 world gauge；
- V14.2 memory 只稳定人体输出，不进入相机尺度求解；
- DA3Metric-Large 使用缓存的冻结推理，原始输入为 cut 前 5 帧加 cut 后第一帧；
- 所有 SMPL-X body/vertex 计算在 GPU 上执行。

入口：

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/v14_3_projection_consistent_reanchoring_probe.py \
  --device cuda:5

PYTHONPATH=src:. .venv/bin/python \
  scripts/v14_3_human_continuity_visualization.py \
  --device cuda:5
```

机器可读结果和可视化：

```text
output/v14_3_projection_consistent_reanchoring/quantitative/
output/v14_3_projection_consistent_reanchoring/visualization/index.html
```

## Coupled Correction

| Method | Camera T | T P90 | Human root | Joints | Vertices | Joint success | Torso reproj |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 3.718 | 0.926 | 0.946 | 0.945 | 14.4% | 19.2 px |
| V18 camera-only | 0.872 | 2.079 | 0.676 | 0.700 | 0.691 | 23.3% | 19.2 px |
| V18 coupled full-root | 0.872 | 2.079 | **0.444** | **0.472** | **0.468** | **47.2%** | **6.6 px** |
| DA3 camera-only | 0.518 | 1.180 | 1.005 | 1.028 | 1.019 | 22.8% | 19.2 px |
| DA3 coupled full-root | **0.518** | **1.180** | **0.220** | **0.295** | **0.287** | **58.9%** | 19.6 px |
| Human + DA3 fixed fusion | 0.617 | 1.398 | 0.235 | 0.299 | 0.292 | 53.3% | 10.0 px |
| GT depth + GT motion coupled | 0.418 | 0.947 | 0.000 | 0.176 | 0.163 | 69.4% | 19.7 px |
| Boundary Oracle + raw human | 0.000 | 0.000 | 0.949 | 0.966 | 0.958 | 25.6% | 19.2 px |

V18 coupled 相对 V18 camera-only：camera 完全不变，human root `0.676 -> 0.444 m`；
68.3% 样本的人体改善超过 5 cm。DA3 coupled 相对 DA3 camera-only：human root
`1.005 -> 0.220 m`；88.3% 样本改善超过 5 cm，仅 2.8% 明显变差。

Boundary Oracle 配 raw human 的 root 仍有 `0.949 m`，进一步证明只把 camera 拉到 GT
并不能修好最终人体。Coupled 不是指标包装，而是缺失的几何一致性步骤。

完整 projected mesh 与真实人体检测框也支持同一结论：

| Method | Mesh bbox IoU | Width ratio | Height ratio |
|---|---:|---:|---:|
| Fixed Explicit | 0.620 | 0.830 | 0.843 |
| V18 coupled | **0.872** | **1.005** | **1.018** |
| DA3 alpha=0.75 | 0.729 | 1.142 | 1.153 |
| DA3 full | 0.631 | 1.455 | 1.445 |
| Human + DA3 fusion | 0.756 | 1.141 | 1.152 |

V18 coupled 不仅改善 joints，也把完整 mesh 尺寸拉回检测框。DA3 full 会系统性放大
人体，MVHuman100 height ratio 达 `2.30`；`alpha=0.75` 仍为 `1.47`。这与 RGB 视频中
MVHuman 人体超出画面的失败完全一致，是 DA3 当前不能作为主方法的直接视觉证据。

## Depth-Only vs Full-Root

V18 full-root 与 depth-only 的 camera/human 指标近似相同，但 full-root 将 torso
重投影从 `15.7 -> 6.6 px`，180 个样本中 99.4% 更好。因此 V18 应使用 full-root。

DA3 full-root 与 depth-only 的 camera/human 指标也近似相同；full-root 平均重投影
`22.5 -> 19.6 px`，但只有 66.7% 样本改善，depth-only 的 joint success 略高
(`61.1%` vs `58.9%`)。因此 DA3 的横向 root 仍不如 V18 稳定；若只看鲁棒阈值，
depth-only 更保守，若看平均投影，full-root 略好。

## DA3 Metric Cue

DA3 coupled 的分数据源 camera/human root mean：

| Source | Fixed camera | DA3 camera | DA3 human root | V18 human root |
|---|---:|---:|---:|---:|
| AvatarReX | 1.252 | **0.197** | **0.099** | 0.108 |
| THuman | 0.483 | **0.286** | **0.131** | 0.164 |
| MVHuman100 | 3.362 | **0.749** | **0.322** | 0.984 |
| MVHuman200 | 1.780 | **0.946** | **0.365** | 0.547 |

DA3 对 MVHuman 的 metric mismatch 特别有效，并未系统性破坏 THuman。四个 source 的
camera 和 human mean 都优于 Fixed；DA3 相对 V18 的主要价值确实是独立 metric depth。

但 DA3 full-root 相对 V18 full-root 的 torso 重投影在 180/180 样本上更差，scene
discontinuity 也从 `0.998 -> 1.382 m`。`alpha=0.75` 可将 DA3 重投影改善到
`13.5 px`、scene 降到 `1.272 m`，代价是 camera `0.518 -> 0.580 m`。该 alpha 是
本轮统一诊断值，尚未在独立 dev set 固定，因此不能直接宣称最终超参数。

固定 Human+DA3 融合将重投影降到 `10.0 px`、scene 降到 `1.171 m`，但 paired camera
改善/退化率为 28.9%/42.8%，没有稳定优于 DA3，暂不作为主方法。

## Projection-Preserving Body Scale Diagnostic

额外测试了按 `new_depth / raw_depth` 同比缩放完整人体，以判断 DA3 是否只缺 body
projection scaling：

| Method | Joints | Torso reproj | Foot-scene |
|---|---:|---:|---:|
| DA3 rigid body | 0.295 | 19.6 px | 0.499 m |
| DA3 projective body scale | 0.288 | 17.6 px | 0.572 m |
| DA3 alpha=0.75 rigid body | **0.295** | **13.5 px** | **0.375 m** |
| DA3 alpha=0.75 projective body | 0.275 | 17.5 px | 0.411 m |

完整同比缩放只在 MVHuman100 明显有效，却显著破坏 AvatarReX；总体也让 foot-scene
更差。它再次表现为 source-dependent scale patch，不满足跨源统一要求，故不保留。

## Scene Safety Boundary

| Method | Scene discontinuity | Foot-scene |
|---|---:|---:|
| Fixed Explicit | **0.587 m** | **0.268 m** |
| V18 coupled | 0.998 m | 0.273 m |
| DA3 coupled | 1.382 m | 0.499 m |
| DA3 alpha=0.75 | 1.272 m | 0.375 m |

这些值不是 coupled 公式出错，而是 raw Human3R pointmap 没有获得 DA3 metric scale。
刚体 Boundary 只能改变 gauge，不能修正 shot-local depth scale。33 个样本缺少足够有限
背景点、26 个样本无法计算可靠 foot-scene；聚合仅使用有限样本并报告有效计数。

因此 V14.3 已经获得 camera-human 一致性，但尚未获得完整 camera-human-scene metric
一致性。论文中必须明确这条安全边界。

## Continuity and Multi-Cut

V14.2 保守 memory 在相同 180 cuts 上继续成立：

| Metric | Hard Reset | Continuity memory | Change |
|---|---:|---:|---:|
| Shape jump | 0.718 | **0.558** | -22.3% |
| Body-scale jump | 0.00751 | **0.00577** | -23.2% |
| Local-pose residual | 5.37 deg | **4.58 deg** | -14.8% |

加入 coupled alignment 后，continuity 不改变 camera、root anchor 或 scene：V18 root
保持 `0.444 m`，DA3 root 保持 `0.220 m`；DA3 joints/vertices 还小幅从
`0.295/0.287 -> 0.293/0.285 m`。这说明 alignment 与 continuity 可以安全联合。

8-cut recurrent evidence 仍为 shape drift `0.582 -> 0.484`，Immediate Commit 的
memory root error 为 `1.135 m`，Align-Then-Commit 为 `0.931 m`。world root 和 global
orientation 依赖 Boundary gauge，Align-Then-Commit 仍是长期 memory 的必要协议。

正常无 cut 路径不触发 V14.3：沿用 V14.1 实际 no-cut 检查，camera、pointmap 和
SMPL-X shape 最大差异均为 `0`；普通帧不运行 DA3，也没有新增推理开销。

20 个代表案例覆盖四个 source，并同时包含成功、无变化、轻微退化和失败案例。每个
案例提供 16 或 20 帧的 RGB mesh、固定第三方世界视角、连续性 timeline 和 root/depth
difference 视频。Continuity 在这些序列中的平均 2D joint/mesh 改变量仅为
`1.15/1.53 px`，最大 mesh 改变量约 `4.11 px`。时间线能稳定看到 shape/local-pose
曲线更平滑，但 RGB 上多数差异很轻微；因此它属于题目中的情况 E，只能作为轻量数值
正则，不能包装成主要视觉贡献。world trajectory 的明显变化来自 coupled alignment，
而不是 continuity memory。

## Final Answers

1. **V14.2 continuity 能否肉眼观察？** 多数案例不能清楚观察，只能在 timeline 和少数
   high-jump 案例看到轻微差别；平均 mesh 位移仅 `1.53 px`，不是主要视觉贡献。
2. **主要改善什么？** 主要是 shape 和 root-centered local pose，其次是 body scale；
   world trajectory 主要由 coupled alignment 改善。
3. **“camera 对、人不对”是否源于不同 root？** 是主要原因。Boundary Oracle + raw
   human 仍有 `0.949 m` root error，coupled closure 为 `2.73e-7 m`。
4. **同一 calibrated root 能否同时改善 camera 和 human？** 能。V18/DA3 camera 指标
   完全保留，human root 分别降到 `0.444/0.220 m`。
5. **Depth-only 是否更稳定？** V18 不是，full-root 投影明显更好；DA3 depth-only 的
   success 略高但平均投影更差，二者没有绝对胜者。
6. **Coupled 是否破坏 2D mesh projection？** V18 full-root 不破坏：mesh bbox IoU
   `0.620 -> 0.872`；DA3 会放大人体，尤其 MVHuman100，说明独立 scene depth 与
   当前 body projection 存在冲突。
7. **DA3 是否比人体投影稳定？** 对 GT metric camera/human，尤其 MVHuman，是；对 RGB
   reprojection 和 raw Human3R scene consistency，不是。
8. **DA3 是否解决 MVHuman world-scale mismatch？** 显著缓解，但未解决 pointmap
   world scale，因此只解决了 camera-human 部分。
9. **Continuity 与 coupled 能否安全联合？** 能。Memory 不改变 camera/scene/root
   anchor，且 joints/vertices 不退化。
10. **最终采用什么？** Coupled correction 原理应成为正式模块；当前完整几何主方法
    采用 **V18 Human Projection Coupled + Continuity**。DA3 Coupled 作为更强 metric
    candidate/诊断保留，待 scene metric scale 解决后再升级为最终主方法。固定融合和
    projective body scaling 暂不保留。

## Decision

本轮结果是题目情况 A 与 B 的组合，而不是单一情况：

- A 成立：Human Projection Coupled 同时改善 camera 和 human；
- B 对 metric camera-human 成立：DA3 明显更强；
- 但 DA3 尚未获得 scene-safe 的完整三方一致性。

最终模块定义为：

```text
Shot-local Human3R Hard Reset
+ V16 Torso-Motion Rotation (20 deg)
+ Projection-Consistent Human-Camera Coupled Alignment
+ V14.2 Isolated Human Continuity Memory
+ Align-Then-Commit
```

DA3 的准确定位是：只在 cut 时运行一次的外部 metric-depth cue，而不是逐帧 depth
分支；它已经证明有明确价值，但必须与 scene metric gauge 问题分开报告和继续解决。
