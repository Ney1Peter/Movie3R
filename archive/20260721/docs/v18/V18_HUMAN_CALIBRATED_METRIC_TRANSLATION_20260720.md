# V18：Human-Calibrated Metric Translation Probe

## 三维连续性审计修正

端口三维可视化暴露出原评测遗漏：下文的 `translation error` 是 **GT camera translation error**，不是最终 Human3R pointmap / SMPL-X 的跨 cut 重合误差。DA3 能把相机位置拉近 GT，但候选只施加一个 Boundary SE(3)，没有修正 post-cut Human3R 局部人体和 pointmap 本身的错误深度。因此 camera metric 改善不等于最终三维重建改善。

对全部 180 cuts 使用最终显示的 Human3R SMPL-X root 做连续性审计：

| Method | Root jump mean | Median | P90 | >1 m |
|---|---:|---:|---:|---:|
| Hard Reset | 0.522 | 0.445 | 0.939 | 7.8% |
| Fixed Explicit | **0.277** | **0.192** | **0.496** | **3.9%** |
| V18 Human Projection | 0.937 | 0.876 | 1.450 | 38.9% |
| DA3 Metric Camera | 1.521 | 1.331 | 3.493 | 61.7% |
| Boundary Oracle + raw Human3R geometry | 1.482 | 1.250 | 3.269 | 63.9% |

这里的 root jump 包含真实的一帧人体运动，但各方法使用完全相同的 frame pair，足以做公平连续性对比。结果表明：**当前最终三维输出仍是 Fixed Explicit 最好；DA3 只能作为“外部 metric depth 能恢复 GT camera depth”的诊断证据，不能作为已验证有效的最终 Boundary candidate。** Boundary Oracle 下人体仍不重合，进一步证明主要矛盾是 Human3R post-cut local geometry/depth 与真实相机坐标不一致，而不是 Oracle transform 用错。

因此下文所有 `0.518 m` 和“四源改善”结论只适用于 camera-pose GT metric；关于“最强可部署 candidate”“跨源稳定最终结果”的原结论由本节取代。可复现审计见：

```text
scripts/v18_final_geometry_continuity_audit.py
output/v18_human_metric_translation/geometry_continuity_audit/v18_final_geometry_continuity_audit.json
```

## 结论

原 camera-only 评测得到一个不训练完整 SE(3)、不使用 raw token、GT depth、GT scene mesh 或 source ID，且在四个数据源上 camera translation 均同方向改善的 Boundary candidate；三维连续性审计后，该结果不能视为最终重建改善。

完成可选 external metric-depth 诊断后，冻结 `DA3Metric-Large` 将 GT camera translation mean 从人体投影的 `0.872 m` 降到 `0.518 m`。但它没有改善 raw Human3R geometry continuity，因此以下流程仅是 camera-pose 诊断版本，不是当前最强可部署 candidate：

```text
Hard Reset
-> V16 torso-motion rotation, 20 deg bound
-> DA3Metric-Large on 5 pre-cut + 1 post-cut frames
-> metric human camera roots
-> last projected world root
-> explicit camera translation
-> one fixed shot-level SE(3)
```

主候选为：

```text
Hard Reset
-> Fixed Explicit
-> V16 torso-motion rotation, 20 deg bound
-> frozen RGB 2D body joints
-> SMPL-X physical-body projection root
-> last pre-cut projected world root
-> explicit camera translation
-> one fixed shot-level SE(3)
```

整体结果：

| Method | T mean | T median | T P90 | T P95 | View error | T-cat | Harmful T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 1.422 | 3.718 | 4.123 | 0.988 | 65.6% | 0.0% |
| V16 torso20 + scene resolve | 1.690 | 1.374 | 3.600 | 4.073 | 0.992 | 66.1% | 9.4% |
| Human projection, no calibration | **0.872** | 0.582 | 2.079 | 2.328 | **0.464** | 34.4% | 7.2% |
| Human projection + shape median | 0.873 | **0.529** | **2.072** | 2.334 | 0.471 | **33.3%** | **6.7%** |
| Human + scene view fusion | 1.232 | 0.873 | 2.519 | 2.976 | 0.495 | 45.0% | 12.2% |
| Human + scene robust fusion | 1.421 | 0.979 | 3.075 | 3.546 | 0.702 | 48.9% | 8.3% |
| GT camera depth upper | 0.608 | 0.412 | 1.401 | 1.565 | 0.240 | 25.6% | 3.3% |
| GT depth + GT motion, torso20 R | 0.418 | 0.256 | 0.967 | 1.201 | 0.122 | 8.3% | 4.4% |

Human projection candidate 在 camera-pose GT metric 上满足四源 mean、P90 和 translation catastrophic 改善；最终三维连续性不满足最低标准。

但是它还没有恢复完全 source-invariant 的绝对深度。MVHuman 的剩余误差来自 Human3R predicted body scale 与数据集 world scale 不一致；pre-cut ratio/affine self-calibration 无法创造缺失的外部尺度，因此反而退化。

DA3 metric depth 能补充真实相机深度信息并降低 camera-pose metric 的 source-scale bias，但尚未校正 Human3R local pointmap / SMPL-X geometry scale。

## External DA3 Metric Depth

本地 checkpoint：

```text
/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large
```

Human3R crop 可直接作为 DA3 的 RGB 输入。DA3Metric-Large 返回 canonical depth，metric-only API 不返回 intrinsics，也不会自动将 `is_metric` 设为 1，因此必须使用处理后的 Human3R intrinsics 显式换算：

```text
depth_meter = raw_depth * mean(fx_processed, fy_processed) / 300
```

随后按 OpenCV/Human3R 相同约定反投影：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth_meter
```

DA3 与 Human3R 的 camera coordinates 因此完全兼容；只需处理 DA3 的 resize 后分辨率与 intrinsics。

全量 180 cuts 的 camera-root depth：

| Method | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| Human3R raw root | 0.943 | 0.741 | 1.975 | 2.197 |
| DA3 at GT pelvis pixel | 0.198 | 0.119 | 0.504 | 0.580 |
| DA3 at frozen-detector pelvis | **0.198** | 0.135 | 0.504 | 0.580 |
| DA3 torso offsets | 0.200 | 0.136 | **0.491** | **0.565** |

GT pixel 与 detector pixel 几乎相同，说明当前 2D detector 不是瓶颈。简单 pelvis depth 与 torso-offset 也基本相同，因此最终优先使用更简单的 pelvis 版本。

最终 translation：

| Candidate | T mean | T median | T P90 | View | T-cat | Harmful T |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 1.422 | 3.718 | 0.988 | 65.6% | 0.0% |
| Human projection | 0.872 | 0.582 | 2.079 | 0.464 | 34.4% | 7.2% |
| DA3 pelvis metric depth | **0.518** | 0.365 | **1.180** | **0.255** | 15.0% | **6.7%** |
| DA3 torso-offset depth | 0.519 | **0.360** | 1.211 | 0.256 | **14.4%** | 7.2% |
| GT depth + GT motion, torso20 R | 0.418 | 0.256 | 0.967 | 0.122 | 8.3% | 4.4% |

按数据源：

| Source | DA3 depth error | Fixed T | Human projection T | DA3 T |
|---|---:|---:|---:|---:|
| AvatarReX | 0.088 | 1.252 | 0.212 | **0.197** |
| THuman | 0.106 | 0.483 | 0.341 | **0.286** |
| MVHuman100 | 0.264 | 3.362 | 1.812 | **0.749** |
| MVHuman200 | 0.380 | 1.780 | 1.207 | **0.946** |

DA3 在四源全部改善，没有出现 learned scale head 的 THuman regression。它对 MVHuman 的改善最大，证明 V17/V18 发现的 source-dependent scale 确实可以由外部 metric-depth cue 缓解。

DA3 还将基于最后一帧的人体 world-root motion error 从人体投影方案的 `0.444 m` 降至 `0.220 m`，因为 pre-cut 和 post-cut 都使用了同一外部米制尺度。当前剩余 `0.518 m` 来自 depth、motion 和约 `18.49 deg` rotation 的共同作用；即使 GT depth 与 GT motion，在 torso20 rotation 下仍有 `0.418 m`。

单个 cut 对 5 个历史帧和 1 个 post-cut 帧的 DA3 inference latency：mean `0.185 s`，median `0.173 s`，P90 `0.220 s`。它只在 cut 时运行，不改变普通帧 FPS。

## 坐标审计与 Partial Oracle

使用一致的 V18 流式 cache 重新审计：

- GT human equation closure 最大误差低于浮点微小量；
- GT world root + GT camera root + GT rotation 可恢复 Boundary Oracle；
- camera-to-world、左乘、pelvis root 和米制单位关系正确。

一致 gauge Partial Oracle：

| Rotation | World root | Camera root | T mean | View error |
|---|---|---|---:|---:|
| GT | predicted CV | predicted | 1.505 | 1.258 |
| GT | GT current | predicted | 0.949 | 0.943 |
| GT | predicted CV | GT full | 0.922 | 0.450 |
| GT | GT current | GT depth only | **0.066** | **0.000** |
| GT | GT current | GT transverse only | 0.943 | 0.943 |
| GT | GT current | GT full | 0.000 | 0.000 |

Human3R raw camera root：

- position error `0.949 m`；
- depth error `0.943 m`；
- transverse error `0.066 m`。

因此当前约 1 米残差几乎就是 camera-frame human root depth error。只修 depth 即降至 `0.066 m`，说明人体深度是正确的物理切入点。

但 human world-motion 也是独立瓶颈：GT camera root + predicted CV 仍有 `0.922 m`。对最终 Human candidate，camera-depth error 与 translation error 的 Pearson/Spearman 为 `0.829/0.856`，motion error 为 `0.836/0.847`。

## 人体投影上限

| Body / Pose / 2D | Root depth error |
|---|---:|
| GT body + GT 2D | 0.000 m |
| GT body + frozen detector 2D, full body | 0.020 m |
| Predicted pose + GT shape + GT 2D | 0.057 m |
| Predicted pose + GT shape + detector 2D | 0.063 m |
| GT pose + predicted shape + GT 2D | 0.472 m |
| Predicted body + GT 2D | 0.462 m |
| Predicted body + detector 2D, torso | 0.437 m |
| Predicted body + detector 2D, full body | 0.469 m |
| Predicted mesh + Human3R mask bbox | 0.453 m |

结论非常明确：

- 2D detector 不是主要瓶颈；
- predicted pose 不是主要瓶颈；
- intrinsics 来自统一 crop 后的已知输入，本轮不存在独立 predicted-intrinsics head；
- predicted shape / physical body scale 是主要瓶颈；
- torso joints 比 full body 略稳，`0.437 m` 对 `0.469 m`，因此最终候选使用 torso projection。

按数据源的 predicted body + detector torso depth error：

| Source | Depth error |
|---|---:|
| AvatarReX | 0.075 m |
| THuman | 0.150 m |
| MVHuman100 | 0.990 m |
| MVHuman200 | 0.566 m |

使用 GT shape/world scale 后，predicted pose + detector 2D 在四源均约 `0.06 m`。MVHuman 的 GT `smplx_world_scale` 分别固定为：

- MVHuman100：`0.5417`；
- MVHuman200：`0.6500`；
- AvatarReX / THuman：`1.0`。

这说明 MVHuman 的评测坐标尺度并不等于 canonical adult body scale。Human3R predicted shape 没有恢复该 world-scale factor，因此“真实成人尺寸”不能无条件等同于四个数据源的 GT metric gauge。

## Per-Sequence Calibration

可部署 camera-root depth：

| Method | Depth error |
|---|---:|
| Human3R raw root | 0.943 m |
| Projection, no calibration | **0.437 m** |
| Projection + pre-cut shape median | **0.435 m** |
| Pre-cut raw-depth ratio calibration | 0.945 m |
| Pre-cut affine depth calibration | 0.948 m |
| Shape median + ratio calibration | 0.930 m |

ratio/affine calibration 的目标是 pre-cut Human3R raw root depth。该 raw depth 本身就是 source-dependent bias，因此标定把投影结果重新拉回错误尺度：最终 translation 回到约 `1.61 m`，harmful correction 超过 `33%`。

结论：

- shape median 可做轻量稳定化，但收益很小；
- root-depth ratio 和 affine self-calibration 不成立；
- 同一人物历史只能稳定已有尺度，不能从无量纲投影中创造数据集 world-scale；
- 下一步若继续，应寻找独立 metric cue 校准 body scale，而不是对齐 Human3R raw root。

## 人体运动

使用 projection + shape median 后：

| Motion model | World-root mean | Final T mean |
|---|---:|---:|
| Last root | **0.444 m** | **0.873 m** |
| Constant velocity, last 2 | 0.455 m | 0.880 m |
| Constant acceleration | 0.477 m | 0.900 m |
| Robust velocity, 3 frames | 0.454 m | 0.878 m |
| Robust velocity, 5 frames | 0.455 m | 0.877 m |
| Torso-compatible damping | 0.451 m | 0.875 m |

当前 180 cuts 以低速运动为主，last root 最稳。复杂外推没有收益，constant acceleration 最差。当前协议没有足够急停、跳跃和高速样本，不能对这些运动得出强结论。

## 跨数据源结果

| Source | Fixed mean / P90 / T-cat | Human mean / P90 / T-cat | Harmful T |
|---|---:|---:|---:|
| AvatarReX | 1.252 / 1.569 / 77.1% | **0.212 / 0.370 / 0.0%** | 0.0% |
| THuman | 0.483 / 0.787 / 2.1% | **0.341 / 0.633 / 0.0%** | 18.8% |
| MVHuman100 | 3.362 / 4.591 / 95.8% | **1.812 / 2.582 / 85.4%** | 0.0% |
| MVHuman200 | 1.780 / 2.667 / 94.4% | **1.207 / 1.760 / 58.3%** | 11.1% |

四源 mean、P90、T-cat 全部同方向改善。THuman 有 `18.8%` 单样本 harmful correction，但整体 mean、P90 和 catastrophic 均改善，不是 V17 的系统性破坏。

按人体朝向，front/back/side 均改善；按初始 scale error，高误差组从 `2.467 m` 降到 `1.153 m`。179/180 样本不是明显截断，当前不能评价严重 truncation。协议全部为单人，不能得出多人 consensus 结论。

## Human 与 Scene

当前 Human3R 场景 candidate 与 Human candidate 没有形成有效互补：

- Human 在 `90.6%` 样本上优于 V16 scene translation；
- Human viewing error `0.471 m`，scene 为 `0.992 m`；
- Human transverse error `0.653 m`，scene 为 `1.173 m`；
- 两者误差 Spearman `0.731`，失败高度相关；
- Oracle best of Human/Scene 仅从 `0.873 m` 降到 `0.854 m`；
- 固定 view fusion 反而退化到 `1.232 m`。

因此当前不要加入 Human/Scene fusion。保留 Human full translation candidate，scene 仅做 residual/check；只有 scene transverse solver 明显改善后才重新研究融合。

## V16 Rotation Bound

| Bound | R mean | R P90 | Harmful | False on Fixed<10 | Gain on Fixed>30 |
|---:|---:|---:|---:|---:|---:|
| 10 deg | 20.32 | 53.99 | 16.1% | 12.2% | 6.57 deg |
| 20 deg | 18.49 | 48.49 | 17.2% | 12.8% | 12.20 deg |
| 25 deg | 17.82 | 45.10 | 17.2% | 12.8% | 14.65 deg |
| 45 deg | 16.04 | 39.33 | 18.3% | 12.8% | 20.90 deg |

V18 使用 `20 deg` 作为保守统一 bound。需要明确：bound clipping 没有解决约 12% 的简单样本 false correction，因为这些错误 residual 本身小于 10 度；`45 deg` 的纯 rotation 数值最好。若要进一步降低 harmful correction，需要几何一致性检查，而不是继续缩小 bound。

## 效率

冻结 Keypoint R-CNN 在单次 cut 上处理 5 个历史帧 + 1 个 post-cut 帧：

- mean `0.141 s`；
- median `0.138 s`；
- P90 `0.152 s`；
- peak allocated memory `1.96 GB`。

包含全部 V18 诊断分支的 projection/PnP/SMPL 评测约 `0.112 s/cut`；实际单候选部署分支更小。当前 training-free probe 的总额外 cut latency 上界约 `0.25 s`。所有计算只在 cut 时触发一次，普通帧 Human3R FPS 不变；最终 camera、pointmap 和 SMPL-X 统一使用同一个固定 SE(3)。

## 最终问题回答

1. **约 1 米残差是否主要等于 Human3R camera root depth error？** 是。raw depth error `0.943 m`，GT depth-only 将理想 translation 降到 `0.066 m`。
2. **SMPL-X 尺寸和 2D 投影能否恢复 source-invariant metric depth？** 物理关系成立，GT shape + detector 2D 可达约 `0.063 m`；当前 predicted shape 不能完全 source-invariant，特别是 MVHuman world scale 未恢复。
3. **主要误差来自哪里？** 首要是 shape/body world scale，其次是 human world-motion；2D joints 和 pose 误差较小，intrinsics 不是本轮瓶颈。
4. **Per-sequence calibration 是否避免 V17 scale regression？** shape median 略有稳定作用；raw-depth ratio/affine 不成立并显著退化。当前有效的是逐样本投影，不是对 raw Human3R depth 的自标定。
5. **人体 cue 是否主要改善 viewing direction？** 是，view error 从 `0.988` 降到 `0.464 m`；同时因 pre/post 使用一致人体投影，transverse 也从 `1.173` 降到 `0.653 m`。
6. **Human 与 scene 是否互补？** 当前不互补。Human 在 view 和 transverse 都更好，固定融合退化。
7. **V16 使用哪个 bound？** V18 固定 `20 deg` 保守 bound；纯 rotation 最优是 `45 deg`，且 clipping 无法消除小 residual false correction。
8. **是否获得首个跨源稳定 metric candidate？** 否。Human projection/no-calibration 和 DA3 只在 GT camera-pose metric 上改善；应用到未校正的 Human3R pointmap / SMPL-X 后，最终三维连续性明显弱于 Fixed Explicit。

External diagnostic 更新：DA3Metric-Large 将 GT camera translation mean 降到 `0.518 m`，证明外部 metric depth 本身有效；它不是当前最强的最终 Human3R Boundary candidate。

## 下一步

当前保留的可部署主线仍是：

```text
Hard Reset
-> Fixed Explicit
```

下一步不应直接蒸馏 DA3 camera translation。应先验证 DA3 是否能一致校正 post-cut Human3R 的显式局部几何，再重新求 Boundary SE(3)：

- 使用 DA3 对 post-cut pointmap depth 和 SMPL-X camera-frame root 做同一尺度/深度校正；
- 校正后再运行 Fixed Explicit 或显式刚体对齐，并同时评测 camera error、root jump 和 pointcloud discontinuity；
- 只有最终 geometry continuity 跨源改善后，才考虑将 DA3 蒸馏为 cut-only metric-depth bridge；
- 评估遮挡、截断和多人 consensus；
- 将大型 2D detector 替换为已有轻量 2D pose 或蒸馏分支。

不要继续使用 Human3R raw root depth 作为 calibration target，也不要恢复 learned scale head、token branch 或 learned selector。

## 产物

- `scripts/v18_human_translation_partial_oracle.py`
- `scripts/v18_cache_human_stream.py`
- `scripts/v18_cache_2d_keypoints.py`
- `scripts/v18_projection_depth_probe.py`
- `scripts/v18_rotation_bound_cleanup.py`
- `scripts/v18_human_metric_translation_eval.py`
- `scripts/v18_da3_metric_depth_probe.py`
- `output/v18_human_metric_translation/partial_oracle/`
- `output/v18_human_metric_translation/stream_cache/`
- `output/v18_human_metric_translation/keypoint_cache/`
- `output/v18_human_metric_translation/projection_depth/`
- `output/v18_human_metric_translation/rotation_bound/`
- `output/v18_human_metric_translation/v16_bound20_scene/`
- `output/v18_human_metric_translation/final_candidates/`
- `output/v18_human_metric_translation/da3_metric_depth/`
