# V10 Oracle State vs Gauge 诊断

日期：2026-07-15

## 1. 目的

这次验证的问题是：

```text
分镜后错位，到底主要是最终输出坐标 gauge 错了，
还是 Human3R recurrent state 在 cut 后被污染了？
```

这个诊断对 V10 很关键。如果只修最终输出就够，模型可以偏向 output-domain relocalization；如果 reset 后明显更好，就说明需要 cut-aware local state control。

## 2. 数据选择

这次不用纯点云输入，因为纯点云路径无法验证 Human3R 的 image encoder / decoder / recurrent state 是否被 cut 污染。

使用 RGB AvatarReX A/B cut：

```text
seqA = lbn1/22053926
seqB = lbn1/22010716
start_frame = 1192
pre_frames = 10
post_frames = 11
cut_idx = 10
```

也就是前 10 帧来自 A 相机，后 11 帧来自 B 相机。这样可以从 `t_b` 到 `t_b+10` 看 cut 后状态是否持续稳定。

## 3. 对比设置

脚本：

```text
scripts/v10_oracle_state_vs_gauge_probe.py
```

输出目录：

```text
output/v10_oracle_state_vs_gauge_probe/avatarrex_lbn1_1192_cut10
```

对比三组：

| Variant | 含义 |
|---|---|
| `A_raw_continue` | 原版 Human3R 遇到 RGB cut 后继续旧 recurrent state |
| `B_continue_oracle_output` | 继续旧 state，但对 cut 后输出施加 boundary oracle SE(3) |
| `C_reset_oracle_output` | cut 后用 fresh Human3R state 重建，再施加 boundary oracle SE(3) |

这里的 oracle SE(3) 是在 boundary 帧估计一个 segment-level transform，然后应用到 cut 后所有帧，不是每帧单独对齐。

D 组，也就是 reset 后把 corrected token/state 写回 recurrent state，目前没有完整实现。这需要改 Human3R 内部 state 写入路径，不能用普通 saved-output 后处理完成。

## 4. 结果

cut 后 11 帧平均指标：

| Variant | Cam T ↓ | Cam R ↓ | Root Gap ↓ | Point Chamfer ↓ | Recovery |
|---|---:|---:|---:|---:|---:|
| `A_raw_continue` | 3.2878 | 134.48 | 0.2448 | 0.2001 | None |
| `B_continue_oracle_output` | 0.1278 | 2.69 | 1.3176 | 2.2495 | None |
| `C_reset_oracle_output` | 0.0042 | 0.05 | 1.3498 | 2.3994 | 0 |

`Recovery` 使用严格定义：

```text
从某个 post-cut offset 开始，到窗口结束，camera_t_error < 0.05 且 camera_r_error < 1 deg
```

完整 JSON/Markdown 指标在：

```text
output/v10_oracle_state_vs_gauge_probe/avatarrex_lbn1_1192_cut10/oracle_state_vs_gauge_metrics.json
output/v10_oracle_state_vs_gauge_probe/avatarrex_lbn1_1192_cut10/oracle_state_vs_gauge_metrics.md
```

## 5. 追加验证

这次补了更细的 gauge-free 诊断，不只看均值。

### 5.1 逐帧误差和阈值成功率

阈值定义：

```text
camera_t_error < 0.05
camera_r_error < 1 deg
```

| Variant | Success Rate | Longest Success Run | off0 | off1 | off2 | off4 | off8 | off10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_raw_continue` | 0.00 | 0 | 3.301/133.81 | 3.308/135.04 | 3.318/135.47 | 3.279/134.86 | 3.289/134.06 | 3.286/133.81 |
| `B_continue_oracle_output` | 0.09 | 1 | 0.000/0.00 | 0.133/1.84 | 0.124/2.41 | 0.108/3.07 | 0.164/3.27 | 0.186/3.01 |
| `C_reset_oracle_output` | 1.00 | 11 | 0.000/0.00 | 0.004/0.00 | 0.005/0.00 | 0.004/0.06 | 0.005/0.06 | 0.004/0.10 |

`B` 在 boundary 帧可以被 oracle SE(3) 对齐到 0，但从下一帧开始又出现误差。这说明旧 state 的问题不是一个固定坐标变换能完全解决的。

### 5.2 Gauge-free RPE

RPE 只看 cut 帧到后续帧的相对运动，不受整段 global gauge 影响：

| Variant | RPE T mean ↓ | RPE R mean ↓ | RPE T max ↓ | RPE R max ↓ |
|---|---:|---:|---:|---:|
| `A_raw_continue` | 0.1278 | 2.69 | 0.1884 | 3.39 |
| `B_continue_oracle_output` | 0.1278 | 2.69 | 0.1884 | 3.38 |
| `C_reset_oracle_output` | 0.0042 | 0.04 | 0.0064 | 0.09 |

`A` 和 `B` 的 RPE 基本一致，这是预期结果，因为 `B` 只是对 `A` 的 post-cut 输出左乘一个固定 SE(3)。`C` 明显更低，直接说明 reset/fresh state 改善了 cut 后的相对轨迹，而不是只换了最终坐标系。

### 5.3 Best-shot Offline Alignment

这一步是离线诊断，不是最终方法。它检查：如果允许对整个 B shot 做最优 SE(3)/Sim(3)，旧 state 的输出能不能被整体补救回来。

最开始只用 camera center 做 Sim(3) 会退化，因为 GT B shot 基本静止，center-only fitting 会把 scale 压到接近 0。因此脚本现在使用 `camera center + right/up/forward 方向端点` 作为 pose landmarks 来拟合。

| Variant | Best SE3 Cam T ↓ | Best SE3 Cam R ↓ | Best Sim3 Scale | Best Sim3 Cam T ↓ | Best Sim3 Cam R ↓ |
|---|---:|---:|---:|---:|---:|
| `A_raw_continue` | 0.0773 | 1.13 | 0.8320 | 0.0658 | 1.13 |
| `B_continue_oracle_output` | 0.0773 | 1.13 | 0.8320 | 0.0658 | 1.13 |
| `C_reset_oracle_output` | 0.0026 | 0.01 | 0.9998 | 0.0026 | 0.01 |

即使给 `A/B` 一个整段最优对齐，仍然明显差于 `C`。这说明旧 state 造成的是 post-cut trajectory shape / relative motion 的误差，不只是 shot-level gauge。

### 5.4 Camera-frame Continue vs Fresh

这里比较的是 output gauge correction 之前的 post-cut 局部输出：连续旧 state 的 B 段 vs fresh-state B 段。

| Metric | Value |
|---|---:|
| mean_smpl_camera_root_l2 | 0.0219 |
| mean_root_centered_pose_l2 | 0.0714 |
| mean_camera_frame_point_chamfer | 0.0763 |
| mean_depth_median_ratio_continue_over_fresh | 1.0127 |
| mean_depth_mean_abs_diff | 0.0364 |
| mean_conf_mean_abs_diff | 3.3395 |
| mean_mask_area_abs_diff | 0.0000 |

局部 camera-frame geometry 有差异，但不是灾难性崩坏；depth median 约 1.3% 差异。主问题更像 camera/recurrent state 的相对轨迹被污染，而不是当前帧局部 SMPL 或 depth 完全错掉。

### 5.5 State-write Ablation 状态

建议里的 `Reset-before-cut / Reset-after-cut / Read-old-write-fresh` 很有意义，但真版本需要改 `forward_recurrent_lighter` 内部的 state 读写方式。

目前脚本已经覆盖了：

| 设置 | 当前对应 |
|---|---|
| `A_raw_continue` | 原始继续旧 state |
| `B_continue_oracle_output` | 继续旧 state，只修最终输出 |
| `C_reset_oracle_output` | cut 后 fresh state，再重锚定 |

还没严谨实现的是：

```text
Read-old/write-fresh:
cut 帧读取旧 state 作为上下文，
但新 shot 写入另一套 fresh state，
后续帧只跟随 fresh state。
```

这个不能只靠 saved-output 后处理表达，需要在模型 forward 里 fork 两套 state：旧 state 只读，新 state 负责写入。

## 6. 结论

最可靠的信号是 camera：

1. `A_raw_continue` 在 cut 后完全错位，camera rotation error 约 134 deg，说明原版 Human3R 直接跨分镜继续 state 不行。
2. `B_continue_oracle_output` 把 boundary 帧修正了，但后续帧仍有残余 drift，平均 camera error 约 0.128 m / 2.69 deg，而且严格 recovery 为 None。
3. `C_reset_oracle_output` 在 cut 后 10 帧内都稳定，平均 camera error 约 0.004 m / 0.05 deg，严格 recovery 为 0。
4. RPE 里 `A/B` 几乎相同，而 `C` 大幅降低，说明问题不只是最终 gauge，而是旧 state 影响了 post-cut 的相对轨迹。
5. best-shot offline alignment 也无法把 `A/B` 修到 `C` 的水平，进一步支持 state pollution / state transition 是关键问题。

这说明问题不只是最终 gauge。只在输出层对旧 state 的结果做 boundary SE(3) 可以解决大部分跳变，但旧 recurrent state 仍会影响后续相机轨迹。`reset/fork local state + segment re-anchor` 明显更干净。

因此当前证据支持 V10 的核心定位：

```text
Movie3R 更像 cut-aware causal reset + streaming re-anchor，
而不是单纯 final-output SE(3) post-processing。
```

## 7. 注意事项

`Root Gap` 和 `Point Chamfer` 这次只是从 Human3R saved-output 反推的 proxy，不是 AvatarReX GT human/pointmap error。它们会受 camera-space SMPL、视角变化和点云采样影响，不能作为主结论。

这次主要结论应以 camera trajectory 为准。后续如果要严谨评估 root/pointmap，需要接入对应数据集的 GT human/world point 或使用更合理的跨视角点云重叠指标。
