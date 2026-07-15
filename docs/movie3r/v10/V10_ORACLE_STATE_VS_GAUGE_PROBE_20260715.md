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

## 5. 结论

最可靠的信号是 camera：

1. `A_raw_continue` 在 cut 后完全错位，camera rotation error 约 134 deg，说明原版 Human3R 直接跨分镜继续 state 不行。
2. `B_continue_oracle_output` 把 boundary 帧修正了，但后续帧仍有残余 drift，平均 camera error 约 0.128 m / 2.69 deg，而且严格 recovery 为 None。
3. `C_reset_oracle_output` 在 cut 后 10 帧内都稳定，平均 camera error 约 0.004 m / 0.05 deg，严格 recovery 为 0。

这说明问题不只是最终 gauge。只在输出层对旧 state 的结果做 boundary SE(3) 可以解决大部分跳变，但旧 recurrent state 仍会影响后续相机轨迹。`reset/fork local state + segment re-anchor` 明显更干净。

因此当前证据支持 V10 的核心定位：

```text
Movie3R 更像 cut-aware causal reset + streaming re-anchor，
而不是单纯 final-output SE(3) post-processing。
```

## 6. 注意事项

`Root Gap` 和 `Point Chamfer` 这次只是从 Human3R saved-output 反推的 proxy，不是 AvatarReX GT human/pointmap error。它们会受 camera-space SMPL、视角变化和点云采样影响，不能作为主结论。

这次主要结论应以 camera trajectory 为准。后续如果要严谨评估 root/pointmap，需要接入对应数据集的 GT human/world point 或使用更合理的跨视角点云重叠指标。
