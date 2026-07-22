# Active Boundary Alignment

当前主线使用七个主版本，共享工具不再编号。

## V10.1 Fixed Explicit

显式基础候选：人体姿态给出粗对齐，人体区域外的 Human3R pointmap 做
小范围 refinement，最终输出一个 shot-level SE(3)。

入口：`scripts/v10_1_fixed_explicit_candidate_probe.py`

## V11.x Retained Methods

- `V11.1`：保留方法对比。包含 Fixed Explicit、Torso Only、Conditional
  Wide Rotation 等统一刚体候选。
- `V11.2`：Contact-Preserving Alignment。视觉接触较好，但会修改局部人体
  关系，因此保留为诊断版本。
- `V11.3`：组件必要性消融，不作为部署方法。
- `V11.4`：Uniform Similarity。相机平移、pointmap、SMPL-X root 和完整人体
  尺寸使用同一个 shot scale，是当前保留的人体大小修正版。

对应入口：

- `scripts/v11_1_boundary_method_comparison_viewer.py`
- `scripts/v11_2_contact_preserving_probe.py`
- `scripts/v11_3_component_ablation.py`
- `scripts/v11_4_uniform_similarity_probe.py`

## V12.x Long-Sequence Viewer

- `V12.1`：构建 10 帧 cut 前 + 10 帧 cut 后缓存。
- `V12.2`：三维长序列对比 viewer。

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v12_2_long_sequence_viewer.py \
  --device cuda:0 \
  --port 8096
```

Viewer 中保留的方法名称：

- `Fixed Explicit`
- `Torso Only`
- `Conditional Wide Rotation`
- `Contact-Preserving Alignment`
- `Uniform Similarity - Torso`
- `Uniform Similarity - Conditional Wide`

## V13.1 Real-Video Fixed Alignment

真实视频验证：cut 后 hard reset，只使用 cut 前两帧和 cut 后第一帧估计一次
Fixed Explicit SE(3)，随后统一变换整个新镜头的 camera、pointmap 和 SMPL-X。

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v13_1_real_video_fixed_alignment_viewer.py \
  --pre_dir output/aist_ms_000000_human3r_original \
  --post_dir output/v55_real_video_explicit_alignment/aist_post_reset \
  --cut_idx 341 \
  --output_dir output/v55_real_video_explicit_alignment/aist_fixed_explicit \
  --port 8099
```

该版本不使用 VGGT、DA3、camera GT、GT depth 或完整未来 shot。

## V14.1 Shot-Aware State Routing

V14.1 不修改 Boundary SE(3)，而是修改 camera cut 处的信息流：

- scene/camera 在第一张 cut 后帧解码前 hard reset，只读写 fresh state；
- human 分支可以读取隔离的跨 shot 人体记忆；
- raw human token 不写入 scene/camera state；
- world root 必须先经过统一 Boundary SE(3)，再提交到长期人体轨迹。

当前推荐的 training-free 配置为：

- raw token mixing：`0`；
- shape memory：`0.25`；
- root-centered local pose memory：`0.15`；
- world trajectory：Align-Then-Commit；
- 不使用固定 world-root jump verify 阈值。

单 cut、四数据源、多 cut 和因果控制结果见
`docs/movie3r/V14_1_SHOT_AWARE_STATE_ROUTING.md`。

入口：

- `scripts/v14_1_shot_aware_state_routing_probe.py`
- `scripts/v14_1_multicut_state_routing_rollout.py`
- `scripts/v14_1_shape_memory_sweep.py`

## V14.2 Canonical Human Memory

V14.2 联合测试 V14.1 canonical shape/scale 是否也能改善 V18 human-projection
Boundary translation。180 个四源 cuts 的结果是：

- continuity 成立：shape jump `0.718 -> 0.558`，scale jump
  `0.00751 -> 0.00577`，local-pose residual `5.37 -> 4.58 deg`；
- alignment 不成立：V18 current body `0.872 m`，canonical alpha=0.25
  `0.871 m`，完整 canonical `0.874 m`；
- GT scale-only 可达到 `0.462 m`，GT beta-only 为 `0.895 m`，说明缺失的是历史
  Human3R memory 无法提供的绝对 world-scale scalar；
- camera-pose metric 改善仍不等于 final Human3R geometry continuity 改善。

因此 V14.2 保留为 `Shot-aware Human Continuity Memory`，不作为 Boundary
alignment module。完整报告见
`docs/movie3r/V14_2_CANONICAL_HUMAN_MEMORY.md`。

入口：

- `scripts/v14_2_canonical_human_memory_probe.py`
- `scripts/v14_2_multicut_memory_replay.py`

## V14.3 Projection-Consistent Re-anchoring

V14.3 修复了 V18/DA3 camera-only 的核心不一致：同一个 calibrated camera-frame
human root 同时用于 camera translation 和完整 SMPL-X placement。

- V18 camera 保持 `0.872 m`，human root `0.676 -> 0.444 m`；
- DA3 camera 保持 `0.518 m`，human root `1.005 -> 0.220 m`；
- coupling 数值闭环最大误差 `2.73e-7 m`；
- V14.2 continuity 可安全联合，不改变 camera/scene/root anchor；
- continuity 平均 mesh 视觉改变量仅 `1.53 px`，保留为轻量数值正则，不作为主要
  视觉贡献；
- DA3 显著改善 MVHuman metric mismatch，但 raw Human3R scene discontinuity
  增至 `1.382 m`，尚未形成完整 camera-human-scene metric solution。

当前 geometry-safe 主候选是 `V18 Human Projection Coupled + V14.2 Continuity`；
DA3 Coupled 保留为 cut-time metric candidate，等待 pointmap metric scale 问题解决。

完整报告和可视化见：

- `docs/movie3r/V14_3_PROJECTION_CONSISTENT_REANCHORING.md`
- `output/v14_3_projection_consistent_reanchoring/visualization/index.html`

入口：

- `scripts/v14_3_projection_consistent_reanchoring_probe.py`
- `scripts/v14_3_human_continuity_visualization.py`

## Cached Outputs

已有输出目录仍保留旧编号，以避免复制数 GB 缓存。这些目录名只是历史缓存
标识，不再代表当前代码版本。无用输出位于 `output/archive/20260721/`。
