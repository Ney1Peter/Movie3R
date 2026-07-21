# Active Boundary Alignment

当前主线只使用四个主版本，共享工具不再编号。

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

## Cached Outputs

已有输出目录仍保留旧编号，以避免复制数 GB 缓存。这些目录名只是历史缓存
标识，不再代表当前代码版本。无用输出位于 `output/archive/20260721/`。
