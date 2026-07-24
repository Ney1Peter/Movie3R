# Movie3R-Single V14.7

## 定位

V14.7 是当前效果优先的单人 camera-cut 对齐主版。它冻结 Human3R，不训练新的
SE(3) 网络，而是在 cut 时对 fresh shot-local reconstruction 求一次显式 similarity：

```text
RGB + intrinsics + cut trigger
-> pre-decode Human3R hard reset
-> Fixed Explicit coarse anchor
-> V16 torso-motion rotation, 20 deg bound
-> V11.4 DA3/Keypoint fused shared shot scale
-> explicit translation from pre-cut human world anchor
-> one fixed shot-level Boundary
-> camera + pointmap + complete SMPL-X use the same s/R/t
-> Align-Then-Commit
```

## 冻结身份

| 项目 | 值 |
|---|---|
| 正式名称 | Movie3R-Single V14.7 |
| 功能 commit | `af92478` |
| Git tag | `movie3r-single-v14.7` |
| 原审计 tag | `v14.7-shot-aware-similarity` |
| Human3R checkpoint | `src/human3r_896L.pth` |
| Human3R SHA-256 | `1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377` |
| 默认 VGGT | 关闭 |
| 默认 continuity | 关闭，可在 alignment 后选用 |

`af92478` 在原 V14.7 冻结基础上加入了任意多 cut 的实际 demo 和 viewer，不改变
V14.6/V14.7 的算法或 180-cut 数值。

## 两种运行档位

### Lite

```text
Hard Reset + Fixed Explicit + V16 + raw Human3R scale
```

不使用 DA3、Keypoint R-CNN 或 VGGT，适合低依赖调试。

### Full

```text
Lite + DA3Metric-Large + 2D keypoints + V11.4 fused shared scale
```

Full 是冻结结果中的默认单人方法。DA3 和 Keypoint R-CNN 是 V11.4 内部 scale cue，
不是两个独立 Boundary 模块。Conditional VGGT 默认关闭。

## 主入口

任意真实图像序列：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v14_7_custom_multicut_demo.py \
  --seq_path /path/to/images \
  --cuts 5 10 15 \
  --output_dir output/v14_7_custom/example \
  --device cuda:0
```

可视化：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v14_7_custom_multicut_viewer.py \
  --result_dir output/v14_7_custom/example \
  --port 8080
```

统一评测和真实 recurrent 审计入口：

- `scripts/v14_4_unified_similarity_reanchoring_probe.py`
- `scripts/v14_5_true_recurrent_multicut_audit.py`
- `scripts/v14_5_multicut_interactive_viewer.py`

## 冻结证据

180-cut、四源、VGGT off：camera translation `0.518 -> 0.463 m`，但 scene
`0.526 -> 0.536 m`，存在轻微 scene trade-off。60-cut capture-disjoint holdout 中 camera
translation `0.663 -> 0.508 m`，scene `0.475 -> 0.547 m`。

## 限制

- 这是 single-human anchor 主线；当前核心实验和 loader 使用 `max_humans=1`。
- 适合 short shot 和稀疏 camera cut，不是无限长度 world mapping。
- 真实 recurrent 8-cut 审计仍有 `0.946 m / 59.03 deg` 累计漂移。
- 方法优先改善 camera-human placement，没有解决原版 Human3R 自身的脚地、悬空或
  human-scene local reconstruction 错误。
- 当前使用已知 cut index 作为触发信号；自动 cut detector 不是本版结论。

## 详细文档

- `LATEST_MODEL.md`
- `docs/movie3r/V14_7_SHOT_AWARE_UNIFORM_SIMILARITY_REANCHORING.md`
- `docs/movie3r/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md`
- `docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md`
- `docs/movie3r/V14_5_FINAL_GEOMETRY_STREAMING_AUDIT.md`
