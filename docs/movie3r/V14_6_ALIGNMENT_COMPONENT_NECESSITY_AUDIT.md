# V14.6 Alignment Component Necessity Audit

> 目的：在 VGGT 默认关闭的前提下，判断 Fixed Explicit、V16 torso、DA3、
> V11.4 uniform similarity、V14.2 continuity 和 Keypoint R-CNN 是否各自有独立作用。

## 1. 结论先行

不能说所有模块都已经被独立证明有效。当前最准确的模块状态是：

| 模块 | 当前状态 | 结论 |
|---|---|---|
| Pre-decode Hard Reset | 核心 | 已由 V14.1 明确证明；不 reset 会发生旧 state 污染 |
| Fixed Explicit | 保留的 coarse anchor/fallback | 是所有当前显式分支的共同起点，但尚未与另一种 coarse initializer 做 clean replacement ablation |
| V16 torso-motion rotation | **核心、独立有效** | camera、rotation 和 human 均显著改善；scene 有明确 trade-off |
| DA3 | **作为 V11.4 联合尺度 cue 保留** | DA3 单独的 background 或 root 分支没有显著 camera 增益；与 Keypoint gate 联合后才形成稳定 V11.4 增益 |
| Keypoint R-CNN | **联合 cue，模型必要性未证明** | 纯 keypoint projection 不显著；它在当前方法中主要帮助 DA3 读取人体 metric root 并约束 background scale |
| V11.4 Uniform Similarity | **effect-first 主 scale block** | camera 显著改善、projection 保持；human root 不变，joints/scene 略退化 |
| V14.2 continuity | 默认关闭、可选 | 只改善 shape/scale/local-pose continuity，不改善 Boundary alignment |
| Conditional VGGT | 默认关闭 | 有 rotation-tail 收益，但不是核心；必须显式 `--enable_vggt` 才运行 |

因此当前默认方法应被理解为三个主要几何步骤，而不是很多平级模型串联：

```text
pre-decode hard reset
-> Fixed Explicit coarse rotation/fallback
-> V16 bounded torso rotation
-> V11.4 one shared shot scale
   (DA3 + Keypoint R-CNN are internal scale cues)
-> one explicit translation solve
-> one fixed Boundary for the short shot
```

V14.2 和 VGGT 均不在默认路径中。

## 2. 新增公平消融

本轮在 GPU 6 上使用统一 V14.4 evaluator 重新评测全部 180 cuts。所有尺度分支共享：

- 同一个 pre-shot gauge；
- 同一个 V16 rotation，VGGT off；
- 同一个 raw Human3R root placement；
- 同一个 translation equation；
- 同一个 scene 有效子集，`147/180`；
- 同一个 pointmap、human 和 projection evaluator。

只改变 post-shot scale cue：

1. `V16 raw scale`：不进行 post-shot scale correction；
2. `DA3 background only`：只用 Human3R background depth 与 DA3 depth 的 median ratio，
   不使用 2D keypoint human root；
3. `DA3 + Keypoint root`：Keypoint R-CNN 定位 torso/pelvis，在 DA3 metric depth 上读取
   human root scale，不使用 background gate；
4. `Keypoint projection only`：使用 2D keypoints 与当前 SMPL-X body 的物理投影比例，
   不使用 DA3；
5. `V11.4 fused scale`：DA3+Keypoint human root 为主，只有 background/root ratio
   小于 `0.95` 时采用相对 root 限制在 `+-15%` 的 background scale。

运行命令：

```bash
TMPDIR=output/v14_5_final_audit/tmp \
.venv/bin/python scripts/v14_4_unified_similarity_reanchoring_probe.py \
  --device cuda:6 \
  --output_dir output/v14_6_alignment_component_necessity/full180_no_vggt
```

完整 JSON：

```text
output/v14_6_alignment_component_necessity/full180_no_vggt/
  v14_4_unified_similarity_reanchoring.json
```

## 3. 统一结果

| 方法 | Camera T mean/P90/P95 | Rotation | Human root | Joints | Scene | Camera success |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 0.712/1.509/1.703 | 24.20 | 0.234 | 0.290 | **0.483** | 41.1% |
| V16 raw scale | 0.518/0.934/1.314 | **16.04** | **0.163** | **0.223** | 0.526 | 55.0% |
| DA3 background only | 0.480/0.903/1.110 | 16.04 | 0.163 | 0.225 | 0.545 | 56.1% |
| DA3 + Keypoint root | 0.492/1.027/1.148 | 16.04 | 0.163 | 0.225 | 0.542 | 58.3% |
| Keypoint projection only | 0.504/0.965/1.088 | 16.04 | 0.163 | 0.225 | 0.539 | 57.2% |
| **V11.4 fused scale** | **0.463/0.918/1.088** | 16.04 | 0.163 | 0.225 | 0.536 | **60.6%** |

所有 scale 方法的 torso reprojection 都保持 `19.2 px`，说明 uniform similarity 没有
改变原 Human3R 的 2D 投影。V11.4 也没有改善 world root；root 的主要增益发生在
Fixed 到 V16/common-anchor 阶段。

## 4. 逐组件判断

### 4.1 Fixed Explicit

Fixed 是当前所有方法的 coarse rotation 初始化，在没有有效 post-cut human 时也是 fallback。
但当前实验没有比较“去掉 Fixed、使用另一种 coarse rotation 后仍运行 V16”的版本，因此：

- 可以说它是当前实现的必要起点；
- 不能说 Fixed 内部每个 pointmap refinement 都被独立证明必要；
- V16 后 translation 会重新显式求解，Fixed 的原始 translation 不是最终 translation。

### 4.2 V16 torso-motion rotation

这是证据最明确的后处理模块：

| 指标 | Fixed | V16 | Delta | Paired p |
|---|---:|---:|---:|---:|
| Camera translation | 0.712 | 0.518 | -0.194 m | `2.20e-14` |
| Camera rotation | 24.20 | 16.04 | -8.17 deg | `3.10e-15` |
| Human joints | 0.290 | 0.223 | -0.068 m | `2.90e-10` |
| Scene | 0.483 | 0.526 | +0.043 m | `7.96e-11` |

Camera/rotation 分别有 `75.6%/77.2%` 样本改善。四个数据源 rotation 均同方向改善。
因此 V16 必须保留，但论文中必须同时报告 scene trade-off，不能称为 camera-human-scene
全面改善。

V16 使用 Human3R/SMPL-X 的三维 torso history，不依赖 Keypoint R-CNN。

### 4.3 DA3 与 Keypoint R-CNN

相对 V16 raw scale：

| Cue | Camera delta | Improved/Harmed | Paired p | Scene delta | Scene p |
|---|---:|---:|---:|---:|---:|
| DA3 background only | -0.039 m | 52.2% / 38.9% | `0.0684` | +0.018 m | `1.57e-5` |
| DA3 + Keypoint root | -0.027 m | 48.9% / 43.3% | `0.1169` | +0.016 m | `0.0229` |
| Keypoint projection only | -0.014 m | 57.2% / 42.8% | `0.1285` | +0.013 m | `0.0167` |

三个单独 cue 的 camera 配对检验均未达到 `p < 0.05`，且 scene 都显著变差。因此不能
把 DA3、Keypoint root 或 Keypoint projection 分别包装成独立有效模块。

但是它们的组合不是无效堆叠：V11.4 相比 DA3+Keypoint root，camera 再下降
`0.028 m`，`p=6.71e-7`；scene 下降 `0.0067 m`，`p=0.0073`。相比 keypoint-only，
camera 下降 `0.041 m`，`p=0.00279`。这说明有用的是可解释的联合门控规则：

- DA3 提供 metric depth；
- Keypoint R-CNN 指定读取人体 metric depth 的位置；
- background cue 只在与 human root cue 关系异常时做有界修正。

当前证据只能证明“human keypoint cue 对联合尺度规则有帮助”，不能证明 Torchvision
Keypoint R-CNN 这一具体网络不可替代。更轻的 detector 或 Human3R 自身 2D projection
仍可能替代它，需要单独 latency/accuracy ablation。

### 4.4 V11.4 Uniform Similarity

相对 V16 raw scale：

- camera `0.518 -> 0.463 m`，下降 `0.055 m`，`p=0.00107`；
- P95 `1.314 -> 1.088 m`；
- camera success `55.0% -> 60.6%`；
- human root 完全不变；
- joints `0.223 -> 0.225 m`，差异不显著，`p=0.406`；
- scene `0.526 -> 0.536 m`，退化 `0.009 m`，`p=0.0380`。

四源 camera 均同方向改善：

| Source | V16 | V11.4 | Scene V16 -> V11.4 |
|---|---:|---:|---:|
| AvatarReX | 0.226 | 0.209 | 0.605 -> 0.611 |
| MVHuman100 | 0.680 | 0.658 | 0.167 -> 0.180 |
| MVHuman200 | 0.792 | 0.770 | 0.230 -> 0.269 |
| THuman | 0.443 | 0.293 | 0.788 -> 0.780 |

所以 V11.4 是有效但目标窄的 camera-oriented scale block。它的主要收益集中在 THuman，
并非四源等量改善；它也不能被描述为 scene improvement 或 human-root improvement。

### 4.5 V14.2 continuity

V14.2 的已验证作用是：

- shape jump `0.718 -> 0.558`，下降 `22.3%`；
- body-scale jump `0.00751 -> 0.00577`，下降 `23.2%`；
- local-pose residual `5.37 -> 4.58 deg`，下降 `14.8%`；
- 8-cut shape drift `0.582 -> 0.484`。

它不改变 camera、Boundary、scene 或 root anchor；在 V14.4 中 joints 仅改善约
`0.0012 m`，肉眼变化也较小。因此它不是 alignment module，应保持默认关闭，只在明确
需要跨 shot 人体外观/姿态平滑时开启，并且必须 Align-Then-Commit。

### 4.6 VGGT

VGGT 有可复现的困难 rotation-tail 收益，但不是默认核心：

- 180 cuts：rotation `16.04 -> 12.09 deg`；
- untouched 60：`17.62 -> 14.08 deg`；
- 419 post-freeze cases：`14.91 -> 13.16 deg`；
- 仍存在少量 harmful triggers，并增加模型、显存和 cut latency。

真实 recurrent 入口现已默认不加载 VGGT；只有传入 `--enable_vggt` 才运行。当前 8107
viewer 使用的是此前生成的 Conditional-VGGT 历史缓存，界面已明确标注，不代表新的默认
方法。

## 5. 最终默认与可选配置

### 默认 effect-first 配置

```text
Hard Reset
+ Fixed Explicit coarse anchor
+ V16 torso rotation, bound=20 deg
+ V11.4 fused DA3/Keypoint shared scale
+ one explicit translation solve
+ one fixed short-shot Boundary
```

### 更轻的最小配置

```text
Hard Reset
+ Fixed Explicit
+ V16 torso rotation
+ raw scale
```

它不需要 DA3 和 Keypoint R-CNN，camera 为 `0.518 m` 而不是 `0.463 m`，但 scene 略好
`0.526 m` 而不是 `0.536 m`。当模型数量、cut latency 或方法简洁性比约 `5.5 cm` camera
均值收益更重要时，这个版本是合理选择。

### 默认关闭

- VGGT：只有明确需要 rotation-tail rescue 时显式开启；
- V14.2 continuity：只有明确需要人体连续性时开启；
- V14.3 coupled root、Unified DA3 root：保留为诊断，不进入主路径。

## 6. 最终回答

1. **不是每个模块都被独立证明有效。**
2. **V16 是最明确、必须保留的独立对齐增益。**
3. **V11.4 对 camera 有显著增益，但 scene 轻微显著退化，human root 不变。**
4. **DA3 和 Keypoint R-CNN 单独都未达到显著 camera 增益；当前应把它们视为 V11.4
   联合尺度估计器的内部 cue，而不是两个独立贡献。**
5. **V14.2 只对 continuity 有效，默认关闭。**
6. **VGGT 默认关闭；它是可选 tail rescue，不属于默认方法。**
7. **最终方法实际上只有两个主要 alignment correction block：V16 rotation 和 V11.4
   shared scale。Fixed 提供 coarse anchor，最后显式求一次 translation。**
