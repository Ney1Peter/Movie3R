# V14.7 Shot-Aware Uniform Similarity Re-anchoring

中文名称：**面向 Camera Cut 的短镜头统一相似变换重锚定**

## 1. 版本定位

V14.7 是当前完整方法的独立版本号。它不是一次新的参数搜索，也没有在 V14.6 结果之后
更改算法；它把此前分散在 Fixed Explicit、V16、V11.4 和 V14.6 审计中的最终保留路径
统一命名，作为论文、代码和可视化中的唯一当前版本。

版本职责如下：

| 版本 | 职责 |
|---|---|
| V10.1 Fixed Explicit | coarse anchor 和 fallback 来源 |
| V16 torso-motion | bounded rotation correction 来源 |
| V11.4 Uniform Similarity | fused shared-scale block 来源 |
| V14.2 | 默认关闭的 continuity 附件 |
| V14.6 | 组件必要性和无 VGGT 公平审计 |
| **V14.7** | **当前完整方法与唯一默认版本** |

因此，对外不再把当前方法称为“V14.6 审计后的 V11.4”，而统一称为 **V14.7**。

## 2. 默认流程

```text
streaming RGB + intrinsics + cut trigger
-> frozen Human3R recurrent inference
-> pre-decode hard reset at the camera cut
-> fresh post-cut shot-local reconstruction
-> Fixed Explicit coarse anchor
-> V16 torso-motion rotation, fixed 20 deg bound
-> V11.4 fused DA3/Keypoint shared shot scale
-> one explicit translation solve from the pre-cut human world anchor
-> one fixed shot-level Boundary
-> transform camera + pointmap + complete SMPL-X in one gauge
-> Align-Then-Commit
```

Cut 后只求一次 shot-level similarity：

```text
X_world = R * (s * X_local) + t
```

同一个 `s/R/t` 作用于：

- camera relative translation；
- scene pointmap；
- SMPL-X root；
- root-centered body offsets；
- joints；
- vertices。

整个 short shot 固定复用该 Boundary，不逐帧重估。

## 3. 默认开启与关闭

默认开启：

- pre-decode Hard Reset；
- Fixed Explicit；
- V16 torso-motion rotation；
- V11.4 fused shared scale；
- explicit translation；
- one fixed Boundary；
- Align-Then-Commit world protocol。

默认关闭：

- Conditional VGGT；
- V14.2 continuity memory；
- V14.3 coupled root；
- V14.4 Unified Human/DA3 diagnostic roots；
- learned SE(3)；
- BA 或完整未来 shot 优化。

DA3 和 Keypoint R-CNN 是 V11.4 尺度估计器内部 cue，不应被描述为两个独立有效的
alignment module。

## 4. 冻结结果

180-cut、四数据源、VGGT off：

| 方法 | Camera T | Rotation | Human root | Joints | Scene |
|---|---:|---:|---:|---:|---:|
| Fixed Explicit | 0.712 m | 24.20 deg | 0.234 m | 0.290 m | 0.483 m |
| Fixed + V16, raw scale | 0.518 m | 16.04 deg | 0.163 m | 0.223 m | 0.526 m |
| **V14.7** | **0.463 m** | **16.04 deg** | **0.163 m** | 0.225 m | 0.536 m |

V16 到 V14.7 的 camera gain 为 `0.518 -> 0.463 m`，paired `p=0.00107`。Scene 同时
从 `0.526 -> 0.536 m`，paired `p=0.038`，因此必须报告轻微 scene trade-off。

60-cut capture-disjoint holdout、VGGT off：

- camera translation：`0.663 -> 0.508 m`；
- rotation：`23.05 -> 17.62 deg`；
- scene：`0.475 -> 0.547 m`。

## 5. 方法边界

V14.7 的准确定位是：

> short-horizon camera-human re-anchoring after sparse camera cuts

它适合一个或两个稀疏 cut 后的 short shot，不是无限长度 world mapping。真实 recurrent
8-cut 审计中 camera drift 为 `0.946 m`，rotation drift 为 `59.03 deg`，因此不能宣称
长期多 shot 稳定。

它优先改善 camera-human placement，没有完成 camera-human-scene 三者同时最优。当前
实验还使用 GT cut index 作为触发信号；自动 cut detector 不属于 V14.7 已验证模块。

## 6. 实现入口

V14.7 不复制历史实现，避免出现两份逻辑逐渐不一致。当前入口为：

```text
scripts/v14_4_unified_similarity_reanchoring_probe.py
scripts/v14_5_true_recurrent_multicut_audit.py
scripts/v14_5_multicut_interactive_viewer.py
```

完整结构说明：

```text
docs/movie3r/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md
```

组件证据：

```text
docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md
```

V14.7 的 Git 冻结 tag 为：

```text
v14.7-shot-aware-similarity
```

V14.6 tag 保留为命名前的审计快照，不删除。
