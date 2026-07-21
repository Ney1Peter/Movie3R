# V17：Direct SE(3) vs. Factorized Translation Bridge

## 结论

本轮结论属于预设的**情况 D：Direct 和 Factorized 都不能作为最终 learned Boundary alignment**。

两条学习路线都能降低部分平均误差，但均未满足跨数据源安全条件：

- Direct Full-SE(3) 在训练域拟合极强，在 held-out source 明显退化；
- Factorized 在 MVHuman 上能明显降低平移，但会严重破坏 THuman；
- 最佳平均平移模型仍将 catastrophic rate 从 Fixed 的 `67.2%` 提高到 `77.2%`；
- 所有学习模型都存在较高 harmful correction，无法替代当前显式方案。

因此当前最终保留：

```text
Hard Reset
-> Fixed Explicit
-> V16 Torso-Motion bounded rotation residual
-> Scene translation re-solving
-> one fixed shot-level SE(3)
```

不继续扩大 learned Boundary alignment，也不重新引入 raw token。

## 实验实现

使用 180 个真实 cross-camera cuts，执行四折 Leave-One-Source-Out：

- AvatarReX held out；
- THuman held out；
- MVHuman100 held out；
- MVHuman200 held out。

每折训练源内部再按完整 capture 留出 unseen camera-pair validation。测试源不参与 normalization、early stopping、residual bound 或模型选择。

所有方法共享 416 维显式输入：

- Fixed / V16 / V15 相对变换；
- Human3R 预测 pointmap 的尺度、协方差、置信度和空间分布；
- camera intrinsics；
- SMPL-X root、shape、torso/motion；
- 显式 fitting diagnostics。

没有输入 raw image token、state token、human token、source ID、camera ID 或 camera-pair ID。模型参数约 `23k`，只在 cut 时执行一次，推理约 `0.01-0.05 ms/cut`，不影响普通帧 FPS。

平移和完整 SE(3) 均定义在 cut 前最后一个 Human3R camera frame 中，避免世界原点造成的 gauge shortcut。随机 gauge 审计最大误差为 `1.4e-6`。

## 阶段零：Direction 与 Scale

整体 Partial Oracle：

| Translation variant | Vector error |
|---|---:|
| Current direction + current scale | 1.715 m |
| GT direction + current scale | 1.412 m |
| Current direction + GT scale | 0.618 m |
| VGGT direction + current scale | 2.150 m |
| VGGT direction + GT scale | 0.958 m |

GT direction 只恢复 `0.303 m`，GT scale 恢复 `1.097 m`。误差主要集中在旧相机 viewing direction：当前为 `1.409 m`，替换 GT scale 后降至 `0.282 m`。

但瓶颈明显依赖数据源：

| Source | Fixed scale / GT scale | Fixed direction error | VGGT direction error | 主问题 |
|---|---:|---:|---:|---|
| AvatarReX | 4.21 / 3.00 m | 3.87° | 35.84° | scale |
| THuman | 5.19 / 5.17 m | 4.18° | 7.59° | 当前已较准 |
| MVHuman100 | 4.70 / 2.00 m | 30.00° | 12.95° | scale 为主，direction 次之 |
| MVHuman200 | 3.10 / 2.23 m | 30.08° | 14.42° | direction + scale |

这解释了后续 learned scale bridge 的失败：训练于 AvatarReX/MVHuman 的模型倾向于缩短 translation，held-out THuman 的原始 scale 本来接近正确，因此会被系统性改坏。

## Held-Out Source 主结果

| Method | T mean | T P90 | R mean | R P90 | Catastrophic | Harmful |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 3.718 | 24.20° | 62.30° | 67.2% | 0.0% |
| V16 Torso Motion | 1.679 | 3.600 | 16.04° | 39.33° | 65.6% | 20.0% |
| Direct Absolute SE(3) | 1.590 | 3.118 | 33.34° | 59.72° | 65.6% | 63.9% |
| Direct Residual SE(3) | 1.586 | 2.678 | 21.21° | 44.69° | 81.7% | 56.7% |
| Factorized Scale-only | 1.562 | 2.377 | 16.04° | 39.33° | 84.4% | 56.7% |
| Factorized Direction-Scale | 1.527 | 2.406 | 16.04° | 39.33° | 85.0% | 45.0% |
| Factorized bounded Δt | 1.517 | 2.536 | 16.04° | 39.33° | 81.1% | 47.2% |
| Direction-Scale + uncertainty | **1.298** | **1.990** | 16.04° | 39.33° | 77.2% | 39.4% |
| VGGT direction + learned scale | 1.783 | 3.753 | 16.04° | 39.33° | 77.8% | 53.9% |

最佳平均模型虽然将 translation mean 降至 `1.298 m`，但 catastrophic 比 Fixed 高 `10.0` 个百分点，不能部署。

## 跨源失效

关键 held-out source 结果：

| Method | AvatarReX T / Cat | THuman T / Cat | MVHuman100 T / Cat | MVHuman200 T / Cat |
|---|---:|---:|---:|---:|
| Fixed | 1.252 / 77.1% | 0.483 / 2.1% | 3.362 / 100% | 1.780 / 97.2% |
| Direct Absolute | 1.488 / 62.5% | **2.991 / 100%** | 0.955 / 64.6% | 0.707 / 25.0% |
| Direct Residual | 1.132 / 64.6% | **1.494 / 87.5%** | 2.265 / 95.8% | 1.408 / 77.8% |
| Direction-Scale + uncertainty | 1.075 / 66.7% | **1.542 / 93.8%** | 1.619 / 87.5% | 0.839 / 55.6% |

Direct 和 Factorized 都主要改善 MVHuman；它们把本来最准确的 THuman 变成灾难性失败。平均数下降来自“修复困难域、破坏简单域”的交换，不是稳定泛化。

## 拟合与泛化断层

| Method | Seen train | Unseen camera-pair val | Held-out source |
|---|---:|---:|---:|
| Direct Absolute | 0.137 m / 2.36° | 1.029 m / 15.18° | 1.590 m / 33.34° |
| Direct Residual | 0.340 m / 2.26° | 0.710 m / 13.83° | 1.586 m / 21.21° |
| Factorized Scale-only | 0.631 m / 16.61° | 0.617 m / 14.58° | 1.562 m / 16.04° |
| Factorized Direction-Scale | 0.280 m / 16.61° | 0.543 m / 14.58° | 1.527 m / 16.04° |
| Factorized bounded Δt | 0.322 m / 16.61° | 0.704 m / 14.58° | 1.517 m / 16.04° |

Direct Absolute 的训练拟合接近 Oracle，但完全 held-out source 的 rotation 增至 `33.34°`。它不是单纯记住 camera ID，因为 ID 从未输入、弱 camera-stat baseline 也只有 `2.006 m / 64.53°`；更准确的解释是模型学到了训练源中的 camera-layout、人体尺度和 pointmap 分布相关 shortcut。

Factorized 的训练拟合更受物理约束，但 translation 仍有显著 source gap，说明 1DoF/3DoF 降维不足以解决 source-dependent metric scale。

## Gauge 与 Uncertainty

raw world-coordinate residual 模型：

- 不做 gauge augmentation：`1.459 m / 21.26°`，cat `78.3%`；
- 随机 gauge augmentation：`1.950 m / 30.98°`，cat `90.0%`；
- 解析 camera-frame invariant 主模型：`1.586 m / 21.21°`，cat `81.7%`。

随机 augmentation 没有改善泛化。在当前小数据规模下，直接输入随机 gauge world block 增加了学习难度；应优先使用解析 gauge-invariant 表示，而不是要求网络自己学习 gauge invariance。raw no-augmentation 数值略强，但更可能利用固定世界布局，不应作为最终方法。

Uncertainty 也不能形成可靠安全机制：

- Direct uncertainty：rotation Spearman `0.661`，translation Spearman `0.285`；
- Factorized uncertainty：translation Spearman `-0.495`。

后者虽然平均误差最好，但对真正 translation error 的排序方向错误，不能用于 fallback。

## 最终问题回答

1. **直接学习完整 SE(3) 是否可行？** 训练拟合可行，严格跨源部署不可行。
2. **是否学习了跨镜头几何？** 只学到部分训练域规律，同时明显依赖 source/camera-layout-correlated statistics；held-out source gap 很大。
3. **是否应该因子化？** 作为分析和约束应该因子化；它比自由回归更易解释，但当前仍不足以获得安全泛化。
4. **V16 torso-motion 是否保留？** 保留。它不训练 translation，四源 rotation 同方向改善，仍是最可靠的新增组件。
5. **Translation 主要缺什么？** 整体以 metric scale 为主，但 MVHuman 同时缺 direction；瓶颈高度 source-dependent。
6. **B1/B2/B3 哪个最好？** 无可部署胜者。确定性版本中 bounded Δt 平均最好；含 uncertainty 的 Direction-Scale 数值最好，但 tail 和 calibration 不合格。
7. **VGGT direction 是否有效？** 只对 MVHuman 有帮助，明显损害 AvatarReX/THuman，不能作为无条件 teacher 或输入。
8. **worst-source 与 tail 谁最稳定？** Fixed + V16 torso 最稳定；所有 learned translation 都引入新的严重 source regression。
9. **最终路线？** 选择“不再训练 learned alignment”。保留 Hard Reset + Fixed Explicit + V16 torso-motion rotation + scene translation re-solving。

## 产物

- `scripts/v17_translation_partial_oracle.py`
- `scripts/v17_build_explicit_feature_cache.py`
- `scripts/v17_train_loso_fold.py`
- `scripts/v17_eval_seen_splits.py`
- `scripts/v17_merge_loso_results.py`
- `output/v17_direct_vs_factorized/partial_oracle/`
- `output/v17_direct_vs_factorized/feature_cache/`
- `output/v17_direct_vs_factorized/loso/`
- `output/v17_direct_vs_factorized/evaluation/`
