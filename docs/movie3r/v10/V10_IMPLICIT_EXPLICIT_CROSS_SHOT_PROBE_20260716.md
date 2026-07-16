# V10 隐式粗对齐 + 显式细对齐 Training-Free 实验

日期：2026-07-16

## 1. 实验问题

本实验回答：冻结的 Human3R/CUT3R token 能否在 camera cut 后，为跨 shot SE(3) 提供比纯显式人体/pointmap 配准更好的初值、匹配或可靠性判断。

这不是最终模型。实验只使用 GT cut_idx，不训练网络，不修改 Human3R 默认推理，也不使用 cut 后 continue-old-state 的错误输出做 bridge。

每个 case 使用：

```text
3 帧 cut 前干净历史
+ 6 帧 cut 后 fresh-state shot
= 9 帧
```

bridge 只读取 cut 后第一帧。得到一个 SE(3) 后，固定应用于后续 6 帧。

## 2. 数据

定量使用 6 个有 raw camera calibration 的 AvatarReX case：

| Case | 视角变化 |
|---|---:|
| zzr 625 | 49.4° |
| lbn2 1151 | 90.0° |
| lbn1 1007 | 95.1° |
| zzr 1079 | 112.7° |
| zzr 329 | 146.4° |
| lbn1 695 | 161.7° |

RICH 新上传目录只有多相机 RGB，没有相机标定，所以没有混入 camera error 和成功率。它可以作为后续无 GT 定性测试或第二阶段训练数据。

当前 6 个 AvatarReX case 的 Laplacian texture score 约为 410–513，文档中的高/中/低纹理是这 6 个 case 内部的相对三等分，不代表覆盖了完整真实世界纹理范围。

## 3. 方法

独立脚本：

```text
scripts/v10_implicit_explicit_cross_shot_probe.py
```

### 3.1 Human3R 运行方式

Human3R 主体严格冻结。实验脚本在运行时临时包装 `_encode_image` 和 `_recurrent_rollout`，保存：

```text
CUT3R encoder scene tokens
Human3R final decoder scene tokens
pose token
human token summary
state summary
pose memory summary
```

包装只在实验上下文中生效，退出后恢复原函数。没有改动 `model.py`、`inference.py` 或 `demo.py`。

### 3.2 Explicit-only

初值由 cut 前历史人体和 cut 后 fresh 人体的世界 root/body frame 得到：

```text
R_init = R_history_body @ R_fresh_body^T
t_init = history_root - R_init @ fresh_root
```

随后执行固定预算的局部背景 pointmap refinement：

```text
8 次迭代
每个历史帧最多 6000 个背景点
排除 human mask 和低置信度点
trimmed nearest-neighbor correspondences
加权 Kabsch residual SE(3)
```

这是局部边界 refinement，不是整段 BA 或全局轨迹优化。

### 3.3 Implicit-only

分别使用 encoder scene token 和 decoder scene token：

1. 在 fresh 边界帧与每个历史帧之间计算 cosine similarity。
2. 保留 mutual nearest token matches。
3. token 只负责确定像素对应，不把 token 当作三维坐标。
4. 从对应像素读取两侧 Human3R pointmap 3D。
5. 使用固定 512 次 RANSAC + weighted Kabsch 得到 coarse SE(3)。
6. 在所有历史帧和两类 token 候选中，按无 GT confidence 选择最高者。

### 3.4 Hybrid

Hybrid 与 Explicit-only 使用完全相同的 pointmap、人体输出、历史窗口、点数、阈值和 8 次 refinement。

唯一额外信息是 token coarse candidate。

首次运行发现 token 会高置信地匹配到不同镜头中的相同 patch 坐标。为避免破坏显式结果，增加 training-free position-collapse guard：

```text
如果最高置信 token candidate 中，
超过 50% matches 的 patch 位移不超过 1.5 patch，
则认为发生位置坍缩，不允许 token candidate 进入 Hybrid。
```

该判断不使用 GT。token candidate 仍会在 Implicit-only 和诊断结果中保留。

## 4. 对比结果

cut 后 6 帧平均结果：

| Variant | Cam T ↓ | Cam R ↓ | Boundary T ↓ | Boundary R ↓ | Root jump ↓ | Root orient jump ↓ | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original continue | 2.6433 | 103.42° | 2.6601 | 109.44° | 0.2407 | 104.53° | 0% |
| Reset raw | 2.8083 | 109.50° | 2.8114 | 109.48° | 0.2613 | 104.69° | 0% |
| Explicit-only | 1.0237 | 11.27° | 1.0270 | 11.26° | 0.1237 | 7.02° | 0% |
| Implicit-only | 2.8162 | 108.76° | 2.8194 | 108.73° | 0.3461 | 103.96° | 0% |
| Safe Hybrid | 1.0237 | 11.27° | 1.0270 | 11.26° | 0.1237 | 7.02° | 0% |
| Oracle SE(3) | 0.0075 | 0.08° | 0.0000 | 0.00° | 1.0488 | 8.84° | 100% |

成功定义为：

```text
post-cut mean camera translation error < 0.10 m
且 rotation error < 5°
```

Explicit-only 明显修正了方向，但平移仍约 1 m，因此没有达到严格成功标准。

Oracle camera 对齐后人体 root jump 仍较大，是因为不同视角下 Human3R 的 SMPL-X camera-frame root 预测本身不完全一致。Oracle 只修 camera/world gauge，并没有使用 GT 人体强行贴合。

## 5. Token 单独诊断

选中的 token candidate：

| Angle | Token confidence | Implicit coarse T error | Implicit coarse R error | 物理正确 match 比例 |
|---:|---:|---:|---:|---:|
| 49.4° | 0.842 | 1.254 | 47.1° | 2.78% |
| 90.0° | 0.991 | 2.674 | 90.0° | 1.22% |
| 95.1° | 0.232 | 2.358 | 94.0° | 0% |
| 112.7° | 0.944 | 3.068 | 112.6° | 0% |
| 146.4° | 0.984 | 3.392 | 145.9° | 0% |
| 161.7° | 0.944 | 3.747 | 162.8° | 0% |

物理正确 match 的诊断只在候选 SE(3) 已经估计完成后，使用 oracle transform 检查匹配两端 3D 距离是否小于 0.2 m。GT 不参与 token matching、RANSAC、候选选择或 refinement。

总体结果：

```text
token match 物理正确率均值：0.67%
token confidence 与正确率 Spearman：0.068
token confidence 与 coarse 方向正确性 Spearman：-0.143
```

因此当前 confidence 几乎不能判断 token matches 是否正确。

## 6. 为什么 token 会失败

匹配可视化显示，大量高置信匹配集中在：

```text
地板
水平支架
重复灯光
相同图像 patch 行列位置
```

这些区域在不同相机里外观相似，但不是同一个物理位置。

decoder token 尤其容易给出接近 0 位移的高置信对应。由这些对应求出的 SE(3) 接近单位变换，因此 coarse rotation error 基本等于真实 camera cut 的视角变化。

另一个重要现象是：错误 token 变换有时能得到比 oracle 更高的最近邻 pointcloud overlap。这是因为摄影棚中存在大量重复平面和结构，错误点云也可以贴到另一个相似表面上。因此：

```text
高 pointcloud nearest-neighbor overlap
不等于跨 shot 世界坐标正确。
```

不能只用局部点云自洽分数选择 token candidate。

## 7. 难度分析

Safe Hybrid 在所有分组中都回退到 Explicit-only，因此二者指标完全相同。

| 分组 | Explicit T/R | Hybrid T/R |
|---|---:|---:|
| 小视角 | 0.837 / 5.5° | 0.837 / 5.5° |
| 中视角 | 0.910 / 12.9° | 0.910 / 12.9° |
| 大视角 | 1.288 / 11.7° | 1.288 / 11.7° |
| 高背景重叠 | 0.942 / 4.8° | 0.942 / 4.8° |
| 中背景重叠 | 1.069 / 18.5° | 1.069 / 18.5° |
| 低背景重叠 | 1.060 / 10.5° | 1.060 / 10.5° |

当前样本量只有 6，分组结果只能用于方向判断，不能作为最终论文统计。

## 8. 推理时间

GPU3 上一个代表 case：

```text
Human3R continue，9 帧：5.93 s
Human3R fresh，6 帧：3.33 s
Explicit refinement：平均 0.158 s
Hybrid refinement：平均 0.125 s
```

token matching/RANSAC 已包含在 case 对齐流程中，但当前脚本没有把它拆成单独计时项，后续若进入第二阶段应补充。

## 9. 结论

本次 training-free probe 没有证明现有 frozen token 能提供有效的跨 shot coarse SE(3)。

更准确地说：

1. token 可以找到外观相似区域，但很难区分重复结构中的真实物理对应。
2. token confidence 对错误匹配过度自信。
3. Implicit-only 与 Reset raw 基本相同，没有恢复跨镜头方向。
4. 不加保护的 Hybrid 会被 token 初值拖入错误解。
5. position-collapse guard 可以避免破坏 Explicit，但最终 Hybrid 与 Explicit 相同，说明当前 token 没有额外贡献。
6. Explicit human body frame 能较好修正旋转，但平移和背景全局位置仍需要更强约束。

因此下一步不建议直接训练一个 coarse SE(3) head 去拟合这些 raw matches。更合理的第二阶段是先解决跨视角 correspondence 与 reliability：

```text
冻结 Human3R
训练极小 token projector / cross-shot matcher
加入 hard negative：相同 patch 位置、重复地板、重复支架、相似但非同一物理区域
监督 correspondence confidence，而不是直接回归大 SE(3)
通过可靠 matches 再进入显式 residual refinement
```

## 10. 输出

```text
output/v10_implicit_explicit_cross_shot_probe/avatarrex_training_free/
  implicit_explicit_probe_metrics.json
  implicit_explicit_probe_metrics.md
  analysis/all_case_variant_metrics.csv
  analysis/case_camera_errors.png
  analysis/mean_per_frame_camera_error_curves.png
  analysis/token_confidence_diagnostics.png
  cases/*/analysis/selected_token_matches_oracle_diagnostic.jpg
```

