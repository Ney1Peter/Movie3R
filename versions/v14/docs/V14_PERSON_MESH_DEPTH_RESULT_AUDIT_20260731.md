# V14 Person Mesh-Depth 三案例结果审计（2026-07-31）

## 1. 审计范围与结论

只读审计对象：

- `versions/v14/probe_b0_da3_person_mesh_depth.py`；
- `output/v14/fine_alignment_research/b0_da3_person_mesh_depth/cases/` 下 3 个 case JSON；
- 聚合报告 `v14_b0_da3_person_mesh_depth.json/md`。

没有修改脚本，没有调阈值或重跑美化结果。

核心结论：9 人中只有 1 人 accepted，不是因为普遍缺少像素，而是因为 7 人的同像素 DA3/mesh surface residual 空间离散度过大，1 人的 mesh 在图内没有任何 triangle silhouette，只有 1 人通过固定 MAD gate。唯一 accepted 人的 predicted residual 为 `-0.10657 m`，GT oracle 为 `+0.14929 m`，符号相反，实际将 root error 从 `0.18343 m` 恶化到 `0.27716 m`。

新脚本相较 bbox 版本确实解决了 pixel mapping 和矩形背景采样的大问题；但它只在一个**一维欧氏 range 模型**下等价于保留 surface-to-root offset，不等价于严格的 root-ray projected offset。当前最明确的可修实现问题是：应计算 `dot(x_DA3-x_mesh, root_ray)`，而不是 `||x_DA3||-||x_mesh||`；同时唯一 accepted 样本的 `MAD > |median residual|`，现有 gate 无法保证修正符号可靠。

## 2. 9 人为何只接受 1 人

固定 gate 参数：

```text
min_pixels             = 96
max_residual_mad_m     = 0.25 m
valid depth scale      = (0.05, 20.0)
confidence retention   = person-visible pixels 内 top 70%
erosion                = 1 次 3x3
```

实际 rejection 计数：

| Reason | 人数 | 含义 |
|---|---:|---|
| `surface_residual_dispersion_gate` | 7 | 像素数和 scale 都通过，但 surface residual MAD 大于 `0.25 m` |
| `too_few_same_surface_pixels` | 1 | rasterized silhouette/visible/valid 全为 0 |
| `accepted` | 1 | 像素数、scale 和 MAD 都通过 |

因此，`1/9` coverage 不是 min-pixel threshold 太高造成的。除一个完全无 silhouette 的人外，其余每人都有 `2,843--37,138` 个 confidence-filtered pixels，远高于 96。低接受率主要是 DA3 surface 与 predicted mesh surface 在人物区域内不呈稳定的单一位移。

## 3. 每人的 gate 与数值

`Coverage` 写作 `visible/silhouette`；`Kept` 写作 confidence filter 后 `valid/visible`。由于代码固定去掉最低 30% confidence，非空样本的 `Kept` 必然约为 70%，它不代表与真实人体 mask 的 IoU。

| Case | Person | Scale | Silhouette | Visible | Valid | Coverage | Kept | Residual median (m) | MAD (m) | 结果/reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c0→c1 | person0 | 2.7822 | 52,811 | 50,528 | 35,369 | 95.7% | 70.0% | -1.0744 | 0.2833 | reject: dispersion |
| c0→c1 | person1 | 2.7822 | 54,485 | 53,055 | 37,138 | 97.4% | 70.0% | +1.4063 | 0.8368 | reject: dispersion |
| c0→c1 | person2 | 2.7822 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | reject: too few pixels |
| c0→c3 | person0 | 2.7147 | 51,253 | 49,777 | 34,844 | 97.1% | 70.0% | +1.1814 | 1.1232 | reject: dispersion |
| c0→c3 | person1 | 2.7147 | 36,589 | 34,287 | 24,001 | 93.7% | 70.0% | -0.1110 | 0.8156 | reject: dispersion |
| c0→c3 | person2 | 2.7147 | 45,982 | 44,786 | 31,350 | 97.4% | 70.0% | +1.2758 | 0.2761 | reject: dispersion |
| c1→c4 | person0 | 2.7038 | 34,297 | 32,853 | 22,997 | 95.8% | 70.0% | **-0.1066** | **0.1603** | **accepted** |
| c1→c4 | person1 | 2.7038 | 4,547 | 4,061 | 2,843 | 89.3% | 70.0% | +0.4863 | 0.5533 | reject: dispersion |
| c1→c4 | person2 | 2.7038 | 23,444 | 22,123 | 15,486 | 94.4% | 70.0% | +0.6487 | 1.0049 | reject: dispersion |

具体 JSON reason 与上表一致。7 个 dispersion reject 中没有人因 scale 或 pixel count 失败。

### 3.1 c0→c1/person2 为什么是 0 silhouette

这不是 rasterizer 把正深度 mesh 错误删掉。只读复算显示该人的所有 vertices 都有正 z，但投影范围为：

```text
x: 473.4 .. 722.8
y: -12.1 .. 1054.7
inside-image vertices: 0
与 504×504 image rectangle 相交的 mesh triangles: 0
```

cached bbox `[480.9, 0, 511, 511]` 是对所有 vertex x/y 独立取 min/max 后 clamp 得到的包围矩形；它可以看起来与图像相交，即使没有一张真实 triangle 与图像相交。mesh z-buffer 返回 0 是合理拒绝，也说明旧 bbox-core 方法在这个人上会采到虚假的“人体区域”。

## 4. 唯一 accepted 人的完整审计

Case/person：`three_t0500_c1_c4_k0/person0`。

| 量 | 数值 |
|---|---:|
| depth scale | 2.703789 |
| predicted root range | 2.288076 m |
| predicted mesh surface range median | 2.171032 m |
| scaled DA3 surface range median | 2.132006 m |
| reported scalar surface-to-root offset | 0.117044 m |
| per-pixel surface residual median / predicted residual | **-0.106567 m** |
| residual MAD | **0.160294 m** |
| applied residual | -0.106567 m（未触发 ±0.30 m cap） |
| GT oracle ray residual | **+0.149285 m** |
| B0 root error | 0.183427 m |
| corrected root error | **0.277164 m** |
| oracle-ray root error | 0.106580 m |
| silhouette / visible / valid | 34,297 / 32,853 / 22,997 |

判断：

1. 像素 coverage 数量充足，overlap/erosion 后仍保留 silhouette 的 95.8%，confidence 后保留 22,997 pixels；不是小样本偶然通过。
2. 但是 `MAD=0.1603 m` 大于 `|median|=0.1066 m`。残差的典型离散度比要执行的修正还大，当前 gate 只检查绝对 `MAD<0.25 m`，没有检查零是否落在 robust residual interval 中，也没有检查 sign confidence。
3. predicted 和 oracle 符号相反，且未触发 cap，所以恶化是 estimator/gate 本身造成的，不是 clip 造成的。
4. `median(observed)-median(predicted)=-0.0390 m`，不等于 `median(observed-predicted)=-0.1066 m`。脚本走的是逐像素对应 residual 的 median，JSON 中单独列出的两个 surface median 不能直接相减来复现 applied delta。

不应通过把 `0.25` 临时调小来“修好”报告：两个接近阈值的 reject 分别是 c0→c1/person0（residual `-1.074`, oracle `-0.390`，同号）和 c0→c3/person2（residual `+1.276`, oracle `-0.323`，反号）。单调调 MAD 阈值不能稳定地区分正确和错误符号。

## 5. Depth scale 是否合理

三案 scale：

| Case | B0 baseline (m) | DA3 baseline (unit) | Scale |
|---|---:|---:|---:|
| c0→c1 | 1.5135 | 0.5440 | 2.7822 |
| c0→c3 | 1.6029 | 0.5905 | 2.7147 |
| c1→c4 | 3.9425 | 1.4581 | 2.7038 |

三个独立 camera pair 的比例只相差约 2.9%，且全部远离代码的 `(0.05,20)` 边界。对 DAE-base 的 arbitrary pair gauge 来说，这组数值内部一致、没有明显 scale 爆炸。Human3R→DA3 映射最大 bbox residual 分别为 `2.14e-5 / 1.91e-5 / 1.33e-5 px`，也没有 pixel bridge 错位迹象。

但这只能说明“按冻结 B0 baseline 标定”的内部计算合理，不能证明 metric ground-truth scale 正确。当前仍只有 forward DA3，没有 reverse scale/person residual 一致性；B0 baseline 的长度误差会直接按比例进入 observed surface range。

另外，scale 是正数，所以它会放大或缩小 surface residual 的幅值，但不会单独解释 accepted 样本的符号反转。

## 6. 是否真正保留了 surface-to-root offset

### 6.1 在一维 radial-range 模型中：是

代码对每个像素定义：

```text
rho_mesh(p) = mesh_z(p) * Human3R_range_factor(p)
rho_da3(p)  = s * DA3_depth_z(p) * DA3_range_factor(p)
delta       = median_p(rho_da3(p) - rho_mesh(p))
rho_root'   = rho_root + delta
```

若定义每像素 scalar offset：

```text
o(p) = rho_root - rho_mesh(p)
```

则：

```text
median_p(rho_da3(p) + o(p))
= rho_root + median_p(rho_da3(p)-rho_mesh(p))
= rho_root + delta
```

所以脚本虽然没有显式使用 341--343 行记录的 summary offset，但确实等价于保留每像素的**欧氏 radial-range offset**。`surface_to_root_offset_m` 字段只是 `root_range-median(mesh_range)` 的诊断摘要，并非实际参与计算的单个 offset。

### 6.2 在要求的 root-ray 几何中：不等价

严格的 root ray 修正应为：

```text
x_D(p) = s * d_D(p) * inverse(K_D) p
x_M(p) = mesh_z(p) * inverse(K_H) p
r       = normalize(root_camera)
delta_p = dot(x_D(p) - x_M(p), r)
delta   = robust_median(delta_p)
```

当前脚本用的是：

```text
delta_p_current = ||x_D(p)|| - ||x_M(p)||
```

二者只在 observed point、mesh point 与 root ray 共线，或所有 pixel rays 都近似 root ray 时相等。脚本使用整张 silhouette，手脚和人体边缘显然不满足。并且 Human3R 与 DA3 各自 K 不同，同一 image pixel 反投影的预测 ray 也不完全相同。

因此，文档/markdown 中“保留 predicted surface-to-root offset”只能理解为一维 range 近似，不能表述为已经实现了严格的 root-ray surface offset。

## 7. 可修实现问题

以下是语义/审计问题，不是为了美化 3-case 指标而调参：

### P0：将 range difference 改为 root-ray projection difference

这是最直接的几何语义修正。z-buffer 已经提供 `mesh_z(p)`，DA3 已有 `depth/K`；应分别反投影成 3D camera points，再计算：

```text
dot(x_DA3(p)-x_mesh(p), root_ray)
```

这同时正确处理不同 pixel ray 与 Human3R/DA3 intrinsics 的差异。

### P0：增加 residual sign-reliability gate

唯一 accepted 样本的 `MAD/|median|=1.50`，说明当前 `MAD<0.25 m` 只控制绝对散布，不能判断修正方向。应机器检查 robust interval 是否跨 0、bootstrap/分块 median 的符号一致性，以及空间分区后的 residual sign agreement。目的不是把阈值调到刚好拒绝此样本，而是要求 correction direction 可观测。

### P1：增加 forward/reverse person scalar 一致性

当前只跑 forward `[pre,post]`，没有 reverse。应分别标定 pair scale、生成 root-ray scalar candidate，再比较 scalar；不能平均 raw point cloud。

### P1：增加 image-boundary 与 observed-person support gate

mesh z-buffer 去除了 predicted-mesh 间遮挡，但没有：

- 排除紧贴 image boundary 的人；
- 与 combined semantic person mask 求交；
- 处理未进入 identity pairs 的遮挡人物；
- 验证 projected mesh 与 RGB 中实际人体重合。

高 pixel coverage 只表示 predicted mesh 自己覆盖多，不能证明这些 DA3 pixels 是该人。

### P1：结构化 rejection reason

251--258 行把 `MAD` 超限和 `depth_scale` 越界统一写为 `surface_residual_dispersion_gate`。应分别输出 `residual_mad_too_large`、`depth_scale_out_of_range`、`too_few_after_confidence` 等机器 reason，避免错误归因。

### P2：改进 mapping/visibility diagnostic 的表述

`mapping_diagnostic=ok` 只证明 recovered projection bbox 与 cached bbox 一致，不证明 mesh 在图内有 triangle，也不证明与观测人体重合。c0→c1/person2 就是 bbox mapping 通过但 silhouette 为 0 的反例。报告应将其命名为 `projection_convention_recovered`，不要写成笼统的 `mapping valid`。

### P2：补齐严格 fallback 与 GT-free 审计

- camera 只比较 local ndarray copy；没有输出 camera hash 和 rejected-person SMPL hash；
- `delta=0` 的 translation 数值上会回退，但没有逐字节人体参数检查；
- candidate 函数不访问 `cache["gt"]`，但 cache person keys 仍来自 GT identity assignment，不能据此宣称最终 runtime 已完全 GT-free。

## 8. 最终判断

这次 `1/9 accepted` 是有信息量的失败：mesh z-buffer 成功揭露了旧 bbox 方法看不见的“人物根本不在图内”问题，并且大多数人的 DA3/mesh same-pixel residual 确实高度不一致。当前 precision gate 阻止了 7 个高离散候选，这是合理的。

但唯一 accepted 人符号错误，说明现有 gate 尚未达到 precision-first；同时当前 estimator 仍是 radial-range 近似，而非严格 root-ray surface-to-root 方法。因此可以相信“当前 mesh-range estimator 尚不可部署”，不能外推为“显式 mesh + DA3 root-ray 方法不可行”。下一次实验应先修正 3D root-ray projection 语义和 sign-reliability/bidirectional gate，再在相同 3 case 上做不调参的等协议复测。
