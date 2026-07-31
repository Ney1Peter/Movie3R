# V14 DA3 Person Pointmap 实现审计（2026-07-31）

## 1. 审计结论

当前可实现、且不破坏已经冻结的 B0 camera 的最小方案是：

1. B0 继续负责粗对齐和最终 camera；camera 不因人体 pointmap 再次改变。
2. DA3 只提供 post frame 的人体可见表面深度。由于当前 checkpoint 是 `DAE-base`，必须先用同一次 DA3 pair 的相机 baseline 与冻结 B0 baseline 求一个**整对共享尺度**。
3. 在 DA3 的 processed-image 坐标中，用逐人 SMPL-X mesh z-buffer silhouette 选像素；不能把 Human3R 的 896 mask 直接索引 DA3 depth，也不能把整个人 bbox 当逐人 mask。
4. 每个 DA3 像素先按 z-depth 反投影，再转成冻结 B0 post-camera 坐标。结合 Human3R mesh 在相同像素处的“表面到 root 的纵向偏移”，估计该人的 root ray depth。
5. 只允许每个人沿自己原来的视觉 ray 做一个 capped rigid translation。任何尺度、坐标、mask、置信度或前后向一致性 gate 失败时，该人必须 bit-exact 保持 B0 结果。

这是一条合理的精对齐 probe 主线，但目前还不是已经被实验确认的最终方法。本文只审计实现语义与最小安全 gate，不宣称 DA3 person depth 已经有效。

## 2. 源码事实

### 2.1 DA3 输出语义

- `/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/src/depth_anything_3/specs.py`
  中的 `Prediction` 明确定义：
  - `depth: (N,H,W)`；
  - `conf: (N,H,W)`；
  - `extrinsics: (N,4,4)`；
  - `intrinsics: (N,3,3)`；
  - `processed_images: (N,H,W,3)`；
  - `is_metric` 与 `scale_factor`。
- 官方导出函数
  `/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/src/depth_anything_3/utils/export/glb.py::_depths_to_world_points_with_colors`
  使用：

  ```python
  ray = inv(K) @ [u, v, 1]
  X_camera = ray * depth[v, u]
  X_world = inv(extrinsics) @ [X_camera, 1]
  ```

  因为 `ray[2] = 1`，所以 DA3 `depth` 是 **camera z-depth**，不是单位 ray 上的欧氏距离。该文件也明确标注输入 extrinsics 为 `w2c`。
- `/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/src/depth_anything_3/api.py::_align_to_input_extrinsics_intrinsics`
  只有在调用方提供 input extrinsics 时才做 Umeyama 对齐并相应缩放 depth。当前 RGB-only 推理没有输入 B0 extrinsics，因此不会自动得到 B0/米制尺度。
- 当前正式路径使用 `checkpoints/DAE-base`，见
  `versions/v14/probe_b0_da3_shared_pose.py` 和
  `versions/v14/probe_b0_da3_egohumans.py`。它不是 `DA3Metric-*`。因此同一次 pair 中 depth、camera translation 和 pointmap 共享一个任意尺度，数值不能直接当米。
- `conf` 是模型的 depth confidence/ranking signal，不是校准概率。跨图片使用固定概率阈值没有依据；优先使用人内相对分位数。

### 2.2 DA3 像素坐标

`Depth-Anything-3/src/depth_anything_3/utils/io/input_processor.py` 的默认设置是
`process_res=504, process_res_method="upper_bound_resize"`：保持宽高比 resize，再把尺寸调整到 patch size 14 的整数倍；batch 内尺寸不同时还会 center-crop 到共同最小尺寸，并同步更新输入 K（如果提供了 K）。

因此，合法的 DA3 像素域只能由以下输出确定：

```text
prediction.processed_images.shape[1:3]
prediction.depth.shape[1:3]
prediction.intrinsics
```

Human3R 896 `img_mhmr`/`K_mhmr` 域中的 mask、head location 或 mesh UV 必须经过完全相同的 resize/pad/crop 仿射变换后，才能索引 DA3 depth。简单地把 896 方图整体 `cv2.resize` 到 `(W_D,H_D)` 会连同 padding 一起缩放，通常是错的。

### 2.3 Human3R 人体信息

- `src/dust3r/utils/smpl_layer.py::SMPL_Layer` 可由 `smpl_rotmat`、`smpl_shape`、`smpl_transl` 和 `K` 重建 `smpl_v3d/smpl_j3d/smpl_v2d/smpl_j2d`。当前工程用 `person_center="head"`，所以 `smpl_transl` 的语义是 head-centered primary keypoint translation，不应误当标准 SMPL pelvis translation。
- `src/dust3r/smpl_model.py` 生成并保存 `K_mhmr`、`img_mhmr`、`smpl_mask` 和投影结果。
- `versions/v14/probe_v14_internal_root_depth.py` 已能读取 `pts3d_in_self_view/conf_self`、重建每人 mesh，并读取 `prediction["msk"]`。但其中 `msk` 是一张合并人体/语义 mask；`split_person_mask` 只是按最近 head location 分割前景，不是真正的逐人可见 silhouette。它适合诊断，不应直接作为最终部署 mask。
- `versions/v13/native_token_probe.py` 中的 `head_locations/head_scores` 是逐检测信息，但没有证据表明 token debug 已提供逐人像素 mask。

结论：当前最可靠的逐人像素归属应来自每人 SMPL-X mesh triangle rasterization + z-buffer，而不是 bbox 或最近 head 的语义 mask 切分。

## 3. 四种“深度”必须严格区分

对 DA3 post frame 像素 \(p=[u,v,1]^T\)：

| 名称 | 公式 | 单位/语义 |
|---|---|---|
| DA3 z-depth | \(d_D(u,v)\) | DA3 任意尺度；等于 DA3 camera 的 z 坐标 |
| B0 camera z-depth | \(z_B = [x_B]_z\) | 标定后 B0 尺度；仍是 z 坐标 |
| B0 欧氏 ray length | \(\rho_B=\lVert x_B\rVert_2\) | camera center 到点的直线距离 |
| 人物固定 ray 参数 | \(\lambda= r_H^T x_B\) | 点在该人物单位视觉 ray 上的投影 |

其中 \(r_H\) 是 Human3R/B0 当前人物 root 的单位视觉 ray。除主点附近且 ray 几乎沿 z 轴外，`d_D`、`z_B`、`rho_B` 和 `lambda` 都不相等。

## 4. 唯一合法的尺度与坐标变换

### 4.1 记号

```text
E_i^D : DA3 world-to-camera extrinsic
C_i^D = inverse(E_i^D) : DA3 camera-to-world
C_i^B : 冻结 B0 camera-to-world
c_i^D, c_i^B : 上述 c2w 的 translation/camera center
K_D : DA3 processed-image 域的 intrinsics
```

`pre` 和 `post` 必须来自同一个 two-frame DA3 forward。不能把不同 forward 的 depth 和 extrinsics 混用。

### 4.2 用 B0 baseline 标定 DA3 pair 的共享尺度

```text
L_B = ||c_post^B - c_pre^B||
L_D = ||c_post^D - c_pre^D||
s   = L_B / L_D
```

对 DA3 pair 必须统一执行：

```text
camera translation <- s * camera translation
depth              <- s * depth
point coordinates  <- s * point coordinates
```

实现时推荐以 DA3 pre camera center 为尺度原点，避免依赖 DA3 world origin：

```text
c_i^{D,s} = s (c_i^D - c_pre^D)
C_i^{D,s} = [R_i^D | c_i^{D,s}]
```

只缩 depth 或只缩 camera translation 都会破坏“pointmap 与 camera 属于同一 gauge”这一不变量。

### 4.3 DA3 post pixel 转冻结 B0 camera point

先在 DA3 post camera 中反投影：

```text
q_D       = inverse(K_D) [u,v,1]^T
x_D_cam   = d_D(u,v) q_D
x_D_cam_s = s x_D_cam
```

矩阵完整版为：

```text
X_D_world_s = C_post^{D,s} [x_D_cam_s, 1]^T
A_B<-D      = C_post^B inverse(C_post^{D,s})
X_B_world   = A_B<-D X_D_world_s
x_B_cam     = inverse(C_post^B) X_B_world
```

因为这里用 post camera 自身作 anchor，上式应数值等价于：

```text
x_B_cam = s d_D(u,v) inverse(K_D) [u,v,1]^T
```

这个等价关系应成为单元测试：矩阵路径与简化路径的最大误差应小于 `1e-5 * max(1, ||x_B_cam||)`。不满足通常意味着混用了 w2c/c2w、frame index 或只缩放了部分量。

### 4.4 从人体表面点恢复 root ray depth

设当前 Human3R/B0 人物 root 在 post camera 中为 \(h\)，其固定单位 ray：

```text
r_H      = h / ||h||
lambda_H = dot(h, r_H) = ||h||
```

只取逐人 mesh z-buffer silhouette 内的 DA3 点。对于同一像素 \(p\)，令 \(m(p)\) 为 Human3R mesh z-buffer 给出的当前可见表面点（B0 post-camera 坐标）。由于 DA3 测到的是人体表面而不是内部 root，必须扣除当前 mesh 的 surface-to-root offset：

```text
o_mesh(p)            = dot(m(p) - h, r_H)
lambda_root_sample(p)= dot(x_B_cam(p), r_H) - o_mesh(p)
```

最终候选：

```text
lambda_D = weighted_median_p(lambda_root_sample(p), weight=relative_conf(p))
delta    = clip(lambda_D - lambda_H, -0.20m, +0.20m)
h_new    = h + delta * r_H
```

更新 SMPL-X 时，对该人的全部 vertices/joints/primary translation 施加同一个 rigid translation：

```text
Delta_camera = delta * r_H
Delta_world  = R_post^B Delta_camera
```

不能做 per-person similarity，也不能因这一候选回头修改 B0 camera。若没有可靠的逐像素 mesh surface，允许用人体中部小区域的 robust surface offset 作第一版近似，但必须单独标记为 fallback，不能把可见表面深度直接当 root depth。

## 5. Human3R mesh 到 DA3 pixel 的正确映射

推荐保存一条显式的原图到各模型输入域的 2D 仿射链：

```text
p_H = A_H<-orig p_orig
p_D = A_D<-orig p_orig
p_D = A_D<-orig inverse(A_H<-orig) p_H
```

其中 `A_H<-orig` 包含 Human3R 到 896 的等比 resize 与 padding；`A_D<-orig` 包含 DA3 resize、patch-size 调整及 batch center crop。逐人 mesh 可先用 `K_mhmr` 投影到 Human3R 896 域，再用上式映射 vertices 到 DA3 grid，并在 DA3 grid 上做 triangle rasterization/z-buffer。

实现要求：

1. mask shape 必须严格等于 `prediction.depth[post].shape`；禁止依赖隐式 resize。
2. 用至少 20 个可见 mesh vertices 做映射 round-trip 测试，`p_H -> p_D -> p_H` 的中位误差应小于 0.5 px。
3. 每人单独 rasterize；多人 pixel ownership 由最近 z-buffer 决定。两个 silhouette 的边界/深度无法可靠区分时，该像素不属于任何人。
4. silhouette 向内侵蚀 2--3 DA3 pixels，排除衣物外缘、背景泄漏、image boundary、invalid depth 与 sky。
5. 合并语义 `msk` 只作为额外交集或 sanity check，不负责逐人 identity。

## 6. 最小可部署 gate

以下 gate 以 precision-first 为目标。阈值是第一轮 probe 起点，必须在验证集上重新标定，但失败行为不能改变。

### 6.1 Pair/camera gate（沿用冻结正式方法）

来自 `versions/v14/b0_da3_fine_alignment.py::DA3FineAlignmentConfig`：

```text
forward/reverse rotation spread <= 5 deg
forward/reverse direction spread <= 5 deg
right rotation vs B0            <= 15 deg
direction vs B0                 <= 30 deg
```

并增加：

```text
L_B >= 0.02 m
L_D >= 1e-4 DA3-unit
s is finite
0.1 <= s <= 10.0       # 首轮 precision gate；超出只表示不采用，不代表样本一定错误
```

baseline 太短时无法可靠识别 DA3 scale，必须 fallback，不能借人体尺寸猜 scale 后仍声称是 camera-anchored 方法。

### 6.2 Person pixel gate

```text
mesh projection finite ratio                  >= 0.95
eroded, non-overlap, positive-depth pixels    >= max(64, 0.20 * eroded_mask_area)
保留 person-mask 内 confidence top 50%
mask touching image boundary                  <= 0.15 of mask perimeter/pixels
combined semantic mask 与 mesh mask IoU        >= 0.25（若 semantic mask 可用）
```

DA3 confidence 只在人内排序；不要使用固定 `conf > 1.05` 之类的跨图阈值。

### 6.3 Robust geometry gate

先去掉 `lambda_root_sample` 的 10/90 percentile，再做 confidence-weighted median。设候选为 \(\hat\lambda\)：

```text
MAD(lambda samples) <= max(0.08 m, 0.08 * abs(lambda_hat))
median tangential distance to root ray <= 0.20 m
angle(DA3 person centroid ray, Human3R root ray) <= 10 deg
abs(forward lambda_hat - reverse lambda_hat) <= 0.20 m
abs(lambda_hat - lambda_H) <= 0.20 m before commit
```

前向与反向应先各自产生一个完整候选并做一致性 gate；不要把两个独立 gauge 的原始 point cloud/depth 直接平均。只有都经各自 pair scale 标定、post-camera anchoring且候选一致后，才可合并两个标量候选。

### 6.4 多人和输出 gate

- 每人独立 gate；某一人失败只回退该人，不影响其他人。
- overlap pixel 不可同时计入两个人。
- 同一 cut 中多人候选若出现明显的前后顺序翻转或互穿，相关人全部 fallback。
- camera、scene pointmap、其他人的 SMPL-X 不得被这个 person gate 改写。
- fallback 时输出的该人参数必须与输入逐字节相同；不要“clip 后的小修正”冒充 fallback。

## 7. 可机器检查的风险清单

至少应把下表实现为 assertion 或结构化 rejection reason，而不是日志文字：

| ID | 风险 | 机器检查 | 失败动作 |
|---|---|---|---|
| R01 | 把 DA3 z-depth 当欧氏 ray length | 验证 `x = d * inv(K)p` 且 `abs(x_z-d)<eps`；禁止 `normalize(ray)*d` | reject person/case |
| R02 | 混淆 w2c/c2w | `E @ C` 与 `C @ E` 都接近单位阵；矩阵路径与简化 post 路径一致 | reject case |
| R03 | 只缩 depth 或只缩 camera | scale 后 DA3 baseline 必须等于 B0 baseline；重建点与 camera 使用同一 `s` 字段 | reject case |
| R04 | 直接把 DAE-base 数值当米 | checkpoint 为非 metric 或 `prediction.is_metric==0` 时，必须存在有效 `scale_source="b0_baseline"` | reject case |
| R05 | 混用不同 DA3 forward 的输出 | depth/K/E 必须携带相同 `inference_id`、frame index 与 processed shape | reject case |
| R06 | Human3R 896 mask 直接索引 DA3 depth | `mask.shape == depth.shape` 且存在非 identity preprocess transform/round-trip report | reject person |
| R07 | bbox/合并语义 mask 冒充逐人 mask | mask source 必须为 `mesh_zbuffer`；overlap count 必须记录并排除 | reject or explicit diagnostic fallback |
| R08 | 可见表面深度直接当 root depth | 每个样本必须记录 `surface_to_root_offset`；缺失时不能走正式 gate | reject person |
| R09 | confidence 被当校准概率 | 阈值类型必须是 `within_person_percentile`，并记录 percentile 与有效数量 | reject person |
| R10 | forward/reverse 原始点云直接平均 | 两方向只能在产生 B0-scaled scalar `lambda` 后比较/融合 | reject case |
| R11 | 改人时反向改写 camera | 输出 camera hash 必须等于输入冻结 B0 camera hash | reject output |
| R12 | gate 失败仍产生位移 | `accepted == false` 时 SMPL 参数字节/hash 必须完全一致 | reject output |
| R13 | per-person similarity 改变体型 | 首版只允许 translation；shape、pose、vertex pairwise distance 不变 | reject output |
| R14 | 使用 GT 作为部署输入 | runtime payload 中不得出现 GT identity/mesh/camera/depth；GT 只能在 evaluator 分支读取 | reject run |

建议每个 case 输出：

```json
{
  "inference_id": "...",
  "checkpoint": "DAE-base",
  "is_metric": false,
  "scale_source": "b0_baseline",
  "b0_baseline_m": 0.0,
  "da3_baseline_unit": 0.0,
  "scale": 0.0,
  "pixel_transform_roundtrip_median_px": 0.0,
  "camera_gate": {},
  "people": [{
    "mask_source": "mesh_zbuffer",
    "valid_pixel_count": 0,
    "confidence_percentile": 0.5,
    "lambda_current_m": 0.0,
    "lambda_candidate_m": 0.0,
    "lambda_mad_m": 0.0,
    "forward_reverse_delta_m": 0.0,
    "accepted": false,
    "rejection_reasons": []
  }]
}
```

## 8. 最小伪代码

```python
def propose_person_root_depth(b0_pair, da3_forward, da3_reverse, human):
    assert_same_inference_payload(da3_forward)
    assert_da3_w2c_inverse_consistency(da3_forward)

    s = norm(b0_pair.c_post - b0_pair.c_pre) / norm(
        da3_forward.c_post - da3_forward.c_pre
    )
    if not camera_and_scale_gate(b0_pair, da3_forward, da3_reverse, s):
        return fallback_exact(human, "camera_or_scale_gate")

    mask, mesh_surface = rasterize_person_mesh_zbuffer_in_da3_grid(
        human.mesh_camera,
        human.K_mhmr,
        human3r_to_da3_pixel_transform,
        da3_forward.depth.shape[-2:],
    )
    mask = erode_and_remove_overlap_invalid_sky(mask)
    keep = mask & within_person_conf_top_half(da3_forward.conf, mask)
    if not pixel_gate(keep):
        return fallback_exact(human, "person_pixel_gate")

    p = homogeneous_pixels(keep)
    observed = s * da3_forward.depth[keep, None] * (inv(da3_forward.K) @ p.T).T
    ray = normalize(human.root_camera)
    surface_offset = (mesh_surface[keep] - human.root_camera) @ ray
    samples = observed @ ray - surface_offset
    lambda_fwd = trimmed_weighted_median(samples, da3_forward.conf[keep])

    lambda_rev = independently_build_reverse_scalar_candidate(...)
    if not robust_and_bidirectional_gate(lambda_fwd, lambda_rev, samples, observed, ray):
        return fallback_exact(human, "robust_geometry_gate")

    delta = clip(robust_merge(lambda_fwd, lambda_rev) - norm(human.root_camera), -.20, .20)
    return rigid_translate_along_ray_only(human, delta * ray)
```

## 9. 实现审计边界

在本文落盘时，`versions/v14/probe_b0_da3_person_pointmap.py` 尚未出现在工作树中，因此本文没有修改或审阅该脚本。待脚本生成后，应逐项对照第 7 节做只读代码审计，特别检查：

1. depth 是否被错误 normalize 成单位 ray distance；
2. DAE-base 是否先用 B0 baseline 标定了整对共享尺度；
3. Human3R 到 DA3 的 pixel transform 是否显式包含 padding/crop；
4. 是否使用 mesh z-buffer surface offset，而非 bbox median depth；
5. gate 失败是否真的 bit-exact 回退且 camera hash 不变。
