# AnchorToken V6 筛选版汇报图

整理时间：2026/05/15

根目录：`/data/wangzheng/iJCV-CODE/Movie3R/output/anchor_token_report_v1/`

## 0. 目录定位

这是一个筛选版展示目录，只保留适合快速汇报的关键可视化图片。

完整的 guitar-only Step1-5 计算结果保存在：

```text
/data/wangzheng/iJCV-CODE/Movie3R/output/anchor_token_report_v1_guitar_only_step2_5/
```

该完整目录包含：

```text
01_aabb_step1/
02_correction_proxy/
03_anchor_token_prototype/
04_specificity_controls/
05_topk_quality_gate/
README.md
```

本筛选版目录当前只保留：

```text
01_aabb_step1/
02_correction_proxy/
03_anchor_token_prototype/
README.md
```

## 1. 样本

当前筛选版只使用两组 `BBQ_001_guitar`：

| 样本 | 类型 | 作用 |
|------|------|------|
| `BBQ_001_guitar_cam06_cam07_f00000244` | strong | 高质量 anchors，展示 affine correction 明显有效 |
| `BBQ_001_guitar_cam01_cam03_f00000005` | weak | 少量 anchors，展示低 anchor 数时需要 fallback / gate |

## 2. Step1：encoder token 中是否还能看到 anchor correspondence

目录：

```text
01_aabb_step1/
```

Step1 的问题是：

```text
XFeat + RICH mesh 找到真实几何 anchor 后，
这些 anchor 映射到 Human3R patch，并经过 encoder 后，
对应 patch token 是否仍然比 random token 更相似？
```

每个样本只保留 5 张筛选图：

```text
00_semidense_xfeat_mesh_inliers.jpg
01_ref_human3r_crop.jpg
02_cur_human3r_crop.jpg
03_human3r_patch_anchor_correspondences.jpg
04_similarity_map_anchor_00.jpg
```

含义：

```text
00_semidense_xfeat_mesh_inliers.jpg
XFeat semi-dense matches 经过 RICH static mesh 验证后的几何 inliers。

01_ref_human3r_crop.jpg
shot boundary 前一帧 / reference frame 的 Human3R crop。

02_cur_human3r_crop.jpg
shot boundary 后一帧 / current frame 的 Human3R crop。

03_human3r_patch_anchor_correspondences.jpg
mesh-verified anchors 映射到 Human3R patch grid 后的对应关系。

04_similarity_map_anchor_00.jpg
选一个 reference anchor patch token，和 current 图所有 encoder patch token 做 cosine similarity，
再 reshape 成 current patch grid 的 heatmap。
```

注意：encoder 后的 patch token 不是 RGB patch，不能直接还原成原图。这里可视化的是 token similarity map，而不是图像重建。

关键结论：

```text
Step1 证明几何 anchors 经过 Human3R encoder 后，仍然能在 patch token 空间里看到对应关系。
strong sample 中真实对应 token 的 cosine 更高、rank 更靠前。
weak sample 虽然仍有信号，但 anchor 数少且 rank 不稳定。
```

## 3. Step2：anchors 是否能提供 correction evidence

目录：

```text
02_correction_proxy/
```

Step2 的问题是：

```text
给定 Step1 已经验证过的 anchor pairs，
能不能从 reference anchor 位置预测 current anchor 位置？
```

每个样本保留 3 张筛选优化后的 overlay：

```text
correction_overlay_no_correction.jpg
correction_overlay_translation.jpg
correction_overlay_affine.jpg
```

图中元素：

```text
magenta / 粉色点：GT mesh-verified current anchor 位置
彩色点：对应方法预测出来的 current 位置
细线：预测位置到 GT anchor 位置的误差线
线越短：预测越准，correction 越好
```

三种方法：

```text
no correction
不做任何 re-anchor，直接假设 ref patch 的 normalized 坐标在 cur 中不变。

translation
对所有 anchor 的 x_cur - x_ref 偏移做加权平均，全图使用同一个 dx,dy。

affine
用 anchors 拟合二维仿射变换：x_cur = A x_ref + b。
它可以表达平移、旋转、缩放、非均匀缩放和 shear，比单一 translation 更适合视角变化。
```

当前两组 Step2 指标：

| 样本 | anchors | no correction | translation | affine |
|------|---------|---------------|-------------|--------|
| `guitar cam06->cam07 f244` | 40 | 3.16 | 4.39 | 1.03 |
| `guitar cam01->cam03 f5` | 7 | 10.03 | 2.47 | 0.48 |

解释：

```text
1. strong sample 中 affine 明显优于 no correction，说明 anchors 能提供 coarse re-anchor evidence。
2. translation 只允许全图同一个偏移，在 cam06->cam07 中反而变差，说明不同区域偏移不是常量。
3. weak sample 虽然 affine 拟合误差低，但 anchors 只有 7 个，后续接入时仍应走 fallback / quality gate。
```

## 4. Step3：AnchorToken local residual 是否优于 affine-only

目录：

```text
03_anchor_token_prototype/BBQ_001_guitar_cam06_cam07_f00000244/
03_anchor_token_prototype/BBQ_001_guitar_cam01_cam03_f00000005/
```

Step3 的问题是：

```text
在 Step2 已经证明 affine coarse re-anchor 有效之后，
AnchorToken 能不能进一步学习局部 residual，
让 held-out anchors 的预测比 affine-only 更准？
```

当前筛选版每个样本保留两张 Step3 主图：

```text
anchor_token_affine_residual_table.jpg
anchor_token_affine_vs_residual_overlay.jpg
```

`anchor_token_affine_residual_table.jpg` 是核心指标表。

两组样本的关键数值：

| 样本 | anchors | affine median | affine + residual median | affine mean | affine + residual mean | within 1 patch | anchors improved |
|------|---------|---------------|--------------------------|-------------|------------------------|----------------|------------------|
| `guitar cam06->cam07 f244` | 40 | 1.13 | 0.83 | 1.13 | 0.87 | 42.5% -> 57.5% | 29 / 40, 72.5% |
| `guitar cam01->cam03 f5` | 7 | 1.05 | 1.14 | 1.91 | 1.84 | 28.6% -> 42.9% | 5 / 7, 71.4% |

strong sample 详细数值：

| 指标 | affine-only | affine + residual | gain |
|------|-------------|-------------------|------|
| valid held-out anchors | 40 | - | - |
| median patch error | 1.13 | 0.83 | 0.26 |
| mean patch error | 1.13 | 0.87 | 0.25 |
| p75 patch error | 1.48 | 1.15 | 0.48 |
| within 1 patch | 42.5% | 57.5% | +15.0 pp |
| within 2 patches | 95.0% | 100.0% | +5.0 pp |
| anchors improved | - | - | 29 / 40, 72.5% |

`anchor_token_affine_vs_residual_overlay.jpg` 是读图主图。

图中元素：

```text
magenta / 粉色点：held-out GT reference anchor 位置
blue / 蓝色点：affine-only prediction
orange / 橙色点：affine + local AnchorToken residual prediction
误差线越短：预测越准
```

解释：

```text
1. affine-only 提供全局 coarse re-anchor，能处理主要的视角变化。
2. local residual 使用剩余 anchors 的局部偏差，修正 affine 无法表达的局部误差。
3. strong sample 中 affine + residual 的 median error 从 1.13 降到 0.83，且 72.5% held-out anchors 得到改善。
4. weak sample 只有 7 个 anchors，mean error 和 within-1-patch 有改善，但 median 不稳定，因此更适合作为 fallback / quality gate 示例。
5. 这说明 AnchorToken 不只是重复 affine，而是在 affine 基础上提供有用的局部 correction signal；但低 anchor 数时仍需要 gate。
```

## 5. 使用建议

汇报时建议按这个顺序看图：

```text
1. Step1 的 03_human3r_patch_anchor_correspondences.jpg
   说明真实几何 anchors 可以映射到 Human3R patch。

2. Step1 的 04_similarity_map_anchor_00.jpg
   说明经过 encoder 后，对应 patch token 仍然能在 current 图里形成相似性热区。

3. Step2 的 correction_overlay_no_correction.jpg
   说明不做 re-anchor 时预测位置和 GT anchor 有明显误差。

4. Step2 的 correction_overlay_translation.jpg
   说明平均平移不一定可靠。

5. Step2 的 correction_overlay_affine.jpg
   说明 affine coarse re-anchor 能明显缩短误差线。

6. Step3 的 anchor_token_affine_residual_table.jpg
   说明 affine + local residual 在 held-out anchors 上整体优于 affine-only。

7. Step3 的 anchor_token_affine_vs_residual_overlay.jpg
   说明 local residual 如何在图像 patch grid 上缩短局部误差线。
```

## 6. 当前结论

这个筛选版目录用于展示三件事：

```text
1. Step1：Human3R encoder token 中仍然保留可用的 anchor correspondence。
2. Step2：这些 anchors 不只是匹配点，还能提供 shot-boundary correction evidence。
3. Step3：AnchorToken local residual 能在 affine-only 基础上进一步降低 held-out anchor patch error。
```

更完整的 Step3-5 证据，包括 specificity controls 和 top-K quality gate，请看：

```text
/data/wangzheng/iJCV-CODE/Movie3R/output/anchor_token_report_v1_guitar_only_step2_5/
```
