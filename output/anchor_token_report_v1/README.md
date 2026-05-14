# AnchorToken V6 汇报版实验记录

生成时间：2026/05/13

根目录：`/workspace/code/Movie3R/output/anchor_token_report_v1/`

## 0. 总目标

我们近期做的核心事情不是继续把旧的 global ShotToken 塞进 decoder，而是重新设计一个更局部、更几何的 `AnchorToken`。

目标是验证：

```text
在 AABB shot boundary 上，AnchorToken 是否真的携带可用于纠正 pose/camera re-anchor 的信息。
```

当前所有实验都属于 decoder-before proxy：

```text
先不改 Movie3R 主模型，不训练新模块。
先验证 AnchorToken 在进入 decoder / pose path 前是否是正确的、具体的、可用的。
```

核心逻辑：

```text
XFeat semi-dense + RICH mesh verification
-> 找到真实 static background anchors
-> 映射到 Human3R encoder patch token
-> 验证 patch token 对应关系是否存在
-> 验证 anchors 是否能提供 correction
-> 验证 AnchorToken 是否能提供 local residual correction
-> 验证 shuffled / wrong-boundary 负例是否退化
```

## 1. 样本设置

AABB 格式：

```text
[A@t, A@t+1, B@t+2, B@t+3]
```

核心 shot boundary：

```text
A@t+1 -> B@t+2
```

本次汇报版重跑 3 个样本：

| 样本 | 类型 | 作用 |
|------|------|------|
| `BBQ_001_guitar cam06->cam07 f244` | strong | 高质量 anchor，展示方法有效 |
| `BBQ_001_juggle cam02->cam01 f197` | strong | 大量 anchor，展示稳定性 |
| `BBQ_001_guitar cam01->cam03 f5` | weak | 少量 anchor，展示 fallback/gate 必要性 |

## 2. Step1：AABB boundary 上能否找到 anchor，并映射到 encoder patch token

脚本：

```text
scripts/verify_rich_aabb_anchor_step1.py
```

输出目录：

```text
01_aabb_step1/
```

验证问题：

```text
外部 XFeat/mesh 找到的真实对应点，在 Human3R 图像切 patch、变 token、经过 encoder 后，是否仍然能在对应 patch token 里看到相似特征？
```

结果：

| 样本 | mesh anchors | unique patch anchors | positive cosine | random cosine | rank median | pos > random |
|------|--------------|----------------------|-----------------|---------------|-------------|--------------|
| `guitar cam06->cam07` | 77 | 41 | 0.594 | 0.249 | 4 | 92.7% |
| `juggle cam02->cam01` | 490 | 179 | 0.750 | 0.282 | 3 | 97.8% |
| `guitar cam01->cam03` | 9 | 7 | 0.486 | 0.315 | 38 | 85.7% |

含义：

```text
1. AABB shot boundary 上确实能找到真实 static background anchors。
2. 这些 anchors 映射到 Human3R encoder patch token 后，对应 token 明显比 random token 更相似。
3. strong samples 中 rank 很靠前，说明 encoder token 中保留了可用跨视角对应关系。
4. weak sample anchor 少，rank 明显变差，说明需要 quality gate。
```

推荐展示图：

```text
01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244/aabb_comparison.jpg
01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244/pair_01_A_t1_to_B_t2_BOUNDARY/00_semidense_mesh_inliers_raw_space.jpg
01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244/pair_01_A_t1_to_B_t2_BOUNDARY/01_anchor_patches_on_human3r_crop.jpg
01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244/pair_01_A_t1_to_B_t2_BOUNDARY/10_encoder_cosine_pos_vs_neg.jpg
01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244/pair_01_A_t1_to_B_t2_BOUNDARY/12_encoder_true_match_rank.jpg
```

## 3. Step2：anchor 是否能提供 correction 信息

脚本：

```text
scripts/analyze_rich_aabb_anchor_correction.py
scripts/build_rich_anchor_evidence.py
```

输出目录：

```text
02_correction_proxy/
```

验证问题：

```text
我们的目标不是单纯找匹配点，而是看这些匹配点能不能提供纠正 shot boundary 偏移的信息。
```

比较方法：

```text
no correction: current patch 直接找 reference 同位置
translation: 所有 patch 使用同一个平均 dx,dy
affine: 用 anchors 拟合二维整体变换
```

结果：

| 样本 | anchors | no correction | translation | affine |
|------|---------|---------------|-------------|--------|
| `guitar cam06->cam07` | 41 | 3.16 | 4.32 | 1.04 |
| `juggle cam02->cam01` | 179 | 3.16 | 1.04 | 0.76 |
| `guitar cam01->cam03` | 7 | 10.03 | 2.47 | 0.48 |

含义：

```text
1. anchors 不只是匹配点，确实能提供 correction evidence。
2. 简单 translation 不可靠，因为不同区域不是同一个偏移。
3. affine 更适合作为 coarse re-anchor prior。
4. weak sample 即使 affine 拟合误差低，也要因为 anchor 数少而低 gate。
```

推荐展示图：

```text
02_correction_proxy/BBQ_001_guitar_cam06_cam07_f00000244/correction_prediction_overlay_clean.jpg
02_correction_proxy/correction_proxy_summary.jpg
02_correction_proxy/BBQ_001_guitar_cam06_cam07_f00000244/correction_prediction_overlay.jpg
02_correction_proxy/BBQ_001_guitar_cam06_cam07_f00000244/correction_sampling_error_chart.jpg
02_correction_proxy/evidence/anchor_evidence_summary.jpg
02_correction_proxy/evidence/BBQ_001_guitar_cam06_cam07_f00000244/lookup_error_chart.jpg
02_correction_proxy/evidence/BBQ_001_guitar_cam06_cam07_f00000244/affine_correction_field.jpg
```

## 4. Step3：AnchorToken 是否能在 affine 基础上提供 local residual correction

脚本：

```text
scripts/prototype_rich_anchor_tokens.py
```

输出目录：

```text
03_anchor_token_prototype/
```

AnchorToken 结构：

```text
AnchorToken_k = {
    key_cur_feature: F_cur[j],
    value_ref_feature: F_ref[i],
    ref_pos_norm: pos_ref[i],
    cur_pos_norm: pos_cur[j],
    delta_uv_norm: pos_cur[j] - pos_ref[i],
    confidence,
    mesh_error_px,
    encoder_cosine
}
```

验证方式：

```text
leave-one-out：拿掉一个真实 anchor，用剩余 AnchorTokens 预测它应该对应的 reference patch。
```

结果：

| 样本 | tokens | same | affine | token-soft | token-affine-residual |
|------|--------|------|--------|------------|-----------------------|
| `guitar cam06->cam07` | 41 | 3.16 | 1.15 | 1.41 | 0.82 |
| `juggle cam02->cam01` | 179 | 3.16 | 0.82 | 1.58 | 0.66 |
| `guitar cam01->cam03` | 7 | 10.03 | 1.05 | 1.46 | 1.14 |

含义：

```text
1. 只把 AnchorToken 当 nearest-neighbor memory 不够，token-soft 不稳定。
2. 最有效形式是 global affine coarse re-anchor + local AnchorToken residual。
3. strong samples 中 token residual 优于纯 affine。
4. weak sample 中 token residual 不优于 affine，需要 fallback。
```

推荐展示图：

```text
03_anchor_token_prototype/BBQ_001_guitar_cam06_cam07_f00000244/anchor_token_lookup_overlay_clean.jpg
03_anchor_token_prototype/anchor_token_prototype_summary.jpg
03_anchor_token_prototype/BBQ_001_guitar_cam06_cam07_f00000244/anchor_token_leave_one_out_chart.jpg
03_anchor_token_prototype/BBQ_001_guitar_cam06_cam07_f00000244/anchor_token_lookup_overlay.jpg
```

## 5. Step4：AnchorToken 是否是具体信息，而不是泛泛 shot label

脚本：

```text
scripts/validate_anchor_token_specificity.py
```

输出目录：

```text
04_specificity_controls/
```

验证问题：

```text
AnchorToken 进入 decoder / pose path 前，是否真的像 human token 一样携带明确、有用的信息？
```

对照组：

```text
correct_anchor_token: 正确 boundary 的 token
spatial_only_token: 忽略 feature，只靠空间位置
shuffled_value_token: key/位置正确，但 residual value 打乱
wrong_boundary_token: 使用另一个 boundary 的 token residual
```

结果：

| 样本 | tokens | affine | correct token | spatial-only | shuffled | wrong-boundary |
|------|--------|--------|---------------|--------------|----------|----------------|
| `guitar cam06->cam07` | 41 | 1.15 | 0.77 | 0.84 | 1.18 | 1.33 |
| `juggle cam02->cam01` | 179 | 0.82 | 0.65 | 0.68 | 0.82 | 0.78 |
| `guitar cam01->cam03` | 7 | 1.05 | 1.11 | 1.13 | 1.16 | 1.13 |

含义：

```text
1. strong samples 中 correct token 优于 affine-only。
2. shuffled value 和 wrong-boundary token 会退化。
3. 这说明 token key-value 绑定和 boundary specificity 有意义。
4. AnchorToken 不是泛泛的“有 anchor”提示，而是具体 local residual correction evidence。
5. weak sample 再次证明低 anchor 数需要 fallback。
```

推荐展示图：

```text
04_specificity_controls/anchor_token_specificity_summary.jpg
```

## 6. Step5：推理/训练时是否需要保留所有 anchors

脚本：

```text
scripts/validate_rich_anchor_token_selection.py
```

输出目录：

```text
05_topk_quality_gate/
```

验证问题：

```text
实际模型接入时是否需要把所有 anchors 都传入，还是少量 top-K tokens 就够？
```

结果：

| 样本 | tokens | gate | best strategy | affine | token residual | improvement |
|------|--------|------|---------------|--------|----------------|-------------|
| `guitar cam06->cam07` | 41 | strong | diverse top-8 | 1.10 | 0.77 | +0.32 |
| `juggle cam02->cam01` | 179 | strong | random K=64 baseline | 0.81 | 0.65 | +0.16 |
| `guitar cam01->cam03` | 7 | fallback | random K=4 | 1.25 | 1.24 | +0.02 |

含义：

```text
1. 不需要保留所有 anchors。
2. strong samples 中 8-16 个高质量 / 空间分散 tokens 通常足够。
3. anchor < 8 时 gain 不可靠，应 fallback 或弱启用。
```

推荐展示图：

```text
05_topk_quality_gate/anchor_token_selection_summary.jpg
05_topk_quality_gate/BBQ_001_guitar_cam06_cam07_f00000244/anchor_token_selection_chart.jpg
```

## 7. 离线 cache 状态

已生成 guitar high-overlap cache：

```text
/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1/
```

统计：

```text
candidates: 185
cached: 185
skipped: 0
frame_stride: 10
camera_pairs: 6-7,5-6,4-5,3-4,1-2
top_k_tokens: 16
mean quality_gate: 0.793
mean unique_anchor_patch_pairs: 120.49
size: 923K
```

当前不是相机全排列，而是 high-overlap pairs。后续建议：

```text
第一阶段：high-overlap pairs 作为 clean training signal。
第二阶段：加入 medium-overlap pairs 提高多样性。
第三阶段：low-overlap pairs 作为 hard validation / fallback 测试。
```

## 8. 当前最终结论

已经证明：

```text
1. AABB shot boundary 上能找到 mesh-verified static background anchors。
2. 这些 anchors 可以映射回 Human3R encoder patch token，并且对应 token 明显更相似。
3. anchors 不只是匹配点，能提供 correction evidence。
4. affine 比 mean translation 更适合作为 coarse re-anchor prior。
5. AnchorToken residual 在 strong samples 中能进一步优于 affine-only。
6. shuffled value / wrong-boundary 负例退化，说明 AnchorToken 携带具体 local correction 信息。
7. 低 anchor 数样本不稳定，必须使用 quality_gate/fallback。
```

尚未证明：

```text
AnchorToken 接入 Movie3R pose/camera path 后一定改善最终 3D / camera / SMPL 输出。
```

下一步：

```text
把 offline cache 接入 dataset / loader，先做 pose/camera path 的受控小模型实验。
仍然不改 encoder，不让 anchor 进入完整 decoder token sequence。
```
