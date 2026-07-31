# V14 Person Pointmap Probe 只读代码复审（2026-07-31）

## 1. 结论先行

复审对象：`versions/v14/probe_b0_da3_person_pointmap.py`（592 行）。本次没有修改该脚本，也没有修改正在开发的 mesh 脚本。

当前 `-0.833` 负相关**不能解释为 DA3 人体 pointmap 本身失败**。代码中 DA3 z-depth 转 range 的公式是正确的；在这 3 个 case 中，输入和 Human3R cache 都是方图，512 bbox 到 504 depth 的线性映射也基本正确。负结果的首要来源是：

1. 使用经过裁边的 predicted bbox core 代替逐人可见 mesh mask，core 可能落在背景、遮挡者或错误身体部位；
2. 把不同相机视角下的可见人体表面 range 差直接当 root range 差，没有 surface-to-root correction；
3. 报告所谓 `raw_depth_residual_m` 并不是纯 DA3 residual，而是
   `Human3R memory range difference + DA3 surface range difference`；因此当前相关性不能单独归因给 DA3；
4. 没有 forward/reverse person-depth 一致性 gate，也没有逐人的 mesh overlap、truncation、surface spread gate；
5. 相关性只有 4 个 accepted pair，统计量不稳定。

所以，当前结果足以否定的是**“bbox-core pre/post surface-range transfer”这个具体 estimator**；它不能否定“DA3 + mesh z-buffer + surface-to-root + bidirectional gate”主线。

## 2. R01--R14 pass/fail 表

表中 `PASS（本实验）` 只表示当前三组 2048-square 数据的这次运行没有触发该语义错误，不代表代码可泛化到非方图。

| ID | 状态 | 代码证据 | 复审结论 |
|---|---|---|---|
| R01 z-depth/range | **PASS** | `range_map`，123--128 行 | 使用 `depth * sqrt(x²+y²+1)`，正确地把 DA3 z-depth 转成欧氏 camera range；没有把 z-depth 直接当 range，也没有错误使用 `normalize(ray)*depth`。负相关不是这一项造成的。 |
| R02 w2c/c2w | **PASS（缺断言）** | 254--255 行 | 对 `prediction.extrinsics` 求逆后取 camera center，符合 DA3 extrinsics 为 w2c 的定义。缺少 `E@C≈I` 和矩阵/简化路径单测，但当前使用方向正确。 |
| R03 共享 pair scale | **PARTIAL** | 254--260、306、310 行 | `s=L_B/L_D` 且只乘同一次 forward 的 pre/post range 差，标量计算在 gauge 上成立；但没有显式保存/验证“depth、camera translation、points 共用同一 s”，也没有短 baseline gate。正数 `s` 只会改幅值，不会单独翻转符号。 |
| R04 非 metric checkpoint | **PARTIAL** | 40、256--257、333 行 | 明确用 `DAE-base` 并通过 B0 baseline 标定，未直接把 DA3 unit 当米；但 `run_da3` 丢弃了 `prediction.is_metric/scale_factor`，输出也没有结构化 `scale_source`。目前语义基本正确、机器审计合同不完整。 |
| R05 同一 inference payload | **PASS（结构性）** | 208--226、254--261 行 | depth/K/E 来自同一次 two-image `model.inference` 返回的同一个 dict，frame 0/1 没有跨调用混用。缺少 `inference_id`、shape 和 frame-index assertions。 |
| R06 processed-image 映射 | **PASS（本实验）/FAIL（通用）** | 101--105、258--270、518--526 行 | cache contract 是 2048 square→512 square，FrameReader 强制 768 square，日志确认 DA3 为 504×504；因此本实验 `bbox*504/512` 正确。代码没有保存 `processed_images` 或 resize/crop transform，换成非方图、padding 或 batch crop 后会错，不能外推。 |
| R07 逐人 mask/overlap | **FAIL** | 7--8、101--120、263--290、532 行 | 主 mask 是 rectangular bbox torso core，不是 `mesh_zbuffer`。所谓 overlap 也只是 bbox-core overlap，不能排除遮挡人和背景。多个 accepted bbox 明显贴边/截断，但没有 truncation gate。 |
| R08 surface-to-root | **FAIL（首要）** | 276--314 行 | 直接用 `post_surface_range-pre_surface_range` 更新 root，没有同像素 mesh surface-to-root offset。跨相机时可见表面、朝向、遮挡和所取身体部位都变化，可自然造成符号翻转。 |
| R09 confidence 语义 | **PASS** | 146--153 行 | confidence 只在人内取 30 percentile 并作相对权重，没有当校准概率使用固定阈值。建议仍记录 percentile 和有效比例。 |
| R10 forward/reverse | **FAIL** | 208--213、526 行 | 只运行 `[pre,post]` forward，没有 reverse person candidate 和一致性 gate。它没有直接平均两个 raw gauge（这一窄项没犯错），但完整 R10 安全合同未实现。 |
| R11 camera 冻结 | **PASS（局部）** | 251--252、337--338、533--537 行 | proposal 内 camera 只读，并以 `np.array_equal` 检查 local snapshot，当前报告 camera exact 为 True。该检查只覆盖一个 local array，不是完整输出 camera 文件/hash 审计。 |
| R12 fallback bit-exact | **FAIL** | 308--325、337--338 行 | invalid 时 `delta=0`，translation 数值上回退；但没有检查人体 SMPL 参数/array hash。similarity 的 `camera + 1*(points-camera)` 也不保证逐字节相同。当前唯一 bit-exact assertion 只检查 camera。 |
| R13 禁止 per-person similarity | **FAIL** | 164--180、320--322、363--381 行 | probe 正式生成并汇报 per-person similarity 分支，它会缩放人体尺寸及人间尺度。translation 分支可保留为诊断，similarity 不应进入首版部署候选。 |
| R14 GT 不进入 candidate | **FAIL（严格部署合同）** | 230--238、347--393；cache 来源 `versions/v13/gt_id_consensus.py:613--660` | proposal 函数在 evaluation 前不读取 `cache["gt"]`，这是好的；但 `cache["humans"]` 的 person0/1/2 键来自构建 cache 时的 GT mesh identity assignment。即便代码只把名称当 opaque key，这个运行也不能证明真正 anonymous/GT-free deployment。 |

汇总：`PASS 5`（其中 3 项仅当前范围成立），`PARTIAL 2`，`FAIL 7`。最关键的功能性 FAIL 是 R07、R08、R10；最关键的实验合同 FAIL 是 R14。

## 3. 负相关是否来自实现语义错误

### 3.1 Range vs z：不是错误

代码：

```python
x = (u - cx) / fx
y = (v - cy) / fy
range = depth_z * sqrt(x*x + y*y + 1)
```

这正是从 z-depth 到欧氏 range 的转换。后面 `current_range=||root-camera||` 和沿单位 ray 施加 `delta` 也都使用欧氏距离，因此量的类型一致。

仍需注意：bbox 内不同像素拥有不同 ray。把许多 surface points 的欧氏 range quantile 当成 root-ray range 是 R08 的近似错误，但这和“误把 z-depth 当 range”不是同一个问题。

### 3.2 Bbox/processed-image 坐标：当前缩放正确，但 mask 内容不正确

当前 cache 在 `versions/v13/gt_id_consensus.py` 中由 2048×2048 full frame resize 到 512×512，无 crop；当前 FrameReader 又将同一 full frame resize 到 768×768，DA3 日志为 504×504。因此：

```text
p_DA3 = p_cache * 504 / 512
```

对这三组 case 是正确的。负相关不能简单归因于漏掉了 processed-image crop。

但 bbox 本身由投影 mesh 的 min/max 得到后 clamp 到 image boundary。当前 accepted 人中有明显退化 bbox，例如：

```text
three_t0500_c0_c1_k0/person2 post bbox = [480.9, 0.0, 511.0, 511.0]
three_t0500_c0_c3_k0/person2 post bbox = [0.0, 4.6, 125.5, 511.0]
three_t0500_c1_c4_k0/person1 post bbox = [354.1, 0.0, 511.0, 511.0]
```

对 clamp 后 bbox 再取 25%--72% 高度、中心 40% 宽度，不再保证对应 torso/pelvis；严重截断时 core 很可能是背景或人体边缘。`relative_mad` 小只能说明错误区域深度平滑，不能证明它属于这个人。

### 3.3 Baseline gauge：公式方向正确，但 gate 不完整

代码使用：

```text
s = ||c_post^B0-c_pre^B0|| / ||c_post^DA3-c_pre^DA3||
```

并计算 `s*(post_range-pre_range)`。对于 DAE-base 的 pair gauge，这个标量用法是成立的。`s>0`，所以它不会独自造成 correlation sign 反转。

风险在于：

- 没有使用正式 DA3 forward/reverse pose gate；
- 没有 forward/reverse scale 和 per-person range 一致性；
- 只检查 `0.05<s<20`，没有 B0/DA3 baseline 的绝对下限；
- B0 baseline 本身的长度误差会线性放大 DA3 surface term。

同三组 case 在已有 `da3_shared_pose_three_dev` 报告中都通过正式 camera gate，所以“DA3 camera 完全坏掉”不是首要解释；但 person depth 仍需独立的双向 gate。

### 3.4 Surface-to-root 缺失：是直接语义错误

当前真正计算的是：

```text
delta_pred = (pre_root_range_H3R - current_post_root_range_B0)
           + s * (post_bbox_surface_range_DA3 - pre_bbox_surface_range_DA3)
```

它隐含假设：两个相机看到的 bbox core 对应相同人体表面，且该表面到 root 的纵向 offset 不变。跨 shot/跨相机正好不满足这个假设。正确 mesh 版本至少需要：

```text
lambda_root_sample(p)
  = dot(x_DA3_surface_B0cam(p), root_ray)
  - dot(x_mesh_surface_B0cam(p)-root, root_ray)
```

然后在人内做 robust median，而不是比较两个 bbox 的 surface quantile。

### 3.5 当前相关性还混入了 Human3R memory term

对 4 个 accepted pair，将输出按上述公式拆分：

| Case/person | H3R memory term (m) | DA3 scaled surface term (m) | 报告 predicted (m) | Oracle ray (m) |
|---|---:|---:|---:|---:|
| c0→c1 / person1 | +0.925 | +0.326 | +1.251 | -0.391 |
| c0→c1 / person2 | +0.045 | +1.987 | +2.032 | -0.279 |
| c0→c3 / person2 | -0.646 | +2.241 | +1.595 | -0.323 |
| c1→c4 / person1 | -0.581 | +0.525 | -0.056 | +0.069 |

这说明：

- 报告的 0% sign agreement 与 -0.833 correlation 是**融合 residual 对 oracle**，不是纯 DA3 对 oracle；
- 单看 DA3 surface term，4 个中有 1 个符号与 oracle 一致，仍然很差，但不是报告的 0%；
- 第一个样本主要被 H3R memory term 推错；第二、第三个主要被 bbox surface term 推错；第四个则是两项相互抵消后翻错。

因此 markdown 中 “pointmap-vs-oracle” 这一名称过强。更准确的名称应是 `bbox_surface_memory_fusion_vs_gt_ray`。这不影响“当前 estimator 应拒绝”的结论，但影响失败归因。

## 4. 当前实验中可以相信的结论

1. 在这 3 个指定 case、9 个 GT 可评估人物上，当前 bbox-core estimator 接受 4 人；4 人修正后都没有改善，整体 root mean 从 `0.3209 m` 变为 `0.4207 m`。
2. 对这 4 个 accepted 人，当前**融合 residual**与 oracle residual 的 sign agreement 为 `0/4`，Pearson 为 `-0.833`。它足以停止继续调 bbox quantile/MAD 阈值作为主线。
3. camera local array 在 proposal 内保持 `np.array_equal`，所以这次人体失败不是脚本把 camera 再次改坏造成的。
4. oracle 沿 ray translation 将 9 人 root mean 从 `0.3209 m` 降到 `0.1031 m`，说明 B0 剩余人体误差中确实有很大的 radial component；“冻结 camera、逐人沿 ray 精修”方向仍有价值。
5. oracle translation 的 joint/vertex mean 优于 oracle similarity，且 similarity 会改变体型；首版只做 rigid translation 是更合理的限制。

## 5. 不能外推的结论

1. 不能说 DA3 depth 或 DA3 person pointmap 与真实人体修正天然负相关；当前没有真正的 person mask，也没有 root-depth estimator。
2. 不能说 mesh z-buffer + surface-to-root 方法会失败；当前代码没有实现它。
3. 不能把 `-0.833` 当稳定统计规律；样本量只有 4，且三个来自同一 timestamp/相近人物集合。
4. 不能把当前结果外推到非方图或原版 demo 输入；bbox 到 processed-image 的通用 resize/pad/crop 链没有实现。
5. 不能声称 absolute metric depth 已验证；尺度来自冻结 B0 baseline，且没有 person forward/reverse 检查。
6. 不能声称候选已经完全 GT-free；cache 的人物 key 来自 GT identity assignment。
7. 不能把 `camera_bit_exact=True` 外推为完整 runtime 输出都 bit-exact；当前只比较函数内的 camera ndarray copy，没有检查磁盘输出、历史 camera 或人体 fallback hash。
8. 不能把当前 `pointmap_memory_similarity` 当可部署候选；它违反人体 rigid-shape 约束。

## 6. 对下一版 mesh probe 的最小验收条件

在解释新结果前至少应满足：

1. `mask_source == "mesh_zbuffer"`，排除多人 overlap、image boundary 和侵蚀边缘；
2. 显式保存 Human3R→DA3 pixel transform，并对当前方图得到接近 `504/512` 的 round-trip 验证；
3. 保存 `prediction.is_metric`、`inference_id`、processed shape、B0/DA3 baseline 和 `scale_source`；
4. 每个像素扣除 mesh surface-to-root offset，再估计 root-ray lambda；
5. forward/reverse 各自产生 scalar lambda，只有二者差小于阈值才接受；
6. 输出分别报告 `pure_da3_lambda`、`memory_term` 和 `fused_delta`，禁止再用 “pointmap residual” 混称融合量；
7. 只提交 translation 分支；gate 失败时检查 camera hash 与该人 SMPL 参数 hash 全部不变；
8. 报告 accepted count、rejected reasons 和置信区间，至少扩大到独立 timestamps/camera pairs 后再讨论相关性。
