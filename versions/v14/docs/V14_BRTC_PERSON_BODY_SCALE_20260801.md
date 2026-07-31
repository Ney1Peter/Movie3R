# Frozen BRTC person-local robust body-scale: blind validation

Frozen policy SHA256: `8a77bb5e4a1a6483dbd304aa1157704a938a3547a49b1f0fc13cd74a887767f7`.

## 结论

该实验是 **NO-GO**，不能替换 BRTC v1，也不应根据 held-out 结果继续调 fraction/cap。

稳定骨长确实捕获了一部分跨 shot 的人体尺度抖动：three offset1 和 box 的 fixed、
pelvis-centered joint/vertex 全部改善，EgoHumans 的 W、WA、pelvis MPJPE/MPVPE 与 joint
Accel 也略有改善。但是 dance 的 fixed vertex 与 joint Accel 变差，EgoHumans 的 fixed
joint/vertex 分别变差 0.239/0.811 mm。因此它没有形成跨数据、跨指标共同成立的尺度
修正规律。

## 方法

模块在冻结 BRTC v1 完成平移之后运行。对每个 BRTC-accepted anonymous match，只读取
last-pre 与 current-post 的 Human3R joints，并在同一个 matched person 内比较 21 条
SMPL torso/limb 边：双腿、pelvis-spine-neck-head 链，以及双肩到手腕链。脸、手指与
脚趾不参与。

每条有效边产生：

```text
log_ratio_e = log(pre_bone_length_e / post_bone_length_e)
robust_ratio = median(log_ratio_e)
MAD = median(abs(log_ratio_e - robust_ratio))
scale = clip(exp(fraction * robust_ratio), 1-cap, 1+cap)
```

边数不足或 MAD 超过冻结阈值时 scale 精确回退 1.0。尺度围绕 post person 自己的
`native root` 施加，只缩放 joints/vertices：

```text
root'     = root
joints'   = root + scale * (joints - root)
vertices' = root + scale * (vertices - root)
```

相机、native root、native-root pair layout、BRTC-rejected 与 unmatched person 都
bit-exact。无图像、GT runtime 输入、未来帧、新预训练模型或身份名。

## 冻结前开发

只读取 `three offset0` 的 41 cuts / 122 people，扫描 36 组
`fraction∈{0.25,0.5,0.75,1.0}`、`cap∈{0.05,0.1,0.2}`、
`max_log_MAD∈{0.01,0.02,0.03}`，`min_valid_edges=12` 固定。入选要求 fixed joint、
fixed vertex、pelvis-centered joint/vertex 全部不差；root 与 pair-root 精确；个人
joint/vertex >5cm harm 不超过 5%。36/36 组通过，冻结前按 joint+vertex 最小选中：

```text
fraction=1.0, relative_cap=0.2, max_log_MAD=0.03, min_valid_edges=12
SHA256=8a77bb5e4a1a6483dbd304aa1157704a938a3547a49b1f0fc13cd74a887767f7
```

| Method | Joint | Vertex | Pelvis joint | Pelvis vertex | Root | Pair vector | Person harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 270.404 | 248.604 | 140.819 | 108.131 | 225.088 | 260.536 | 0% |
| body-scale | 269.730 | 248.105 | 140.258 | 107.642 | 225.088 | 260.536 | 0% |

以上单位为 mm。108 个 BRTC-accepted person 全部产生非 1 scale，范围
`0.892037..1.111171`；BRTC-rejected/unmatched 的最大变化为 0。

## three_offset1

| Method | Root | Joint | Vertex | Pelvis joint | Pelvis vertex | Pair dist | Pair vec | Joint harm >5cm | Vertex harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.231437 | 0.274493 | 0.252451 | 0.140895 | 0.109146 | 0.098351 | 0.258779 | 0.0% | 0.0% |
| body-scale | 0.231437 | 0.273976 | 0.252071 | 0.140606 | 0.108974 | 0.098351 | 0.258779 | 0.0% | 0.0% |

- Scale actions: `110/110`; range `0.891402..1.112871`; fallback `{}`.
- Root max change: `0.000e+00`; rejected/unmatched max change: `0.000e+00`.

## dance

| Method | Root | Joint | Vertex | Pelvis joint | Pelvis vertex | Pair dist | Pair vec | Joint harm >5cm | Vertex harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.125131 | 0.177804 | 0.152914 | 0.108467 | 0.080166 | 0.044141 | 0.078318 | 0.0% | 0.0% |
| body-scale | 0.125131 | 0.177544 | 0.153758 | 0.107908 | 0.079807 | 0.044141 | 0.078318 | 0.0% | 0.0% |

- Scale actions: `121/121`; range `0.945276..1.052672`; fallback `{}`.
- Root max change: `0.000e+00`; rejected/unmatched max change: `0.000e+00`.

Post-shot Accel: `{'trajectory_count': 24, 'candidate_root_accel_delta2_mm_per_frame2': 26.579933379249436, 'candidate_joint_accel_delta2_mm_per_frame2': 66.61927471298733, 'candidate_vertex_accel_delta2_mm_per_frame2': 39.81294799549325, 'brtc_v1_root_accel_delta2_mm_per_frame2': 26.579933379249436, 'brtc_v1_joint_accel_delta2_mm_per_frame2': 66.42773743101297, 'brtc_v1_vertex_accel_delta2_mm_per_frame2': 39.81560397997634}`.

## box

| Method | Root | Joint | Vertex | Pelvis joint | Pelvis vertex | Pair dist | Pair vec | Joint harm >5cm | Vertex harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.372345 | 0.421610 | 0.434528 | 0.175169 | 0.169103 | 0.063069 | 0.427334 | 0.0% | 0.0% |
| body-scale | 0.372345 | 0.420747 | 0.433462 | 0.174748 | 0.168285 | 0.063069 | 0.427334 | 0.0% | 0.0% |

- Scale actions: `154/154`; range `0.959239..1.051740`; fallback `{}`.
- Root max change: `0.000e+00`; rejected/unmatched max change: `0.000e+00`.

Post-shot Accel: `{'trajectory_count': 30, 'candidate_root_accel_delta2_mm_per_frame2': 53.234274574536435, 'candidate_joint_accel_delta2_mm_per_frame2': 81.34189915654208, 'candidate_vertex_accel_delta2_mm_per_frame2': 62.86174645195458, 'brtc_v1_root_accel_delta2_mm_per_frame2': 53.234274574536435, 'brtc_v1_joint_accel_delta2_mm_per_frame2': 81.36936090135175, 'brtc_v1_vertex_accel_delta2_mm_per_frame2': 62.90006695047891}`.

## EgoHumans same-forward CPU

| Method | W | WA | Pelvis MPJPE | Pelvis MPVPE | Fixed joint | Fixed vertex | Joint Accel | Root | Pair dist | Pair vec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b0_brtc_lc | 314.059 | 202.461 | 109.266 | 129.960 | 384.729 | 385.238 | 125.270 | 380.654 | 177.025 | 333.870 |
| b0_brtc_person_body_scale | 313.612 | 202.199 | 109.103 | 129.859 | 384.968 | 386.049 | 125.141 | 380.567 | 177.045 | 333.742 |

- Person harm: `{'fixed_joint': {'count': 121, 'mean_delta_mm': 0.23865871305660408, 'improve_rate': 0.15702479338842976, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}, 'fixed_vertex': {'count': 121, 'mean_delta_mm': 0.8109709004336697, 'improve_rate': 0.0743801652892562, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}, 'pelvis_mpjpe': {'count': 121, 'mean_delta_mm': -0.16293609089181327, 'improve_rate': 0.2231404958677686, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}, 'pelvis_mpvpe': {'count': 121, 'mean_delta_mm': -0.10112618331627234, 'improve_rate': 0.23140495867768596, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}}`.
- Runtime: `{'boundary_count': 6, 'first_frame_replay_max_abs_delta': 0.0, 'root_max_abs_change': 0.0, 'nonidentity_scale_count': 11, 'scale_count': 11, 'scales': [0.9967151560295465, 0.9813680829697828, 1.0175718808220213, 1.0593337424278044, 0.9934318876237637, 0.9663300990003305, 0.967172088365524, 1.0443466907772836, 0.9604772790991688, 1.0209166087559212, 0.9993068985366906]}`.

- Held-out winner: **False**.
- Decision: **NO_GO_ARCHIVE**.
- Frozen bytes were not changed after any held-out result was opened.

## 结果分析

- three offset1 是最干净的正向确认：joint/vertex 改善 0.517/0.380 mm，pelvis-centered
  joint/vertex 改善 0.289/0.172 mm，110/110 个 accepted person 动作，个人 >5cm harm
  为 0，root/pair 完全不变。
- box 也全面改善：joint/vertex 改善 0.863/1.066 mm，pelvis-centered 改善
  0.421/0.818 mm；post-shot joint Accel `81.369→81.342 mm/frame²`。
- dance 暴露冲突：joint 与两项 pelvis-centered 指标改善，但 fixed vertex
  `152.914→153.758 mm`，post-shot joint Accel `66.428→66.619 mm/frame²`。说明骨长
  consistency 不等价于完整 mesh 在世界坐标中的误差改善。
- EgoHumans 上 W/WA 改善 0.448/0.262 mm，pelvis MPJPE/MPVPE 改善 0.163/0.101 mm，
  camera-centered Accel 改善 0.759 mm/frame²，world-joint Accel 改善
  0.129 mm/frame²；但 fixed joint/vertex 变差 0.239/0.811 mm。121 个 person-frame 中
  没有任何 >1cm 或 >5cm 伤害，说明退化很小但稳定存在，仍不满足严格全指标晋级。

## native root 与 EgoHumans 派生 pelvis 的语义

runtime 的 `person["root"]` 确实逐 bit 不变，controlled 数据的 native-root pair 两项也
逐 bit 不变。EgoHumans evaluator 的 `fixed_world_root` 和 pair-root 却是从缩放后的
SMPL vertices 再用 joint regressor 求 pelvis，而不是读取 native root metadata。当
native root 与 regressed pelvis 不完全重合时，围绕 native root 缩放会让这个“派生
pelvis”产生很小变化：fixed root 改善 0.087 mm，pair distance 变差 0.020 mm，pair
vector 改善 0.127 mm。这不是 runtime 偷改 root，而是两个 root 定义不一致。该差异也
提示最终方法必须统一 native root、mesh pelvis 与评测 root 的定义。

## 完整性与文件

- runtime：`versions/v14/b0_person_body_scale_consistency.py`
- dev/freeze/heldout probe：`versions/v14/probe_brtc_person_body_scale_consistency.py`
- tests：`tests/test_v14_brtc_person_body_scale.py`
- dev scan：`output/v14/fine_alignment_research/brtc_person_body_scale/DEV_SCAN.json`
- frozen policy：`output/v14/fine_alignment_research/brtc_person_body_scale/FROZEN_POLICY_BEFORE_HELDOUT.json`
- held-out：`output/v14/fine_alignment_research/brtc_person_body_scale/HELDOUT_RESULTS.json`

冻结策略在 held-out 前后 SHA256 完全一致。EgoHumans camera max change、native-root max
change、first-frame replay delta 均为 0；11/11 个 BRTC-accepted boundary person 执行
scale。5 个定向单元测试覆盖统一尺度恢复、native-root bit exact、MAD fallback bit
exact、unmatched exact 与 combined BRTC accepted-only 行为，全部通过。

该失败仍有研究价值：Human3R 的 body-scale flicker 是真实且可观测的，但一个统一
person-local scalar 只能改善“相对形状”类指标，无法保证完整 mesh 的 fixed-world
指标。后续如果处理尺度，应先统一 mesh pelvis/root 语义，或在显式刚体/骨架配准中
联合约束，而不是继续单独调这个 scalar scale。
