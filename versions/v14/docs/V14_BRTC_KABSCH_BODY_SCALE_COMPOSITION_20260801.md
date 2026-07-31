# Frozen individual Kabsch v1 + frozen body-scale v1 composition

Posthoc composition only: no parameter was selected or changed.

Kabsch policy SHA256: `cd51d67ef2779f7959b08a445ae9879fc9244285b099fd72a2348adc12718111`; body-scale policy SHA256: `8a77bb5e4a1a6483dbd304aa1157704a938a3547a49b1f0fc13cd74a887767f7`.

## 结论

组合为 **NO-GO_POSTHOC_COMPOSITION**，不能作为第二个 qualified candidate。body-scale
在 individual Kabsch 之上继续改善 W、WA、pelvis MPJPE/MPVPE 和 Accel，但同时使
EgoHumans fixed joint/vertex 分别回退 0.106/0.612 mm，pair distance 回退 0.011 mm；
dance fixed vertex 也回退 1.025 mm。它没有全门槛优于 individual Kabsch。

组合相对 BRTC v1 仍然全面更好，说明 Kabsch 原有收益足以覆盖这次小幅回退：Ego fixed
joint/vertex 仍净改善 0.690/0.836 mm，W/WA 净改善 1.864/2.747 mm。但“仍优于旧
baseline”不能证明 scale 是有价值的增量；删除 scale 后的 individual Kabsch 更简单，
且 fixed joint/vertex 更好。因此当前只保留 individual Kabsch 候选。

## 固定组合与双状态因果传播

本实验没有 dev scan、threshold 选择或新的 policy。直接读取已经冻结的两个文件：

```text
Kabsch: max_angle=25°, fraction=0.5, observable gate=0
Scale:  fraction=1.0, cap=0.2, max_log_MAD=0.03, min_edges=12
```

顺序写成 `BRTC -> Kabsch -> scale`。两者都绕同一个 native root，旋转矩阵 R 与 uniform
scale s 满足：

```text
root + s * ((x-root) R^T) = root + (s*(x-root)) R^T
```

因此动作本身可交换。流式 Ego replay 保留两类状态：冻结 BRTC translation reference
只负责复现 v1 root；local body state 因果继承上一 shot 的 rotation+scale，供下一 cut
估计。第二 cut 确实读取 inherited scale state。未来帧、图像、GT inference 和新模型
均为 0。

## MultiHuman incremental result versus individual Kabsch

| Split | Method | Root | Joint | Vertex | Pelvis joint | Pelvis vertex | Pair dist | Pair vec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| three_offset1 | individual_kabsch | 0.231437 | 0.271315 | 0.250248 | 0.139418 | 0.105786 | 0.098351 | 0.258779 |
| three_offset1 | composition | 0.231437 | 0.270858 | 0.249870 | 0.138900 | 0.105370 | 0.098351 | 0.258779 |
| three_offset1 | pass |  |  |  |  |  |  | True |
| dance | individual_kabsch | 0.125131 | 0.168764 | 0.148234 | 0.106799 | 0.078886 | 0.044141 | 0.078318 |
| dance | composition | 0.125131 | 0.168686 | 0.149259 | 0.106449 | 0.078552 | 0.044141 | 0.078318 |
| dance | pass |  |  |  |  |  |  | False |
| box | individual_kabsch | 0.372345 | 0.418583 | 0.429938 | 0.171546 | 0.163023 | 0.063069 | 0.427334 |
| box | composition | 0.372345 | 0.417720 | 0.428809 | 0.171152 | 0.162238 | 0.063069 | 0.427334 |
| box | pass |  |  |  |  |  |  | True |

## EgoHumans

| Method | W | WA | Pelvis MPJPE | Pelvis MPVPE | Fixed root | Fixed joint | Fixed vertex | Pair dist | Pair vec | Root Accel | Joint Accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brtc_v1 | 314.059 | 202.461 | 109.266 | 129.960 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 |
| individual_kabsch | 312.769 | 200.029 | 101.526 | 119.928 | 380.688 | 383.933 | 383.791 | 176.559 | 333.091 | 115.698 | 123.167 |
| kabsch_body_scale | 312.196 | 199.714 | 101.341 | 119.832 | 380.602 | 384.039 | 384.403 | 176.570 | 332.967 | 115.680 | 122.931 |

- Incremental delta composition-Kabsch: `{'w_mpjpe_mm': -0.5734226708777328, 'wa_mpjpe_mm': -0.3140581795981632, 'pelvis_mpjpe_mm': -0.18445220072297275, 'pelvis_mpvpe_mm': -0.09576338641839754, 'fixed_world_root_mm': -0.08636283164935321, 'fixed_world_joint_mm': 0.10640757201377937, 'fixed_world_vertex_mm': 0.6117989469493637, 'pairwise_root_distance_mm': 0.010970137054698625, 'pairwise_root_vector_mm': -0.12388680363301319, 'accel_delta2_mm_per_frame2': -0.716692390348328, 'world_root_accel_delta2_mm_per_frame2': -0.017979524747204323, 'world_joint_accel_delta2_mm_per_frame2': -0.23580002026137947}`.
- Net delta composition-BRTC: `{'w_mpjpe_mm': -1.863725566651624, 'wa_mpjpe_mm': -2.7465126902827706, 'pelvis_mpjpe_mm': -7.925020991544841, 'pelvis_mpvpe_mm': -10.127824812453767, 'fixed_world_root_mm': -0.05208635606754797, 'fixed_world_joint_mm': -0.6897567437121097, 'fixed_world_vertex_mm': -0.8355339507886583, 'pairwise_root_distance_mm': -0.4549064430222529, 'pairwise_root_vector_mm': -0.9021604160172956, 'accel_delta2_mm_per_frame2': -3.870910581818734, 'world_root_accel_delta2_mm_per_frame2': -0.3330784588600295, 'world_joint_accel_delta2_mm_per_frame2': -2.339541260059306}`.
- Incremental person harm: `{'fixed_joint': {'count': 121, 'mean_delta_mm': 0.10640757201375484, 'improve_rate': 0.1652892561983471, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}, 'fixed_vertex': {'count': 121, 'mean_delta_mm': 0.6117989469492939, 'improve_rate': 0.10743801652892562, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}, 'pelvis_mpjpe': {'count': 121, 'mean_delta_mm': -0.18445220072294893, 'improve_rate': 0.2231404958677686, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}, 'pelvis_mpvpe': {'count': 121, 'mean_delta_mm': -0.09576338641841536, 'improve_rate': 0.21487603305785125, 'harm_over_1cm_rate': 0.0, 'harm_over_5cm_rate': 0.0}}`.
- Ego all-gate pass vs individual Kabsch: `False`.

- Overall qualified second candidate: **False**.
- Decision: **NO_GO_POSTHOC_COMPOSITION**.

## 关键增量

### 相对 individual Kabsch

- three offset1：joint/vertex 再改善 0.457/0.378 mm，pelvis joint/vertex 再改善
  0.518/0.416 mm，全门槛通过。
- box：joint/vertex 再改善 0.863/1.129 mm，pelvis joint/vertex 再改善
  0.394/0.785 mm，全门槛通过。
- dance：joint 与 pelvis 指标略好，但 vertex `148.234→149.259 mm`，因此失败。
- EgoHumans：W/WA 再改善 0.573/0.314 mm，pelvis MPJPE/MPVPE 再改善
  0.184/0.096 mm，camera-centered Accel、root Accel、joint Accel 分别再改善
  0.717/0.018/0.236 mm/frame²；但 fixed joint/vertex 变差 0.106/0.612 mm，pair
  distance 变差 0.011 mm。

Ego 121 个 matched person-frame 的 fixed joint/vertex 增量伤害都小于 1cm，>5cm harm
为 0。这说明组合不是灾难性失败，而是一个稳定但方向冲突的小 trade-off；严格候选门槛
仍必须判 NO-GO。

### 相对 BRTC v1 的净结果

| Metric | BRTC v1 | Kabsch+scale | Net gain |
|---|---:|---:|---:|
| W | 314.059 | 312.196 | 1.864 |
| WA | 202.461 | 199.714 | 2.747 |
| pelvis MPJPE | 109.266 | 101.341 | 7.925 |
| pelvis MPVPE | 129.960 | 119.832 | 10.128 |
| fixed joint | 384.729 | 384.039 | 0.690 |
| fixed vertex | 385.238 | 384.403 | 0.836 |
| joint Accel | 125.270 | 122.931 | 2.340 |

全部单位为 mm 或 mm/frame²。这个表仅用于确认 Kabsch 收益没有被 scale 完全破坏，不能
用来把组合晋级为第二候选。

## Runtime 完整性

- 两个 frozen policy 文件在评估前后 SHA256 完全相同。
- Ego 6 个 boundary、11/11 个 BRTC-accepted person 都执行 frozen scale。
- first-frame composition replay delta 为 0，native root 与 Kabsch max delta 为 0，
  camera bit-exact。
- 第二 cut 的 inherited scale state 已实际启用。
- MultiHuman 三个 split 的 native root 与 rejected/unmatched max delta 均为 0。
- 两个新增测试验证同 root 下 rotation/scale 可交换，以及 scale state 跨两 cut 因果继承。

产物：

```text
versions/v14/eval_brtc_kabsch_body_scale_composition.py
tests/test_v14_brtc_kabsch_scale_composition.py
output/v14/fine_alignment_research/brtc_kabsch_body_scale_composition/REPORT.json
versions/v14/docs/V14_BRTC_KABSCH_BODY_SCALE_COMPOSITION_20260801.md
```
