# B0+BRTC-LC 的 Multi-THuMBS 指标缓存可用性审计

> 日期：2026-08-01。全程只读取已有 JSON；未使用 GPU、未重新推理。

## 1. 最终结论

必须把两层结果分开：

1. 旧 EgoHumans raw 连续链保存了逐帧轨迹，可计算完整的本地 provisional 指标；
   但它没有 B0 或 BRTC-LC。
2. 当前最佳 B0+BRTC-LC 保存的是独立 cut 的 first-post fixed-world 结果，能够严格
   报 root/joint/vertex/layout proxy，不能把它们改名成论文 W/WA/MPJPE/MPVPE/Accel/ATE/IDs。

因此目前仍没有 B0+BRTC-LC 在 EgoHumans 同数据、同 provisional evaluator 下的完整对表。

## 2. raw EgoHumans：可计算但不是当前方法

| Scope | W | WA | MPJPE | MPVPE | Accel Δ² | Accel physical | ATE | IDs/stream |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Human3R raw, 3×15 frames | 1088.3 | 405.1 | 109.3 | 130.0 | 58.33 | 52.49 | 1.848 | 4.00 |

这组数据属于本地 EgoHumans `001_legoassemble` 自建短链，只能诊断 raw Human3R；
不能作为 BRTC-LC 结果，也不是论文官方 split。

## 3. B0+BRTC-LC：当前能严格报告的 fixed-world proxy

| Split | Method | Root | World joint | World vertex | Pair distance | Pair vector |
|---|---|---:|---:|---:|---:|---:|
| fresh three offset1 | b0 | 377.9 | 411.7 | 389.1 | 134.1 | 329.7 |
| fresh three offset1 | b0_brtc_lc | 231.4 | 274.5 | 252.5 | 98.4 | 258.8 |
| post-hoc dance+box | b0 | 479.8 | 499.5 | 498.3 | 69.4 | 308.8 |
| post-hoc dance+box | b0_brtc_lc | 263.9 | 314.6 | 310.9 | 54.8 | 274.2 |

fresh `three offset1` 是策略冻结后的自动-ID 确认：42 cuts、
125 人、覆盖率 88.0%、camera 最大改动 `0.0`。BRTC-LC 的 world joint/vertex 为 `274.5/252.5 mm`。

`dance+box` 已用于发现独立修正的 layout failure，因此共识版只能算 post-hoc support；其 world joint/vertex 为 `314.6/310.9 mm`。

这些值没有 pelvis alignment、trajectory Sim(3) 或论文 aggregation，禁止与
Multi-THuMBS 的 MPJPE/MPVPE 同列比较。

## 4. BRTC 论文指标可用性

| 指标 | 状态 | 原因 |
|---|---|---|
| W-MPJPE | `unavailable_for_strict_BRTC_comparison` | Saved BRTC artifact contains one corrected post boundary frame per independent cut, not a continuous corrected identity trajectory under the paper protocol. W-MPJPE additionally needs a declared initial-frame alignment and full track. |
| WA-MPJPE | `unavailable_for_strict_BRTC_comparison` | Saved BRTC artifact contains one corrected post boundary frame per independent cut, not a continuous corrected identity trajectory under the paper protocol. WA-MPJPE additionally fits on the complete evaluated trajectory. |
| MPJPE | `unavailable_as_paper_column` | Saved joint_error is fixed-world, unaligned error, not per-frame pelvis-centered MPJPE. BRTC is a rigid translation, so a correctly pelvis-centered MPJPE would be unchanged, but its B0 value was not stored in this report. |
| MPVPE | `unavailable_as_paper_column` | Saved vertex_error is fixed-world, unaligned SMPL-X error, not topology-declared, pelvis-centered MPVPE. Rigid BRTC translation would cancel under pelvis alignment. |
| Accel | `unavailable_for_BRTC` | Saved BRTC artifact contains one corrected post boundary frame per independent cut, not a continuous corrected identity trajectory under the paper protocol. The paper also does not publish Accel coordinates, fps, or unit. |
| ATE | `unavailable_as_ATE` | BRTC camera is bit-exact B0, so it cannot change ATE. Saved BRTC reports contain only per-cut first-post camera error, not a declared aligned camera trajectory ATE. |
| IDs | `unavailable_as_official_IDs` | Fresh `three` retains automatic boundary association correctness, but not native continuous track IDs or the paper's miss/entry/exit/aggregation protocol. |

## 5. 论文参考线：当前不能判断胜负

Multi-THuMBS EgoHumans 报告：

```text
W/WA/MPJPE/MPVPE = 279.0/166.0/228.3/262.2 mm
Accel/ATE/IDs = 27.3/0.7/0.97
```

raw EgoHumans 与 BRTC proxy 各缺一半条件，任何‘已经打过’或‘没有打过’的数值
结论都不成立。当前唯一可靠结论是 BRTC-LC 显著改善 fixed-world root/layout，
但刚性平移不会修复 pelvis-centered 内部 pose/shape。

## 6. 为什么不能仅靠现有缓存补出 BRTC EgoHumans

- EgoHumans raw evaluator 的三条 V13 cache 没有当前 frozen B0+BRTC shift；
- B0+DA3 EgoHumans JSON 只保存 boundary 和标量误差，没有可直接复用的 BRTC 人体几何；
- BRTC `three/dance/box` cache 属于另一 MultiHuman capture、相机和 cut 构造；
- 混拼上述产物会把不同 checkpoint/forward/cache 当成同一次预测，结论无效。

## 7. 最小闭环

下一次只需在 EgoHumans 已有 chain 上用同一 frozen forward 保存：

```text
per-frame B0 camera c2w
stable/native identity
B0 and BRTC corrected 24 joints + 6890 vertices
GT visibility/miss/FP association
all pre/post frame indices and timestamps
```

随后复用 `eval_multithumbs_protocol.py`，即可得到 B0 与 B0+BRTC-LC 的同口径
provisional W/WA/MPJPE/MPVPE/Accel/ATE/IDs。作者协议公开后再做正式对榜。

## 8. 产物

```text
versions/v14/eval_brtc_multithumbs_cache_audit.py
versions/v14/docs/V14_BRTC_MULTITHUMBS_CACHE_AUDIT_20260801.md
output/v14/fine_alignment_research/brtc_multithumbs_cache_audit/audit.json
```
