# Frozen B0+BRTC variants：EgoHumans 同 forward Multi-THuMBS provisional 评测

> 2026-08-01；全程 CPU，未使用 DA3/GPU。三路共享当前 V14/V9 checkpoint 的
> 人体、相机与检测，只改变跨 shot 对齐。不是 Multi-THuMBS 官方 split/协议。
> Active B0：`checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth`，SHA256
> `8379243216775adbc886d00e6f93b6492f7d8f1bd67adb4e8ad6fbdd84e47123`。cache/B0 case 中旧 `/dev/shm` 路径
> 已由 `cmp -s` 验证为 bit-exact alias。

## 1. 方法

- `raw_reset`：三个五帧 shot 各自留在本地 gauge；
- `b0`：将缓存的两个 frozen B0 边界按时间顺序累乘；
- `b0_brtc_lc`：在每个边界用匿名 root+torso+joints Hungarian 匹配，调用
  `versions/v14/b0_person_triangulation.py`，把修正平移传播到对应 post shot；
- `b0_brtc_completeness_weighted`：先完整运行 frozen BRTC v1，再把 accepted final
  shift 乘 `matched / max(pre人数, post人数)`；完整匹配时与 v1 bit-exact；
- `b0_brtc_damped_0p8`：使用独立冻结的常数 `0.8` 缩放 individual proposal，再做
  原有 group/layout consensus；
- `b0_brtc_huber_irls_frozen`：使用在独立 `three offset0` 冻结的可靠性加权
  Huber-IRLS ray center，不在本 EgoHumans confirmation 上调参；
- 因果组合严格为 `G0=I, G1=B01, G2=B01@B12`；第二个 cut 的 pre 人体包含
  第一个 cut 已传播的 person shift，再估计第三个 shot 的新 shift；
- GT identity 只用于 evaluator；BRTC 匹配和修正不读 GT。

## 2. 主要指标

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | PA-MPJPE | PA-MPVPE | Accel Δ² | Accel physical | ATE | IDs/stream | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_reset | 1088.2 | 405.1 | 109.3 | 130.0 | 68.8 | 78.7 | 58.33 | 52.49 | 1.848 | 5.67 | 89.6% |
| b0 | 350.6 | 235.2 | 109.3 | 130.0 | 68.8 | 78.7 | 58.33 | 52.49 | 0.119 | 1.00 | 89.6% |
| b0_brtc_lc | 314.1 | 202.5 | 109.3 | 130.0 | 68.8 | 78.7 | 58.33 | 52.49 | 0.119 | 1.00 | 89.6% |
| b0_brtc_completeness_weighted | 312.7 | 202.2 | 109.3 | 130.0 | 68.8 | 78.7 | 58.33 | 52.49 | 0.119 | 1.00 | 89.6% |
| b0_brtc_damped_0p8 | 312.8 | 201.0 | 109.3 | 130.0 | 68.8 | 78.7 | 58.33 | 52.49 | 0.119 | 1.00 | 89.6% |
| b0_brtc_huber_irls_frozen | 314.4 | 202.0 | 109.3 | 130.0 | 68.8 | 78.7 | 58.33 | 52.49 | 0.119 | 1.00 | 89.6% |

这里 `pelvis MPJPE/MPVPE` 对应本地 Human3R/GVHMR 惯例；`PA-*` 另做每帧
Sim(3) Procrustes。论文主文未确认其精确 pelvis/PA 口径。Accel 同时给离散
二阶差分和按 30 fps 换算的物理单位；论文 `27.3` 的公式/单位仍未知。

## 3. 对齐敏感的 fixed-world / layout proxy

| Method | Root | World joint | World vertex | Pair distance | Pair vector | World root Accel Δ² |
|---|---:|---:|---:|---:|---:|---:|
| raw_reset | 1481.0 | 1359.7 | 1313.3 | 188.5 | 1937.8 | 599.59 |
| b0 | 420.2 | 416.2 | 414.9 | 188.5 | 388.4 | 160.52 |
| b0_brtc_lc | 380.7 | 384.7 | 385.2 | 177.0 | 333.9 | 116.01 |
| b0_brtc_completeness_weighted | 378.8 | 382.9 | 383.6 | 175.9 | 333.6 | 115.88 |
| b0_brtc_damped_0p8 | 379.6 | 381.2 | 381.2 | 173.3 | 336.0 | 119.61 |
| b0_brtc_huber_irls_frozen | 378.8 | 382.9 | 383.4 | 175.2 | 331.2 | 116.89 |

## 4. 当前答案

BRTC-LC 在同 forward 连续链上确实有效：相对 B0，W-MPJPE `350.6→314.1 mm`，WA-MPJPE `235.2→202.5 mm`，fixed-world root `420.2→380.7 mm`，pair vector `388.4→333.9 mm`。

本轮最明确的新规律是 `b0_brtc_completeness_weighted`。它相对 current BRTC 同时改善：

- W：`314.059 → 312.735 mm`；
- WA：`202.461 → 202.156 mm`；
- fixed-world root：`380.654 → 378.760 mm`；
- world-root Accel：`116.014 → 115.878 mm/frame²`；
- corrected-post >5 cm harm：`23.8% → 17.5%`。

这是本 EgoHumans 小样本中唯一在 W、WA、fixed-world root、world-root Accel 和 harm 五项上
同时严格优于 current BRTC 的候选。规律也可解释：5/6 个完整匹配 boundary 保持
v1 原样，唯一 `1→3` 的不完整集合自动缩到 `1/3`，直接减少缺人场景中过激的
多人 group action。它是无 GT、无新阈值、无未来帧的可部署安全变体。

但它随后在独立 MultiHuman variable-visibility 22-cut 确认集上没有通过严格
non-regression：相对 v1 虽将 harm `4.5%→2.3%`，并改善两种 layout，却使
root/joint/vertex 分别退化约 `9.8/10.0/8.1 mm`。所以它不能冻结为新主线，只能作为
“人数变化时需要更保守，但线性 completeness 阻尼过强”的探索证据。此外 EgoHumans 的
17.5% harm 仍高于 10% 安全线，first-post harm 仍为 `12.5%`；本地 W/WA 距论文参考值
还差 `+33.7`/`+36.2 mm`。


## 5. 安全性与真实性审计

| Method | Accept | Mean root Δ | Improve | Harm >5cm | First-post harm | Camera max Δ |
|---|---:|---:|---:|---:|---:|---:|
| b0_brtc_lc | 11/14 (78.6%) | -59.8 mm | 43.8% | 23.8% | 18.8% | 0.0e+00 |
| b0_brtc_completeness_weighted | 11/14 (78.6%) | -62.6 mm | 43.8% | 17.5% | 12.5% | 0.0e+00 |
| b0_brtc_damped_0p8 | 11/14 (78.6%) | -61.4 mm | 45.0% | 18.8% | 18.8% | 0.0e+00 |
| b0_brtc_huber_irls_frozen | 11/14 (78.6%) | -62.5 mm | 43.8% | 23.8% | 18.8% | 0.0e+00 |

- generic callback harness 重放 current BRTC v1：bit parity=`True`，geometry max Δ=`0.000e+00`；
- frozen v1 runtime SHA256 仍为 `98b839f4ae2ff130b0c6ecbc4e0e634ba626d2433f148bee3e55ac169aab3327`；
- CPU current-checkpoint 几何回放 cached GPU B0：`True`，labels exact=`True`，最大 root/joint/vertex 差 `0.63 mm`。
- B0 aligned camera 回放最大矩阵差：`4.343e-05`；
- V13 hard-reset raw camera 最大矩阵差：`4.329e-05`，指标 parity=`True`；
- 匿名边界关联 evaluator-only 正确率：`14/14` (`100.0%`)；
- BRTC gate 接受：`11/14` (`78.6%`)；
- 第二个 cut 的 pre 是否继承第一个 cut 的 person shift：`True`，非零继承 `6/7` 人。
- 所有 refinement 的 camera 均与 B0 bit-exact；unmatched person 最大改动均为 `0`。

## 6. 与论文的关系

Multi-THuMBS EgoHumans 参考线为 W/WA/MPJPE/MPVPE = `279.0/166.0/228.3/262.2 mm`，Accel/ATE/IDs = `27.3/0.7/0.97`。

| Method | Local W | Gap to 279 | Local WA | Gap to 166 |
|---|---:|---:|---:|---:|
| b0 | 350.6 | +71.6 | 235.2 | +69.2 |
| b0_brtc_lc | 314.1 | +35.1 | 202.5 | +36.5 |
| b0_brtc_completeness_weighted | 312.7 | +33.7 | 202.2 | +36.2 |
| b0_brtc_damped_0p8 | 312.8 | +33.8 | 201.0 | +35.0 |
| b0_brtc_huber_irls_frozen | 314.4 | +35.4 | 202.0 | +36.0 |

所以按本地 provisional 公式，当前最好候选仍没有达到论文 W/WA 参考线。
本地 pelvis MPJPE/MPVPE 和 ATE 虽数值更小，也不能宣称胜出：论文未发布
supplementary/evaluator/split，且本地 pose 只统计成功匹配帧，ATE 采用短链 Sim(3)。
当前只可证明同 forward 内部增益；正式胜负必须等官方 manifest/公式后重跑。

## 7. 缓存结论

V13 compact cache 字段完整；由于 archive/checkpoint 标签不同，本轮先不假设可混用。
实际 parity 审计证明它与 CPU current-checkpoint hard-reset raw 的 camera/aggregate
指标一致。原 B0 JSON 只保存边界和标量误差，缺 joints/vertices；因此本轮仍用
current checkpoint CPU 重建 45 帧紧凑几何，确保 B0/BRTC 的来源无歧义。

## 8. 产物

```text
versions/v14/b0_person_triangulation_completeness_weighted.py
versions/v14/tests/test_b0_person_triangulation_completeness_weighted.py
versions/v14/eval_brtc_multithumbs_egohumans.py
versions/v14/docs/V14_BRTC_MULTITHUMBS_EGOHUMANS_20260801.md
output/v14/fine_alignment_research/brtc_multithumbs_egohumans/
```
