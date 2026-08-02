# V14 P0：统一 B0 的跨域 Gate 结果与下一步

日期：2026-08-03
状态：**P0 部分成功；严格 universal-B0 Gate 为 NO-GO；启动唯一允许的 P0.1 数据比例干预。**

本报告执行 `MOVIE3R_ICLR_PAPER_BLUEPRINT_20260802.md` 和
`V14_ICLR_PROGRESS_GATE_AND_BATCH_CONTROL_20260803.md` 的 P0，而不是把
cross96 的 controlled 结果与 old B0 的真实多人结果拼成一行。

## 1. 要回答的科学问题

已有两个不能直接互换的 checkpoint：

- `old B0`：真实多人 EgoHumans 固定世界人体较强，但 four-source controlled
  camera tail 较弱；
- `cross96`：four-source controlled camera 较强，但真实多人 EgoHumans 的
  fixed-world root/layout 较差。

P0 的唯一问题是：**加入真正的、全画幅、多人相机切换监督后，能否训练一个单一
checkpoint，同时保留 controlled camera tail，并且不牺牲真实多人 fixed-human？**

这不是尝试让 V9 隐式地一步完成精对齐。部署语义保持不变：V9 correction 只在
read-only shadow 中产生 coarse camera proposal；clean raw reset state 是唯一提交
的 recurrent state；显式 `B0 = C_shadow @ inverse(C_raw)` 只改变新的 shot gauge。

## 2. 训练干预和严格不变量

候选：`v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth`
SHA256：`de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265`

它从与 cross96 相同的 formal V9 checkpoint 初始化，保留完全相同的架构、loss、
first-post-only routing、6 epochs 和 final-checkpoint-only 选择。唯一改变是每个
epoch 的 480 个事件按下列单一 shuffled `CatDataset` 混合：

| 来源 | 事件/epoch | 监督 |
|---|---:|---|
| AvatarReX | 96 | 原有 camera + human |
| THuman | 96 | 原有 camera + human |
| MVHuman100 | 96 | 原有 camera + human |
| MVHuman200 | 96 | 原有 camera + human |
| MultiHuman Real-World-Capture `three` | 96 | **官方 per-frame camera/intrinsic only** |

MultiHuman 输入是原始同步六机 `2048×2048` 全画面 `A(t-1), A(t), B(t)`，shot labels
为 `0,0,1`。没有把 person crop 当作全局相机训练数据，也没有伪造不兼容的
SMPL-X 参数 target。训练只用 `three` 的 192 个事件；开发集是 36 个 pair-disjoint、
timestamp-disjoint opposite-camera events。EgoHumans 五链、`dance`、`box` 未进入训练。

无人体 target 的 batch 已通过 forward/backward smoke（loss `0.3038`、峰值约
`6.9 GB`）；为此仅修复了空 `smpl_mask` 时仍构造 image/K Multi-HMR inputs 的
数据通路。

所有评测均验证：

```text
runtime B0:           only pre RGB + first post RGB
future post frames:   0
GT camera/mesh:       runtime never reads; evaluator only
shadow state/mesh:    never committed
BRTC:                 frozen, camera bit-exact, unmatched/rejected person exact B0
external models:      none (no DA3 / VGGT / SLAM / ReID)
```

## 3. 选择阶段的结果

### 3.1 Frozen four-source 180（controlled single person）

| Checkpoint | Composite ↓ | P90 ↓ | P95 ↓ | Catastrophic ↓ | Human head ↓ |
|---|---:|---:|---:|---:|---:|
| old B0 | 2.2875 | 4.9074 | 5.6680 | 107 | 1.4555 |
| cross96 | 1.7333 | 3.9670 | 4.7186 | 86 | **1.2288** |
| **P0 (20% real-multi camera)** | **1.5838** | **3.3656** | **3.9362** | **79** | 1.3818 |

P0 比 cross96 进一步改善 camera mean、P90、P95 和 catastrophic count；它没有把
cross96 的 camera-tail gain 换回 old B0 的较差 tail。P0 的 camera-only B0 路径的
human-head mean 不是更优，因此该项不能被解释为 human fine alignment 的结果。

### 3.2 MultiHuman `three` pair/timestamp-disjoint development（36 real-multi events）

| Checkpoint | Camera composite ↓ | Root ↓ | Joint ↓ | Vertex ↓ | Pair vector ↓ | Cat. ↓ |
|---|---:|---:|---:|---:|---:|---:|
| old B0 | .3341 | .3435 | .3734 | **.3514** | .1218 | 0 |
| cross96 | .3920 | .4007 | .4213 | .4146 | **.1186** | 0 |
| **P0** | **.1462** | **.3339** | **.3638** | .3533 | .1208 | 0 |

单位为 m，除 catastrophic。每个 candidate 都有 100 个 GT-assigned person observations；
assignment/GT 仅在 runtime forward 后使用。

这里 P0 的 camera composite 相对 old B0 降低 56%，root/joint 也小幅改善。vertex
仅比 old B0 高 `1.9 mm`，pair vector 仍在同一量级。更重要的是，三个 checkpoint
的 **raw-reset** 的每个 case、每个报告量均 bit-exact（最大差 `0.0`）；差异只能
来自 shadow-derived B0，不能归因于 raw Human3R、检测 coverage 或 evaluator 分支。
对每个 checkpoint，`B0 @ C_raw` 与 `C_shadow` 的 P95 matrix max-abs 也均小于
`1.2e-7`。

**选择结论。** P0 是进入一次性外部确认的唯一候选：它同时通过 controlled camera
tail、real-multi development camera 和基本 fixed-human safety；cross96、old B0 仅是
对照，不再选择 epoch、threshold、selector 或 BRTC 参数。

## 4. 冻结 EgoHumans 5-chain confirmation

确认清单为 `v14_cross96_brtc_egohumans_confirmation_20260803.json`：5 条互不重叠
的三-shot 链，10 个 boundary、75 帧；每个 cut 的相邻 shot 复用同步物理 timestamp。
这是透明的 local Multi-THuMBS-style evaluator，**不是官方 Multi-THuMBS 对榜**。

| Method | W ↓ | WA ↓ | Fixed root ↓ | Fixed joint ↓ | Fixed vertex ↓ | Pair vector ↓ | ATE ↓ | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old B0 | 407.2 | 262.9 | **463.6** | 450.2 | 441.2 | **432.3** | **.145** | 87.1% |
| old B0 + BRTC | 397.0 | 268.1 | **456.2** | 444.8 | 435.0 | **430.8** | **.145** | 87.1% |
| cross96 B0 | 443.7 | 258.2 | 649.1 | 622.1 | 614.7 | 479.2 | .198 | 87.1% |
| cross96 + BRTC | 434.4 | 257.1 | 640.5 | 616.0 | 608.1 | 456.8 | .198 | 87.1% |
| **P0 B0** | 387.4 | 248.7 | 506.8 | 484.8 | 472.6 | 458.3 | .162 | 87.1% |
| **P0 B0 + frozen BRTC** | **329.3** | **222.6** | 459.8 | **441.0** | **427.8** | 483.0 | .162 | 87.1% |

除 ATE 外单位为 mm。P0+BRTC 的 W 相对其 B0 降 `58.1 mm`，five-chain paired
bootstrap 95% CI 为 `[-137.2, -0.33] mm`；WA/root/joint CI 仍跨零。BRTC 相机
max change 为 `0.0`，23 个 anonymous matches 中接受 18 个，unmatched change 为
`0.0`；first-post corrected-person root harm >5 cm 为 `1/25 = 4.0%`。

这确认了两件正面的事情：P0 是目前 W/WA 最好的**同一 checkpoint** candidate，且
冻结的人体 verifier 能在不改相机的条件下进一步降低 W。它也暴露了关键反例：

```text
P0 B0 fixed root:         506.8 mm  > old B0 463.6 mm
P0+BRTC fixed root:       459.8 mm  > old+BRTC 456.2 mm
P0+BRTC pair vector:      483.0 mm  > old+BRTC 430.8 mm
```

因此虽然 P0 在 camera/W/WA 上显著强，仍不能声称它在所有真实多人 fixed-human
量上 dominate old B0。这个报告不将 P0 的 W/WA 与 old B0 的 root/pair 拼成一条
方法 row。

## 5. P0 Gate 决策

| 条件 | 判定 | 证据 |
|---|---|---|
| 单一 checkpoint | 通过 | P0 从 controlled、MultiHuman dev 到 Ego confirmation 均为同一 SHA256 |
| Controlled camera tail | 通过 | P95 `3.936 < 4.719`，cat `79 < 86`（vs cross96） |
| Real-multi dev camera / basic human | 通过 | composite `.146 < .334`，root/joint 不劣于 old B0 |
| no-cut/raw parity 与 shadow replay | 通过 | raw-reset max difference `0.0`；B0/shadow `~1e-7` |
| External W/WA | 通过（有限） | P0+BRTC `329.3/222.6 mm` 为三者最低 |
| External fixed root/layout no-regression | **失败** | B0 root 与 final pair vector 均劣于 old B0 |
| Official benchmark / pristine capture | 缺失 | 当前是一个 EgoHumans capture 的 local evaluator |

严格决策：

```text
NO_GO_P0_AS_UNIVERSAL_B0
KEEP_P0_AS_BEST_CURRENT_UNIFIED_CAMERA/W/WA_CANDIDATE
```

这不是对“state–gauge decoupling”故事的否定，而是清楚地定位剩余误差：camera gauge
改善和 camera-relative human scale/root/layout 的改善并非同一自由度。BRTC 是安全的
camera-invariant local verifier，但目前无法保证 pair layout，因此不应通过继续调
BRTC 来掩盖 B0 的跨域 tradeoff。

## 6. 唯一的下一实验：P0.1 real-multi ratio intervention

P0.1 预先固定于 `config/train_v14_1_cut_first_cross_source_multihuman_p0_r33.yaml`。
它只改变训练事件比例，并保持总更新数 2880 不变：

```text
P0:   [96, 96, 96, 96,  96] = controlled 80% / real-multi 20%
P0.1: [80, 80, 80, 80, 160] = controlled 67% / real-multi 33%
```

其可证伪假设是：若 P0 external root/layout tradeoff 来自真实多人全画幅相机的覆盖
不足，提高该覆盖可降低 fixed-root/pair residual；若 controlled tail 明显反弹，则
说明单纯 camera-only ratio 不能同时解决 gauge 与结构，应记录为第二个 No-Go，而
不是转向 BRTC damping、domain selector、latent blend 或 GT identity。

固定评估顺序：

1. four-source frozen180；
2. MultiHuman pair/timestamp-disjoint 36-event development；
3. 只有前两者同时超过 P0/old 的预注册门槛，才运行原样的 EgoHumans confirmation。

P0.1 之后若仍不存在 fixed-human non-regression 的统一 B0，则停止 ratio sweep，论文
必须收窄为 domain-aware failure analysis，或重新定义一个能直接观测 human
root/layout residual、但不重写 camera 的可训练 typed correction；不能把结果包装为
通用 end-to-end system。

## 7. 可复现产物

```text
config/manifests/v14_multihuman_camera_supervision_20260803.json
src/dust3r/datasets/multihuman_camera.py
config/train_v14_1_cut_first_cross_source_multihuman_p0.yaml
config/train_v14_1_cut_first_cross_source_multihuman_p0_r33.yaml
versions/v14/cut_first_cross_source/evaluate_multihuman_camera_dev.py

output/v14_cut_first_cross_source/eval_multihuman_p0_180/
output/v14_cut_first_cross_source/eval_multihuman_camera_dev/{old_b0,cross96,p0_e6}/
output/v14/fine_alignment_research/multihuman_p0_b0_egohumans_confirmation/
output/v14/fine_alignment_research/multihuman_p0_brtc_egohumans_confirmation/
```

所有新的 cache、checkpoint、report 都位于 Movie3R workspace 下；没有向 `/root`、
系统根目录或 `/tmp` 写入本实验文件。
