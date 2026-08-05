# Movie3R V14：ICLR 进度 Gate 与 EgoHumans 批量对照

日期：2026-08-03

性质：依据 `MOVIE3R_ICLR_PAPER_BLUEPRINT_20260802.md` 的进度审计、冻结确认结果和后续执行决策。本文不把 local/provisional 指标伪装成 Multi-THuMBS 官方对榜。

## 1. 当前判断

Movie3R 的科学主张仍然成立，但**还不具备 ICLR 完整投稿条件**。

已经成立的骨架是：camera cut 必须把“旧 recurrent state 的读权限”与“新 shot state 的写权限”分离；shadow 只产生 proposal，clean reset state 才能提交；`B0`、person root/layout 与 orientation 是不同类型的误差自由度。

最新外部多人批量确认则排除了一个危险的过度结论：

> cross96 在受控的四来源 single-person first-cut benchmark 上是更强的 coarse B0，但它**不是**在所有人类/相机域上都优于旧 B0 的通用 replacement。

因此当前不能将 `cross96 camera result + old-B0 multi-human result` 拼成同一条完整主方法，也不能在摘要中宣称“cross96 已经普遍提高多人跨镜头重建”。

## 2. 本轮冻结 protocol

### 2.1 新的外部确认清单

`config/manifests/v14_cross96_brtc_egohumans_confirmation_20260803.json` 预先固定了 EgoHumans/EgoBody `001_legoassemble` 的五条三 shot 流：

```text
5 chains × 3 clean-reset shots × 5 RGB frames = 75 observations
5 chains × 2 boundaries = 10 cuts
```

边界两侧是同步 exo camera 的同一物理时间戳，以隔离跨 camera/cut gauge；两侧之后的帧只用于评价传播。所有区间与此前三条小 EgoHumans chain 的时间段不重合。

运行时严格只使用：

```text
pre-shot RGB + first post-cut RGB
-> read-only cross96/old shadow camera
-> clean raw-reset camera
-> B0 = C_shadow @ inverse(C_raw)
-> B0-aligned anonymous association
-> frozen BRTC-LC v1 root/layout correction
```

不使用 GT、future post frame、DA3/VGGT/SLAM/ReID 或 shadow state/mesh。GT 只在所有 runtime action 完成后，用于 detection identity audit 和 metric evaluation。

### 2.2 两个严格对照

| Row | Checkpoint | SHA256 | B0 cache | Clean raw branch |
|---|---|---|---|---|
| `old_b0` | `checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth` | `837924…e47123` | `old_b0_egohumans_confirmation` | checkpoint correction OFF |
| `cross96` | `output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth` | `05274f…0a144` | `cross96_b0_egohumans_confirmation` | checkpoint correction OFF |

两条 raw-reset branch 的 75 帧 camera 数值完全相同（最大绝对差 `0.0`），检测 coverage 也相同（`196/225 = 87.1%`）。所以 B0/BRTC 差异不能归因于原始 Human3R 输出、检测成功帧或 evaluator coverage。

## 3. 本轮主要结果

### 3.1 cross96：B0 是有效 gauge，但 BRTC 只是小幅、安全的辅助项

下表均为透明的 local Multi-THuMBS-style evaluator，而非官方 protocol。

| Method | W | WA | Fixed root | Fixed joint | Fixed vertex | Pair vector | ATE | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw reset | 1155.6 | 388.9 | 1566.5 | 1430.8 | 1370.8 | 2383.3 | 1.788 | 87.1% |
| cross96 B0 | 443.7 | 258.2 | 649.1 | 622.1 | 614.7 | 479.2 | .198 | 87.1% |
| cross96 B0 + frozen BRTC | **434.4** | **257.1** | **640.5** | **616.0** | **608.1** | **456.8** | .198 | 87.1% |

单位除 ATE 外均为 mm。B0 相对 raw reset 显著恢复了整体 gauge；BRTC 相对 B0 的增量是：

```text
W              443.7 -> 434.4 mm  (-9.3)
WA             258.2 -> 257.1 mm  (-1.0)
fixed root     649.1 -> 640.5 mm  (-8.6)
pair vector    479.2 -> 456.8 mm  (-22.4)
```

5-chain paired bootstrap：W 的 BRTC−B0 `95% CI = [-15.45, -2.03] mm`；但 WA、fixed root、fixed joint 的 CI 都跨过零。因此 BRTC 在该域的正确定位是 **camera-invariant、low-tail 的 explicit verifier**，不是已经稳定解决 human residual 的强校正器。

安全和在线 contract 都通过：

```text
camera max change                       0.0 (bit-exact)
matched / accepted BRTC persons         23 / 16
unmatched max geometry change           0.0
root harm > 5 cm, all corrected post    2.3% (3/131)
root harm > 5 cm, first post            0.0% (0/25)
second-cut causal person shift observed yes (6 nonzero inherited shifts)
```

但 association 只有 `21/23 = 91.3%` evaluator-only correctness，且一个 shot 只有 `4/15` person-frame detection。不能把它写成“general automatic multi-person tracking solved”。

### 3.2 最关键的 fair control：旧 B0 优于 cross96 的外部多人结果

| Method | B0 W | BRTC W | B0 WA | BRTC WA | B0 fixed root | BRTC fixed root |
|---|---:|---:|---:|---:|---:|---:|
| old B0 | 407.2 | **397.0** | 262.9 | 268.1 | 463.6 | **456.2** |
| cross96 | 443.7 | 434.4 | **258.2** | **257.1** | 649.1 | 640.5 |

cross96 minus old after BRTC：

```text
W               +37.4 mm (worse)
WA              -11.0 mm (better)
fixed root     +184.3 mm (worse)
fixed joint   +171.2 mm (worse)
ATE             +0.053 m (worse)
pair vector     +26.0 mm (worse)
```

直接由 evaluator-only COLMAP camera pose 计算的 B0 camera error 也一致地偏向旧 B0：

| Scope | Checkpoint | T (m) | R (deg) | Composite |
|---|---|---:|---:|---:|
| first post | old B0 | .472 | 5.26 | .577 |
| first post | cross96 | .718 | 8.00 | .878 |
| all post-shot frames | old B0 | .473 | 5.28 | .579 |
| all post-shot frames | cross96 | .718 | 8.04 | .879 |

这与 frozen180 的受控四来源结论不矛盾：cross96 的训练来源正是 AvatarReX、THuman、MVHuman，而这份 EgoHumans confirmation 是训练外的多人真实 capture。它说明当前 cross-source supervision 的泛化范围仍然不足，尤其不能只以 camera-pair-disjoint four-source result 推出对多人域的普适性。

最终决策：

```text
NO_GO_CROSS96_AS_UNIVERSAL_MULTI_HUMAN_B0
```

cross96 仍保留为 **[Established] controlled four-source single-person coarse proposal**；旧 B0 仍保留为 **[Established] old multi-human coarse baseline**。两者不能被合写成一个统一的投稿 row。

## 4. 按 ICLR 蓝图的进度

| 蓝图项 | 状态 | 证据与解释 |
|---|---|---|
| E0 provenance/schema/split | Partial | checkpoint、manifest、cache 已可追溯；已有新 Ego confirmation，但仍缺 actor/capture-disjoint final split 和统一 scene schema。 |
| E1 cross96→BRTC→Kabsch | Pass（single person） | frozen180 同 checkpoint B0→BRTC root `1.3678→.6550 m`，camera 不变；不等于多人 pass。 |
| E2–E5 typed shadow/VSP | No-Go | shadow root/orientation agreement 与 direct commit 未越过 BRTC safety；不再重扫该家族。 |
| E6 camera safety gate | No-Go | selector、fixed mixture、latent SE(3) residual、token/pointmap Kabsch 均未降低 tail 且保持均值。 |
| E7 automatic identity / variable visibility | Missing | 当前几何 association 有不可观测 full-swap 反例；本轮 `21/23` 也证实缺口。 |
| E8 expanded EgoHumans local comparison | Pass（provisional evidence） | 5 chains/10 cuts/75 frames、old/cross96 same-manifest batch、B0/BRTC table 已完成。 |
| E9 long stream/cut detector/runtime/scene | Missing | 无自动 detector、正式 multi-cut memory/FPS、no-future truncation、scene metrics。 |
| E10 baselines/ablations/statistics | Missing | 无 official Multi-THuMBS adapter、no external runnable baselines、3 seed/CI 仍不充分。 |

## 5. ICLR Go/No-Go

### 已能诚实主张的内容

1. 事务式 read-only shadow / clean reset architecture，no-cut/state-purity contract；
2. cross-source cut supervision 可改善其受控评测域中的 coarse camera proposal；
3. camera-only B0 不能保证 human structure；冻结 camera 的 BRTC 可在多种 setting 中提供受限的 root/layout gain；
4. explicit human correction 必须报告 fallback、coverage、harm、association，而不只报告 mean。

### 当前阻止最小投稿版本的项

1. 没有一个同 checkpoint、同 protocol、跨 controlled 和 real multi-human 都通过的 B0；
2. real multi-human external confirmation 中 BRTC 的人体增益小，WA/root 的 CI 未稳定排除零；
3. automatic WHO 只有 91.3%，variable visibility/full swap 未解决；
4. cross96 camera frozen180 仍有 `86/180` catastrophic，且没有可部署 gate；
5. official-equivalent Multi-THuMBS、scene metrics、automatic cut、runtime/memory、pristine final test 仍缺失。

判定：

```text
ICLR_FULL_SYSTEM = NO_GO (continue research, do not begin result packaging)
```

这不是否定论文故事，而是要求它在“一个统一、可提交的 coarse B0”上闭环后才开始包装。

## 6. 后续唯一优先级

不要继续调 BRTC damping、VSP blend、B0 mixture/selector、pooled latent residual 或裸 token/pointmap Kabsch；这些候选已有独立 No-Go 证据。

### P0：先解决 B0 的跨域冲突，再谈 human fine alignment

研究问题：

> 为什么 `cross96` 在 four-source held-out camera pairs 中优于旧 B0，却在冻结的 EgoHumans multi-human streams 中相机和 fixed-world human 更差？能否得到**单一** B0 checkpoint，而不是按数据域选 checkpoint？

执行顺序：

1. **冻结训练/开发/确认划分。** EgoHumans 本文 5-chain confirmation 已读，永久不得用于 checkpoint、loss、epoch、threshold 或 selector 选择；MultiHuman `three` 仅能作 development，`dance/box` 只能作已读 diagnostic，需再建立未读 final capture/actor split。 
2. **只做可解释的 training-data intervention。** 从 formal V9 同一初始化出发，保持 first-post, read-only shadow, clean reset 和原 loss；在 cross96 four-source cut manifest 之外加入真正 multi-person camera-cut supervision，训练时允许多人的 human/camera target，而不是在确认集后加 domain selector。 
3. **先在冻结的受控和 multi-human development 上选择一个 checkpoint。** selection 必须同时约束 camera P95/catastrophic、B0-only fixed-world root/layout 与 raw no-cut parity；不得用 BRTC 或 GT identity 掩盖 B0 退化。 
4. **一次性在未参与选择的 final capture 上运行 full raw/B0/BRTC。** 要求 single checkpoint；报告 per-domain mean/P95/coverage/harm，而不是只报 pooled mean。 

P0 的 Go 条件不是“Ego 某一列改善”，而是：相对旧 B0，不让 external real-multi camera/fixed-human 退化；同时不丢失 cross96 controlled camera tail gain。若做不到，论文必须把当前系统收窄为 domain-specific analysis，而非通用 Movie3R。

### P1：B0 稳定后才进行 identity

E7 的合规路线是 Human3R 已有的 frozen per-person native/refined token 作为**只回答 WHO**的正交 cue，geometry 只回答 WHERE。先在明确 development split 做 normalized-token association + mutual/margin abstention，再冻结；wrong commit 必须接近零，低置信 post person exact B0。不得把 appearance/token 当作新的 global Boundary 回归器，也不得新引入外部 ReID。

### P2：最后的完整系统证据

P0/P1 成立后，再做 automatic cut、long multi-cut state composition、no-future truncation、scene metrics、runtime/memory、official Multi-THuMBS adapter 和可运行 baseline。届时才开始撰写主表和摘要数字。

## 7. 产物

```text
config/manifests/v14_cross96_brtc_egohumans_confirmation_20260803.json
versions/v14/generate_cross96_b0_egohumans.py
versions/v14/eval_cross96_brtc_egohumans.py
versions/v14/compare_b0_checkpoints_egohumans.py

output/v14/fine_alignment_research/cross96_b0_egohumans_confirmation/
output/v14/fine_alignment_research/cross96_brtc_egohumans_confirmation/report.json
output/v14/fine_alignment_research/old_b0_egohumans_confirmation/
output/v14/fine_alignment_research/old_b0_brtc_egohumans_confirmation/report.json
output/v14/fine_alignment_research/b0_checkpoint_comparison_egohumans/README.md
```

所有新缓存、报告与日志都在 `/data/wangzheng/iJCV-CODE/Movie3R` 下；本轮不依赖根目录或 `/tmp` 的文件。
