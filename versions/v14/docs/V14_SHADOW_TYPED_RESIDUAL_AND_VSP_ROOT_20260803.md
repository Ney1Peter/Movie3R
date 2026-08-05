# V14 Shadow Typed Residual 分解与 VSP Root Agreement 实验

日期：2026-08-03

状态：**[Established negative result / No-Go for current VSP root and shadow orientation]**

对应 ICLR 蓝图：Phase 2 与 Phase 3 的 `VSP-0 / VSP-1`。

## 1. 问题

cross96 的 read-only shadow branch 有很强的人体诊断结果：

```text
shadow head error         0.4083 m
clean B0 head error       1.2288 m
BRTC + Kabsch head        0.4651 m
```

但 shadow 的 state/geometry 不能直接提交。此实验问的是：能否把

```text
H_shadow - B0(H_raw)
```

投影成一种只使用 cut 当下可见预测量、可通过 gate、可 exact fallback 的 typed root/orientation commit？

## 2. 数据与严格约束

### 2.1 Frozen180 分解（只做诊断，不调参数）

```text
output/v14_cut_first_cross_source/eval_cross96_shadow_typed_residual_frozen180/report.json
```

同一 cross96 checkpoint，180/180 完成，无失败。对每个 singleton person 保存：

- shadow、B0、BRTC、Kabsch 相对 B0 的 root vector；
- root ray-parallel/ray-orthogonal 分解；
- 与 GT oracle residual 的 cosine、magnitude、signed correlation；
- root global orientation residual；
- BRTC gate、Kabsch gate 和所有 base metric。

这些字段仅用于评估/选择，绝不进入 runtime action。

### 2.2 新的 VSP dev/confirm split

为了避免在 frozen180 上选择 gate，新建：

```text
config/manifests/v14_vsp_pair_disjoint_20260802/
```

每 source：24 dev + 24 confirm；dev/confirm 互不共用相机 pair，并同时排除：

- cross96 train96 pair；
- frozen10 pair；
- frozen180 pair。

开发集 forward：

```text
output/v14_cut_first_cross_source/eval_vsp_dev_96/report.json
```

## 3. Frozen180：什么 residual 真有信号

### 3.1 Root：有语义信号，但尚未优于 BRTC

| Quantity | Frozen180 result |
|---|---:|
| shadow root error | 0.7054 m |
| B0 root error | 1.3678 m |
| BRTC root error | **0.6550 m** |
| shadow direct root improve rate vs B0 | 92.22% |
| shadow direct root harm >5 cm vs B0 | 2.22% |
| shadow vs oracle root cosine | 0.817 mean / 0.913 median |
| shadow ray-parallel signed corr. with oracle | **0.883** |
| BRTC ray-parallel signed corr. with oracle | 0.795 |
| shadow vs BRTC root cosine | 0.874 mean / 0.993 median（BRTC accepted cases） |

解释：shadow 确实学到了正确的主要 depth/ray direction；这不是 evaluator artifact。但 BRTC 利用冻结相机和五个 joint ray 的显式三角化，最终 root mean 仍低于直接 shadow。因此“shadow 有信息”并不推出“把它加到 BRTC 后会更好”。

### 3.2 Orientation：明确 No-Go

| Quantity | Frozen180 result |
|---|---:|
| shadow minus B0 angle | 0.81° |
| oracle minus B0 angle | 44.47° |
| shadow vs oracle rotation cosine | **−0.046** |
| shadow orientation improvement rate | 47.78% |
| Kabsch vs oracle rotation cosine | 0.682 mean / 0.934 median（accepted） |

shadow orientation 相比 B0 几乎没有有意义的全局旋转 residual，也无法预测 GT 旋转方向。**VSP-2（shadow orientation）不应实施。** 这与 Kabsch 有效不矛盾：Kabsch 是由可解释的 pre/post torso correspondence 给出的局部 SO(3) proposal，而不是 shadow latent 的微小姿态差。

## 4. VSP-1：BRTC/shadow agreement root blend 的严格开发选择

测试的 runtime-only family：

```text
c = c_BRTC + alpha * (c_shadow - c_BRTC)
only if:
  BRTC accepted
  cosine(c_shadow, c_BRTC) >= threshold
  ||c_shadow - c_BRTC|| <= threshold
else:
  exact BRTC fallback
```

扫描脚本：

```text
versions/v14/select_vsp_root_agreement_policy.py
```

开发 grid：8 个 `alpha` × 7 个 cosine threshold × 7 个 disagreement threshold = 392 个候选。

资格条件在读 confirm 前预先固定：

- root mean 相对 BRTC 至少改善 1%；
- root harm >5 cm ≤10%；
- P95 不差；
- 至少 3/4 source nonnegative，至少 2/4 source 改善 ≥1%；
- 至少 8 个实际接受动作。

开发结果：

```text
qualified candidates = 0 / 392
best positive mean gain = 0.008% at only 1% coverage
```

因此没有生成 `FROZEN_VSP_ROOT_POLICY_BEFORE_CONFIRM.json`，也**不打开 confirm**。这不是“没有扫到足够大 grid”后的暂时停滞：在该候选家族中，BRTC/shadow 的差异不能仅由二者的一致性和距离可靠地区分为“互补残差”或“额外噪声”。

## 5. 决策

```text
NO_GO_VSP-0 direct/bounded shadow root as a BRTC replacement
NO_GO_VSP-1 BRTC/shadow root agreement blend
NO_GO_VSP-2 shadow orientation commit
```

保留的正向知识：

1. shadow root 的主要信息是 camera-ray depth correction；
2. BRTC 已把这部分更稳定地转化为可提交 action；
3. shadow orientation 不应再投入；
4. 任何未来 shadow method 必须引入一个**正交、可观测**的 verification source（例如可靠的 scene/ray/appearance evidence），不能继续从 shadow 与同一套人体几何的一致性中找 gate。

## 6. 下一条主线

当前最硬的 ICLR 阻碍重新回到 camera safety：cross96 B0 仍有 `86/180` catastrophic cases。下一步按蓝图 Phase 4 评估 cross96 与旧 coarse B0 的互补性和可观测 disagreement；只在 pair-disjoint dev 上选择 rule，再在未读 confirmation 验证。不能用 raw-reset fallback 伪装“安全”，因为 raw reset 本身没有接回 pre-shot world gauge，camera catastrophic 并不会因此减少。
