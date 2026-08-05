# cross96 → B0 → BRTC-LC → Kabsch：同 checkpoint 闭环兼容性评测

日期：2026-08-02

状态：**[Established / Phase-1 single-person Go]**

对应 ICLR 蓝图：`MOVIE3R_ICLR_PAPER_BLUEPRINT_20260802.md` 的 Phase 1。

原始结果：

```text
output/v14_cut_first_cross_source/eval_cross96_brtc_kabsch_frozen180/report.json
output/v14_cut_first_cross_source/eval_cross96_brtc_kabsch_frozen180/report.md
```

评测器：

```text
versions/v14/cut_first_cross_source/evaluate_cross96_brtc_kabsch.py
```

---

## 1. 这个实验关闭了什么证据缺口

此前有两组不能直接拼接的结果：

```text
cross96 checkpoint -> 强 camera B0
旧 B0 checkpoint -> BRTC-LC / person-local Kabsch 人体收益
```

因此此前不能声称“cross96 的 camera 和 BRTC/Kabsch 的人体收益组成一个已验证的端到端方法”。本实验用**同一个 cross96 checkpoint**重新执行全部在线分支：

```text
three first-post-cut inputs
-> read-only shadow rollout (V9 correction enabled)
-> clean raw-reset rollout (correction disabled)
-> B0 = C_shadow @ inverse(C_raw)
-> B0(raw camera + raw person geometry)
-> frozen BRTC-LC v1
-> frozen qualified TORSO4 person-local Kabsch candidate
-> GT evaluation only
```

这里的关键约束是：

- BRTC-LC 与 Kabsch 的所有参数均未改动；
- shadow state、shadow mesh 均不提交；
- BRTC/Kabsch 仅编辑复制后的 post-cut 人体 root/joints/vertices；
- 每个 case 都断言 BRTC 与 Kabsch camera 等于 B0 camera；
- rejected/unmatched case exact B0 fallback；
- 不使用 future post-cut frame、GT、身份标签或外部预训练模型做 runtime action。

## 2. 权重、协议与可追溯性

| 项目 | 值 |
|---|---|
| coarse checkpoint | `output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth` |
| checkpoint SHA256 | `05274f7b4841f6ebc73f2f5bdb419d63d272396724db886b6e10987d7210a144` |
| frozen records | `output/v10_candidate_selection/oracle_gt_4source/selected_records.jsonl` |
| completed/failed | `180/180` / `0` |
| BRTC policy | `BRTC_LC_V1_20260801`，原始参数不变 |
| orientation policy | frozen TORSO4 Kabsch qualified candidate；不是当前默认 runtime |
| camera parity | 与既有 `eval_cross96_180` 的 180 个 B0 camera scalar 均逐项 `0.0` 差异 |

### 2.1 必须保留的 single-person 限制

cross96 训练及本评测的 loader 都使用 `max_humans=1`。所以 runtime 只在 pre/post 都恰有一个预测人体时使用匿名 singleton match `(0, 0)`；否则不行动。

这是一条合法的 single-person compatibility protocol，但不是多人自动身份实验：

- 不报告 ID accuracy；
- 不报告 multi-person layout-vector；
- 不可用作对 Multi-THuMBS 多人官方结果的比较；
- BRTC 的 group/layout-consensus 在该协议中退化为单人 ray triangulation。

## 3. 主结果

单位均为 m；`camera composite = translation + 0.02 * rotation_deg`。

| Method | Camera comp. | Catastrophic | Root | Joint | Vertex | Head |
|---|---:|---:|---:|---:|---:|---:|
| raw reset | 5.5173 | 180 | 1.3132 | 1.4005 | 1.3719 | 1.1046 |
| shadow diagnostic（不可提交） | 1.7333 | 86 | 0.7054 | 0.7530 | 0.7379 | 0.4083 |
| clean B0 runtime | 1.7333 | 86 | 1.3678 | 1.4049 | 1.3933 | 1.2288 |
| + frozen BRTC-LC v1 | 1.7333 | 86 | 0.6550 | 0.7069 | 0.6906 | 0.4672 |
| + TORSO4 Kabsch candidate | 1.7333 | 86 | 0.6550 | 0.6957 | 0.6824 | 0.4651 |

从 clean B0 runtime 到 BRTC-LC：

```text
root    1.3678 -> 0.6550 m   (52.11% reduction)
joint   1.4049 -> 0.7069 m   (49.68% reduction)
vertex  1.3933 -> 0.6906 m   (50.43% reduction)
head    1.2288 -> 0.4672 m   (61.97% reduction)
```

运行/安全分布：

```text
BRTC accepted cases              157 / 180 = 87.2%
BRTC root improved                81.1%
BRTC root harm > 5 cm              1.1% (2 / 180)
BRTC root P95 delta               +4.8 mm
Kabsch joint improved              82.2%
Kabsch vertex improved             82.2%
Kabsch joint harm > 5 cm           1.1%
Kabsch vertex harm > 5 cm          1.7%
```

Kabsch 只旋转 root-centred body geometry，故 root 与 BRTC 完全相同；它相对 BRTC 的增量为：

```text
joint   0.7069 -> 0.6957 m   (additional 1.59%)
vertex  0.6906 -> 0.6824 m   (additional 1.19%)
```

## 4. 跨来源结果

下表为 `B0 root -> BRTC root`（括号为相对下降）；四个来源均为正，且 root >5 cm harm 均不超过 4.2%。

| Source | N | B0 root | BRTC root | Reduction | BRTC accepted |
|---|---:|---:|---:|---:|---:|
| AvatarReX | 48 | 1.1302 | 0.4277 | 62.15% | 42 |
| THuman | 48 | 0.3239 | 0.2015 | 37.78% | 44 |
| MVHuman100 | 48 | 2.3602 | 1.0686 | 54.72% | 42 |
| MVHuman200 | 36 | 1.7532 | 1.0111 | 42.33% | 29 |

TOROS4 Kabsch 对 joint/vertex 同样在四个来源均不退化；不过它的增量相对 BRTC 较小，且历史上只以 proxy tolerance 得到 qualified status，不能过度包装为完全独立的主贡献。

## 5. 蓝图 Go/No-Go 判定

| Phase-1 条件 | 结果 | 判定 |
|---|---|---|
| B0 camera 保持 cross96 数值 | 与旧 cross96 180-case B0 完全一致 | Pass |
| BRTC root gain ≥ 8% | 52.11% | Pass |
| root harm >5 cm ≤ 10% | 1.11% | Pass |
| Kabsch joint/vertex 非退化 | joint/vertex 分别再改善 1.59% / 1.19% | Pass |
| ≥2 source/sequence family 正向 | 4/4 source 正向 | Pass |
| layout-vector gain ≥5% | 此 max_humans=1 protocol 不可计算 | Deferred，不可声称 Pass |

结论：**single-person 同 checkpoint 兼容性明确 Go**。这使得论文可以把 `cross96 B0 + BRTC-LC` 作为一条已验证的端到端单人 first-post-cut pipeline；Kabsch 可作为保守的 qualified refinement ablation/候选项。

但这不是完整 ICLR Go，因为下列根本缺口仍在：

1. camera catastrophic 仍为 `86/180`，人体 refinement 不会、也不应掩盖这个 camera tail；
2. `shadow` human head `0.4083 m` 仍明显优于当前 BRTC/Kabsch `0.4651 m`，说明仍有可利用但未安全提交的 residual；
3. 没有 multi-person layout、automatic ID、variable visibility 与 pristine test；
4. 没有 scene metric、multi-cut、automatic cut、runtime/memory 和 Multi-THuMBS official-equivalent protocol。

## 6. 下一步：不调 BRTC，先做 shadow typed residual 分解

本实验已排除“BRTC 只在旧 B0 上有效”的疑问，因此下一步不应重新扫描 BRTC gate。应在同一 frozen180 输出上分析：

```text
R_shadow = H_shadow - B0(H_raw)
```

优先分析：

1. root residual 的 ray-parallel 与 ray-orthogonal 分量；
2. residual 与 GT correction、BRTC shift 的 signed correlation / cosine / magnitude ratio；
3. shadow direct commit 的 mean、P90/P95、harm 与 source dependence；
4. 哪些 observable 量（BRTC ray gap/parallax/MAD、shadow-B0 magnitude、camera risk）可作为无 GT gate；
5. 只有跨至少两个来源稳定的 residual 才进入 VSP-0（bounded shadow root）或 VSP-1（shadow/BRTC agreement-gated root）。

这会回答下一条论文核心问题：能否把 shadow branch 中显著的人体信息，投影为一个可验证、可 fallback 的 typed commit，而非不安全地提交整份 shadow reconstruction。
