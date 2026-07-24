# V13 Phase 4: Precision-First Shot-Invariant Identity Feasibility

## 1. 结论摘要

本阶段完成了冻结 appearance cue、precision-first identity gate、MultiHuman 独立序列
验证，以及 EgoHumans 多 cut / `3 -> 1 -> 3` 审计。

最终路线判定：**Phase 4A 未通过可部署准入条件，暂不进入 Phase 4B adapter 训练。**

需要同时保留两个不矛盾的结论：

1. V13 Phase 2 的 GT-ID Uniform Multi-Human Consensus 仍然有效；
2. 当前 DINOv2 appearance + beta + local-pose gate 只能安全确认很少一部分 cut，不能把
   GT-ID 收益稳定转化为完整可部署系统。

开发集 `three` 上，冻结 gate 的 identity 结果为：

```text
accepted precision        = 100%
wrong accepted            = 0
accepted coverage         = 14.37%
multi activation coverage = 7.62%
```

但完整端到端结果为：

```text
single highest confidence composite = 0.814
GT-ID uniform multi composite        = 0.664
precision-first composite            = 3.882
precision-first catastrophic rate    = 55.6%
```

这个失败不是 accepted identity 出错，而是 `222/315` 个 cut 没有 accepted identity，
进入 identity-free Fixed Explicit fallback。该 fallback 在 wide-view cut 上本身不可靠。

只看真正启用多人的 24 个 cut：

```text
single composite          = 0.654
GT-ID multi composite     = 0.577
precision-first composite = 0.613
catastrophic              = 0
GT-ID gain retention      = 53.1%
```

因此当前模块是一个低覆盖率的安全确认器，而不是完整 identity bridge。

---

## 2. 冻结范围

本阶段严格复用 V13 Phase 2 的几何路径：

```text
Frozen Human3R
-> pre-decode Hard Reset
-> Fixed Explicit
-> per-human V16 torso residual
-> 20 degree bound
-> one (R_i, t_i) per accepted identity
-> equal-weight SO(3) mean
-> equal-weight translation mean
-> ONE shared Boundary
```

固定 fallback：

```text
N >= 2 accepted identities: Uniform Multi-Human Consensus
N == 1:                    single-human Fixed Explicit + V16
N == 0:                    identity-free Fixed Explicit
```

本阶段没有启用 DA3、VGGT、V11.4 shared scale、Keypoint R-CNN、continuity、scene
refinement、learned fusion、identity-to-SE(3) 或 per-person world transform。

身份模块只回答 WHO，几何仍只回答 WHERE。

---

## 3. Appearance 实现

### 3.1 编码器

使用官方冻结 DINOv2 ViT-S/14：

| 项目 | 值 |
|---|---|
| 模型 | `dinov2_vits14` |
| 参数量 | 22,056,576 |
| 输入 | `224 x 224` letterbox crop |
| 输出 | normalized CLS + normalized mean patch token |
| 最终维度 | 768 |
| 训练 | 无，全部冻结 |
| checkpoint SHA-256 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

DINOv2 是通用冻结视觉表示，不是专门训练的 person Re-ID 模型。本实验回答的是“一个轻量
冻结 appearance cue 是否足够”，不能等价为外部 Re-ID 上界。

### 3.2 可部署 crop

MultiHuman：

```text
full RGB
-> Human3R predicted SMPL-X
-> project predicted vertices
-> predicted full-body bbox
-> 8% padding
-> DINOv2 crop
```

EgoHumans compact cache 原先没有 bbox。本阶段使用 Human3R 预测的 `smpl_rotmat`、
`smpl_shape`、`smpl_transl`、冻结 SMPL-X layer 和 Human3R pseudo intrinsics 重建预测框。

EgoHumans 输入保持完整鱼眼画面，只做：

```text
3840 x 2160 -> 512 x 288
```

没有在 Human3R 输入前裁剪。人物 crop 只用于后续 appearance encoder。GT bbox 从未进入
crop、feature、matching、gate 或 memory。

### 3.3 Feature fusion

所有 cue 先逐行归一化，再等权拼接，避免 768/1024 维 token 数值上淹没 10 维 beta：

```text
appearance
appearance + beta
appearance + local pose
appearance + beta + local pose
appearance + Multi-HMR token
```

语义职责：

- appearance：衣服、纹理和人体外观；
- beta：预测体型兼容性；
- local pose：短时动作兼容性，不作为长期身份声明；
- native token：辅助 cue，不预测 Boundary。

---

## 4. Precision-first gate

### 4.1 搜索空间

先比较 11 类 feature：`H'`、CUT3R head token、Multi-HMR head token、fused prompt、
beta、local pose、appearance 以及四种简单 appearance fusion。

统一比较：

```text
3 prototypes: last / five-frame mean / five-frame medoid
3 distances:  raw L2 / normalized L2 / cosine
2 matchers:   Hungarian / Sinkhorn
```

总计 `11 x 3 x 3 x 2 = 198` 个 feature probes。

对 development feature 再扫描 1,920 个 multi-condition gate。排序目标固定为：

1. wrong accepted 为 0；
2. accepted precision 最大；
3. multi activation coverage 最大；
4. accepted coverage 最大；
5. IDF1 最大。

Boundary 误差没有参与 threshold 选择。

### 4.2 冻结 gate

只在 `MultiHuman three` 上选择：

```json
{
  "feature": "appearance_beta_pose",
  "prototype": "mean",
  "distance": "cosine",
  "max_primary_distance": 0.09979426095336005,
  "min_primary_margin": 0.02420205939635007,
  "min_vote_fraction": 0.6,
  "max_beta_distance": 0.6189280456390862,
  "max_pose_distance": 0.016089346204818167,
  "min_valid_observations": 3,
  "require_mutual": true
}
```

一个 match 必须同时通过 mutual nearest、absolute distance、row/column margin、five-frame
vote、beta compatibility、pose compatibility、至少 3 个有效 appearance observations，且
post appearance crop 有效。任一条件失败即进入 dustbin。

---

## 5. Memory 和 commit

外部 `CausalIdentityMemory` 与 Human3R scene/camera recurrent state 分离。

```text
Match -> Align -> Verify -> Commit
```

Boundary 确定前不更新长期 prototype。正常 no-cut 帧使用 Human3R native track ID 映射到
external ID；cut 时 scene/camera state reset，但 external identity tracklet 保留。

本阶段补充修正：单人物 observation 会连同 `appearance_valid` 一起写入 identity memory。
无效 crop 不再被误计为有效 prototype observation。

EgoHumans 使用 `prototype window = 5`、`tracklet TTL = 8 stream frames`。

---

## 6. 数据和协议

| 数据 | 角色 | Cuts/streams | 人数 | 特点 |
|---|---|---:|---:|---|
| MultiHuman `three` | development | 315 cuts | 3 | 6 cameras，k=0/1/2/4/8 |
| MultiHuman `dance` | frozen evaluation | 36 cuts | 2 | 动作、交叉、遮挡 |
| MultiHuman `box` | frozen evaluation | 36 cuts | 2 | 第二个独立两人序列 |
| EgoHumans `001_legoassemble` | cross-data stress | 3 streams / 6 cuts | 1-3 | 鱼眼、人数变化、`3->1->3` |

MultiHuman 每个 case 使用 5 个 pre-cut frames 和 1 个 fresh post-cut frame。

Appearance crop 有效率：

| Sequence | Valid | Invalid |
|---|---:|---:|
| three | 5,214 | 1 |
| dance | 381 | 0 |
| box | 424 | 3 |
| EgoHumans | 106 | 15 |

---

## 7. Feature probe 结果

以下是每类 feature 在各 sequence 的最佳 IDF1；`dance/box` 只用于冻结规则后的分析，不用于
重新选择 threshold。

| Feature | three | dance | box |
|---|---:|---:|---:|
| local pose | **0.934** | **0.967** | **1.000** |
| beta | 0.877 | 0.951 | 0.881 |
| appearance + beta + pose | 0.798 | 0.951 | 0.910 |
| appearance + beta | 0.756 | 0.951 | 0.887 |
| Multi-HMR token | 0.657 | 0.820 | 0.761 |
| appearance + Multi-HMR | 0.642 | 0.820 | 0.761 |
| refined H' | 0.488 | 0.656 | 0.742 |
| appearance only | 0.487 | 0.694 | 0.672 |
| fused prompt | 0.423 | 0.754 | 0.702 |
| CUT3R head token | 0.401 | 0.918 | 0.812 |

结论：appearance only 在 `three` 上没有明显优于 refined `H'`，且低于 Multi-HMR token。
appearance + beta + pose 比所有单一 native token 更好，但低于 local pose 和 beta。local pose
仍是短时间协议中最强 cue，但它主要表示动作连续性。

---

## 8. Precision gate 身份结果

| Sequence | Accepted / matchable | Precision | Wrong | Accepted coverage | Multi coverage |
|---|---:|---:|---:|---:|---:|
| three | 120 / 835 | 1.000 | 0 | 14.37% | 7.62% |
| dance | 8 / 61 | 1.000 | 0 | 13.11% | 2.78% |
| box | 18 / 67 | 1.000 | 0 | 26.87% | 5.56% |
| EgoHumans | 0 / 16 | N/A | 0 | 0% | 0% |

`three` 的 temporal offset 分组：

| k | Cuts | Accepted | Wrong | Multi coverage |
|---:|---:|---:|---:|---:|
| 0 | 63 | 27 | 0 | 9.52% |
| 1 | 63 | 21 | 0 | 7.94% |
| 2 | 63 | 22 | 0 | 6.35% |
| 4 | 63 | 25 | 0 | 6.35% |
| 8 | 63 | 25 | 0 | 7.94% |

同步 cut 也没有产生足够高的 coverage；motion 增加不是唯一瓶颈。

---

## 9. 端到端 Boundary 结果

### 9.1 MultiHuman `three`

| Method | T | R deg | Composite | P90 | Catastrophic |
|---|---:|---:|---:|---:|---:|
| highest-confidence single | 0.616 | 9.90 | 0.814 | 1.314 | 0% |
| GT-ID uniform multi | 0.522 | 7.07 | 0.664 | 0.989 | 0% |
| automatic unfiltered | 1.102 | 41.90 | 1.940 | 5.808 | 25.7% |
| precision-first | 2.286 | 79.76 | 3.882 | 7.580 | 55.6% |
| no-identity Fixed | 2.833 | 97.99 | 4.793 | 7.580 | 79.4% |

### 9.2 Frozen holdouts

| Sequence | Single | GT-ID | Unfiltered auto | Precision-first | Precision catastrophic |
|---|---:|---:|---:|---:|---:|
| dance | 0.802 | 0.758 | 1.059 | 3.359 | 52.8% |
| box | 0.720 | 0.614 | 1.597 | 2.930 | 41.7% |

### 9.3 只看实际启用 multi 的 cut

| Sequence | Cuts | Single | GT-ID | Precision-first | Retention | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| three | 24 | 0.654 | 0.577 | 0.613 | 53.1% | 0% |
| dance | 1 | 0.745 | 0.485 | 0.485 | 100% | 0% |
| box | 2 | 0.496 | 0.419 | 0.419 | 100% | 0% |

当 gate 真正接受至少两个人且 identity 全正确时，冻结多人几何仍然有效。完整系统失败来自
coverage 和 fallback，而不是 Phase 2 fusion 被推翻。

### 9.4 Risk-coverage

在 `three` 上，零 wrong-accept 工作点只能达到 7.62% multi coverage。放宽 gate 后可提高
coverage，但会重新引入错误身份。部分多条件网格点：

| Multi coverage | Accepted precision | Wrong accepted | Composite | Catastrophic |
|---:|---:|---:|---:|---:|
| 7.6% | 100.0% | 0 | 3.882 | 55.6% |
| 19.7% | 97.0% | 7 | 3.205 | 42.9% |
| 25.1% | 97.8% | 6 | 2.814 | 37.1% |
| 32.7% | 96.8% | 11 | 2.486 | 31.1% |
| 48.3% | 88.4% | 52 | 2.470 | 30.5% |

这些点不是单一 threshold 的严格嵌套曲线。没有找到同时满足高 precision、非低 coverage、
无新增 catastrophic 和不弱于 single baseline 的工作点。

---

## 10. 负面对照

`three` identity control：

| Control | IDF1 | False accepted |
|---|---:|---:|
| automatic unfiltered appearance+beta+pose | 0.798 | 173 |
| wrong-person | 0.094 | 763 |
| shuffled memory | 0.339 | 558 |
| zero memory / array-order tie | 0.785 | 184 |

wrong-person 和 shuffled memory 显著更差，并导致 80.6% 和 62.9% catastrophic。

但 zero-memory 与真实 feature 只差约 0.013 IDF1。该协议中的 Human3R detection order 本身
具有较强偶然稳定性，appearance fusion 相比 array-order control 的独立身份贡献很小。不能把
0.798 IDF1 全部归因于 appearance。

---

## 11. EgoHumans 多 cut 结果

三条真实流：

```text
cam01 296-300 -> cam06 300-304 -> cam07 304-308
cam02 176-180 -> cam05 180-184 -> cam08 184-188
cam03 416-420 -> cam04 420-424 -> cam01 424-428
```

第三条 Human3R detection count 为 `3 -> 1 -> 3`。

冻结规则结果：

```text
accepted / matchable           = 0 / 16
wrong accepted                 = 0
multi activation              = 0 / 6 cuts
inactive identities recovered = 0 / 2
fallback                      = Fixed 6 / 6
```

16 个 Hungarian proposals 的 gate failure 计数：

| 条件 | 失败数 |
|---|---:|
| primary distance | 16 |
| beta distance | 12 |
| pose distance | 10 |
| primary margin | 5 |
| mutual nearest | 4 |
| vote fraction | 3 |
| target crop invalid | 3 |
| insufficient valid observations | 2 |

所有 proposal 的 primary distance 都超过 `three` threshold，说明出现明显跨数据域偏移。

预测 bbox overlay 还显示：鱼眼视角中部分 Human3R SMPL-X 投影框贴边、过宽、只覆盖部分人体，
或包含大量相同背景。EgoHumans 失败既包含 appearance representation 域偏移，也包含可部署
crop 质量问题，不能只归因于 DINOv2。

由于 compact EgoHumans cache 没有冻结 Phase 2 MultiHuman 的完整 V16 geometry evaluator，
本阶段只将其报告为 WHO / TTL / fallback stress test，不声称 EgoHumans Boundary benchmark。

---

## 12. 成功标准审计

| 标准 | 结果 | 判定 |
|---|---|---|
| appearance 或简单 fusion 优于 native token | fusion 优于单一 native token；appearance only 不稳定 | 部分通过 |
| accepted precision 接近 100% | MultiHuman 为 100% | 通过 |
| wrong accept 接近 0 | 冻结 gate 为 0 | 通过 |
| three/dance/box 不弱于 single | 三者完整端到端均明显更差 | 失败 |
| EgoHumans `3->1->3` 安全恢复 | 0/2 inactive identities recovered | 失败 |
| 保留至少 70% GT-ID 收益 | three activated cuts 53.1%；全局为负 | 失败 |
| 无新增 catastrophic | fallback 产生 41.7%-55.6% | 失败 |
| 保持非零多人 coverage | MultiHuman 很低，EgoHumans 为 0 | 失败 |
| wrong/shuffle/zero 明显更差 | wrong/shuffle 是；zero 与真实 feature 接近 | 失败 |

综合判定：**Phase 4A FAIL。**

---

## 13. 最终问题回答

### 1. 冻结 appearance cue 是否明显优于 Human3R 原生 token？

Appearance only 不是。`three` 上 appearance-only IDF1 为 0.487，与 refined `H'` 的 0.488
基本相同，并低于 Multi-HMR token 的 0.657。appearance + beta + pose 达到 0.798，说明简单
fusion 有帮助，但独立 appearance 贡献不够强。

### 2. Appearance、beta 和 local pose 分别承担什么角色？

Appearance 提供衣服、纹理和视觉上下文，但受 crop 和视角变化影响；beta 提供体型兼容性；
local pose 是短时 motion compatibility，不是长期 identity。

### 3. 哪种 prototype 最稳定？

冻结 fusion 选择 five-frame mean；local-pose 最佳探针使用 last。不存在跨所有 feature 和
sequence 的统一最优 prototype。

### 4. Multi-condition gate 能否把 wrong-accept 降到接近零？

能，但代价过大。MultiHuman 上 wrong accepted 为 0，EgoHumans 则拒绝全部 match。

### 5. 多人模式应该在多大 coverage 下安全启用？

当前只观察到 `three/dance/box = 7.6%/2.8%/5.6%` 的零错误 multi coverage。这个 coverage
不具备部署价值，不能作为推荐工作点。

### 6. Automatic ID 能保留多少 GT-ID 多人收益？

`three` 的 24 个 activated multi cuts 保留 53.1%；`dance/box` 的 1/2 个 activated cuts
保留 100%，但样本太少。完整流因 fallback 恶化而为负收益，未达到 70% 门槛。

### 7. `three` 规则能否泛化到 `dance`、`box` 和 EgoHumans？

Precision 能泛化到 `dance/box`，coverage 不能；EgoHumans 上 coverage 完全坍缩。因此不能
称为跨数据泛化。

### 8. `3 -> 1 -> 3` 中 inactive identity 能否安全恢复？

不能。两个重新出现的 inactive identities 均未恢复旧 external ID，全部进入新 track / Fixed
fallback 路径。

### 9. 是否需要训练轻量 shot-invariant adapter？

未来可能需要，但现在不应直接开始。当前 appearance-only 可分性弱、crop 质量未解决，且现有
数据不足以构造严格 subject/capture/camera-pair-disjoint 训练和验证。直接训练很可能只记住
`three` 的人物、衣服和相机。

### 10. 是否保留完整最终贡献？

不能把 `Precision-First Cross-Shot Identity Bridge` 作为已成立贡献。当前可以保留：

```text
GT-ID Uniform Multi-Human Geometric Consensus Oracle
+ precision-first identity feasibility / negative result
```

默认部署仍应使用 single-human Movie3R。Automatic multi-human 保持关闭。

---

## 14. 下一步建议

在训练 adapter 前，先满足两个前置条件：

1. 改善可部署 person crop：优先评估 Human3R mask、可部署 person detector bbox，或对鱼眼更
   稳定的 bbox 生成；不得使用 GT bbox；
2. 加入真正冻结的 person Re-ID encoder 作为单独计时的 reference，区分“DINO 表示不足”与
   “任何 appearance 都不可用”。

只有 appearance / Re-ID reference 在 capture-disjoint 数据上出现清晰 same/different separation，
并能在非零 coverage 下保持接近零 wrong accepted，才进入小型 shot-invariant adapter。

不得通过修改 Uniform Multi-Human Consensus、预测 SE(3) 或读取未来 shot 来补偿 identity 失败。

---

## 15. 实现和产物

核心实现：

- `versions/v13/appearance_identity.py`
- `versions/v13/identity_bridge.py`
- `versions/v13/configs/phase4_precision_config.json`
- `versions/v13/experiments/phase4_precision_identity.py`
- `versions/v13/experiments/phase4_egohumans_identity.py`

主要结果：

- `output/v13/phase4_identity/three/v13_phase4_precision_identity.md`
- `output/v13/phase4_identity/dance/v13_phase4_precision_identity.md`
- `output/v13/phase4_identity/box/v13_phase4_precision_identity.md`
- `output/v13/phase4_identity/egohumans_001_legoassemble/v13_phase4_egohumans_identity.md`

可视化：`feature_probe.png`、`risk_coverage.png`、`same_different_distance.png`、
`precision_match_matrices.png`、`egohumans_precision_matrices.png` 和
`egohumans_predicted_bbox_overlay.png`。

验证：

```text
15/15 direct tests passed
py_compile passed
```
