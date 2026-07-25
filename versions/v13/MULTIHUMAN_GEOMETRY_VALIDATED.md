# V13 Milestone: Multi-Human Geometry Validated

记录日期：2026-07-24

## 里程碑状态

```text
GT-ID multi-human shared-Boundary geometry: VALIDATED
Deployable cross-shot multi-human system: NOT YET VALIDATED
Research decision: PROCEED TO WHO / RE-ID
```

当前结果足以支持以下研究判断：在人物身份关联正确的前提下，多个人共同约束一个
shared Boundary，确实比 first/largest/highest-confidence 等可部署单人 anchor 更稳定。
因此，多人路线不是由错误人物交换造成的假收益或无效方向，值得继续进行跨镜头身份
关联、进入/离开处理和 geometry verification 实验。

这里的“有效”只指 **GT-ID 条件下的多人几何可行性**。当前 identity association 使用
GT SMPL-X mesh projection，不能进入真实部署路径，也不能据此宣称 V13 已经解决多人
cross-shot Re-ID。

## 冻结方法

本里程碑固定为 Lite setting：

```text
5 pre-cut frames + 1 fresh post-cut frame
-> frozen Human3R multi-human reconstruction
-> pre-decode hard reset
-> strict GT-ID association (Oracle WHO)
-> each valid matched human produces R_i and t_i
-> R = SO(3) mean of all valid R_i
-> t = arithmetic mean of all valid t_i
-> one shared Boundary applied to camera, pointmap and all humans
```

固定启用：Fixed Explicit、V16 torso rotation（20 degree bound）、显式 translation 和
one fixed shot-level Boundary。

固定关闭：DA3、Keypoint R-CNN、V11.4 scale、VGGT、continuity、scene refinement、
token Re-ID 和 learned identity adapter。Shot scale 为 `s=1`。

## 支撑结果

### MultiHuman `three`

- 315 个 cuts，308 个 cuts 至少有两名有效 matched humans；
- 308-case common support 上，highest-confidence single 的 composite 为 `0.764`，
  naive multi-human mean 为 `0.657`；
- paired composite improvement rate 为 `74.0%`，`p=1.20e-16`；
- camera rotation 从 `9.96 deg` 降到 `7.01 deg`；
- camera translation 从 `0.565 m` 降到 `0.517 m`；
- 无 catastrophic failure。

在三人均有效的 common support 上，使用 1/2/3 人时 composite 分别为：

| 人数 | Composite mean | P90 |
|---:|---:|---:|
| 1 | 0.843 | 1.315 |
| 2 | 0.681 | 1.024 |
| 3 | **0.611** | **0.920** |

人数增加带来单调改善，说明收益来自多个人提供的独立几何冗余，而不是某个固定人物
恰好较好。

### MultiHuman `dance`

- 36 个 cuts；
- 25 个 cuts 有两人，可执行多人 fusion；
- 11 个 cuts 只有一人，自动 fallback 到单人；
- multi-human support 上 highest-confidence single composite 为 `0.809`，naive
  two-human mean 为 `0.745`；
- 无 catastrophic failure。

`dance` 样本量较小，当前只作为跨序列 pilot，但方向与 `three` 一致。

## 当前解释

1. 多人最明确的独立贡献是减少 torso rotation ambiguity。
2. 只平均 translation 没有改善；rotation consensus 已明显改善，rotation 与 translation
   联合平均后效果最好。
3. 多人的误差在当前数据上具有一定互补性，保留全部有效约束优于按单一 residual 删除人。
4. quality、visibility、motion、dispersion 和 layout 的手工 soft weighting 在 held-out
   数据上没有稳定超过 naive mean，因此不进入默认方法。
5. Naive multi mean 没有超过读取 GT evaluator 的 Oracle Best Single。这不影响“多人优于
   可部署单人选择器”的结论，但说明当前 fusion 仍不是理论上界。

## 下一阶段准入决策

V13 通过多人几何 feasibility gate，可以进入 identity 阶段。下一阶段应按以下顺序进行：

1. 审计和提取 Human3R refined human token；
2. 测试 hard reset 后的 cross-shot token Re-ID；
3. 加入 dustbin、新人物、消失人物和有限 tracklet TTL；
4. 使用 token 回答 WHO，使用显式多人几何回答 WHERE；
5. 用 geometry residual 验证 tentative identity，最多 reject/re-solve 一次；
6. 严格执行 Match-Then-Align 和 Align-Then-Commit；
7. 单人或无可靠匹配时自动退化到 V12/Lite fallback。

Human token 仍不得直接预测 rotation、translation、scale 或 Boundary。所有 matched humans
必须共享同一个 Boundary。

## 不能宣称

- 不能宣称当前 V13 已可部署；
- 不能宣称已经完成 native-token cross-shot Re-ID；
- 不能宣称已经解决人物进入、离开、遮挡和重新出现；
- 不能把 GT mesh projection association 写成推理时可用模块；
- 不能宣称 soft uncertainty fusion 优于 naive mean；
- 不能把两个调试序列视为最终跨数据 benchmark。

## 后续身份阶段状态

该里程碑之后的 Phase 3/4/5 不改变 GT-ID geometry 结论：

- Phase 3 native token bridge 未通过 catastrophic-swap gate；
- Phase 4 precision-first appearance 达到零错误但覆盖过低；
- Phase 5 running-mean persistent state 在 `three` 上将安全 multi coverage 提高到 `50%`，
  Top-6 recall 达到 `92.22%`；
- Phase 5 手工 joint WHO-WHERE scorer 的 zero-wrong coverage 只有 `35.56%`，低于
  identity-only `51.11%`，因此没有进入 commit 或 frozen evaluation。

所以当前研究决定从“继续尝试 WHO”更新为：**保留 GT-ID geometry Oracle 和 persistent
state 分析，但 automatic multi-human 默认继续关闭。**

详细实验和完整数值见：

- `docs/V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md`
- `docs/V13_PHASE2_MULTIHUMAN_FUSION_OPTIMIZATION.md`
- `docs/V13_PHASE5_CAUSAL_IDENTITY_STATE.md`
- `output/v13/phase2_fusion/v13_phase2_fusion.json`
- `output/v13/dance_phase2/fusion/v13_phase2_fusion.json`
