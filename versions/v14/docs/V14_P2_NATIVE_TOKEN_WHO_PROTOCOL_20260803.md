# V14 P2：B0 后的 Native Human Token 匿名关联诊断

日期：2026-08-03  
状态：**已完成；No-Go。**

## 为什么 P2 是 P1 之后唯一合理的短路线

P1 已在正确的 joint-UV 足部 patch 上证明：完整 first-post 多人 stream 中，双足局部
scene support 覆盖不足，不能成为通用 root residual 的观测量。它不应靠放宽 gate 变成
猜测。

但 BRTC 仍有一个前置的、独立的可观测问题：它只有在 pre/post 是同一个人时才有资格
三角化 root。P0 five-chain confirmation 的 evaluator-only association 正确率是 `21/23`
而非 100%；旧几何 dustbin 又在 equal-count full-swap 上有反例。这里需要的不是另一套
root scalar，而是一个不新引入 ReID backbone 的 **WHO certificate**。

P2 只审计 Human3R 同一次 forward 已有的 per-person descriptor：

```text
pre last-frame native human token + post first-frame clean-reset native human token
  -> normalized cosine costs / mutuality / relative margin
  -> (optionally with B0-aligned root+torso geometry as WHERE cost)
  -> high-precision anonymous match certificate
  -> only certified pairs may enter already frozen BRTC/Kabsch
```

它不从 token 回归 camera、B0、root residual、shape 或 pose；也不训练 token、引入
DINO/CLIP/ReID/DA3，或用 GT identity 作运行时 feature。低置信匹配只意味着该 person
exact B0 fallback。

## 固定数据、forward 和 GT 隔离

使用和 P1 相同、已打开的 `three` 36-event P0 development split：

```text
checkpoint: v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth
SHA256:     de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265
runtime:    read-only shadow -> B0; independent clean raw reset; B0 camera frozen
```

每个 case 的 runtime 先生成 native descriptor、B0-aligned geometry cost、candidate
matches 和 all accept/reject decisions；之后才由 evaluator 对 pre/post detection 做 GT
identity assignment。token array 的 row 数必须和 Human3R detection row 一致，否则该
descriptor candidate 逐 case abstain，不能按 row 截断或猜测匹配。

P2 是 development-only association diagnosis；它不把 `dance`、`box` 或已读
EgoHumans confirmation 当作 selector。只有 high-precision certificate 成立，才可在
一个独立 manifest 做 frozen confirmation。

## 固定 candidate 与决策规则

不训练、不扫阈值。对同一个 detection set 预注册计算：

1. `G`: 现有 B0-aligned root+torso+joints Hungarian（无 abstention，对照）。
2. `T-refined`: L2-normalized `refined_human_tokens` cosine Hungarian。
3. `T-head`: L2-normalized `human_head_tokens` cosine Hungarian。
4. `T-mhmr`: L2-normalized `mhmr_head_tokens` cosine Hungarian。
5. `TG`: `T-refined` 与 `G` 各自在该 cost matrix 内除以正的 matrix median 后等权相加
   的 Hungarian；这是固定、无量纲的 WHO/WHERE 分工，而不是 token 预测几何。
6. `TG-cert`: 只接受 `TG` Hungarian pair，且该 pair 同时是 token cost 的 row/column
   mutual nearest，并且 token row 和 column 的 relative second-best margin 都 ≥ `0.10`。
   未满足的检测全部 abstain。

对人数少于 2 或 token 无法对齐 detection row 的 case，`TG-cert` abstain。`0.10` 是
在读取本 split metrics 前写死的 cosine relative-margin guard；不会针对结果修改。

## 可证伪 Go / No-Go

P2 的目的是为 BRTC 提供**安全资格**，因此 precision 重于 coverage。仅当 `TG-cert`
同时满足以下条件才进入一次独立 confirmation：

```text
all accepted pairs evaluator-correct                 (wrong accepted = 0)
accepted coverage >= 20% of GT-evaluable pre/post detections
TG forced-Hungarian accuracy >= G forced-Hungarian accuracy
token rows/detection rows parity = 100%
camera hash and all non-accepted geometry = bit-exact
```

否则记录：

```text
NO_GO_NATIVE_TOKEN_WHO_CERTIFICATE
```

即使 Go，P2 也只证明 B0 后 native tokens 可以作为 **WHO gate**；它不证明 token 能
纠正 root，不允许把 `TG` 当全局 Boundary proposal，也不能声称 variable visibility 或
all identities 已经被解决。

## 已完成结果（2026-08-03）

结论：

```text
NO_GO_NATIVE_TOKEN_WHO_CERTIFICATE
```

固定 checkpoint、36-event `three` development split、runtime/GT 隔离和所有候选规则均
保持为本文预注册版本。完整 cache 和机器可读报告为：

```text
cache:  output/v14/fine_alignment_research/p2_native_token_who/
report: .../P2_NATIVE_TOKEN_WHO_REPORT.json
checkpoint SHA256: de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265
```

每个 event 均先在 prediction-only runtime 中构造 B0-aligned geometry、native descriptors
和所有 assignment/certificate；仅随后 evaluator 才读取 GT identity。三种 token 都与
Human3R detection row 完全对齐（各 `36/36` case），所有 B0 camera hash 均保持不变，
未使用 future post frame、GT runtime feature 或外部预训练模型。

| Candidate | Accepted/evaluable | Correct | Precision | Coverage vs. geometric rows |
|---|---:|---:|---:|---:|
| `G` geometry Hungarian | 96 / 96 | 92 | 95.83% | 100.00% |
| `T-refined` | 96 / 96 | 56 | 58.33% | 100.00% |
| `T-head` | 96 / 96 | 56 | 58.33% | 100.00% |
| `T-mhmr` | 96 / 96 | 58 | 60.42% | 100.00% |
| `TG` equal fusion | 96 / 96 | 70 | 72.92% | 100.00% |
| `TG-cert` mutual + 10% margin | 17 / 17 | 11 | 64.71% | 17.71% |

所以 Go 所需的三个关键条件同时失败：`TG-cert` 有 `6` 个错误接受（而非零），coverage
`17.71% < 20%`，且强制 `TG` 的 accuracy (`72.92%`) 低于 geometry (`95.83%`)。row parity
和 camera bit-exact 两项仅验证实现完整性，不能抵消身份错误。特别地，margin certificate
仍错误接受了 `t0700_c0↔c3` 的各一项以及 `t0700_c2↔c5` 的各两项错配；它并不是安全的
abstention rule。

科学结论是：Human3R 当前 forward 中的 native human descriptors 可按 detection row
稳定导出，但在跨相机/shot 的匿名多人关联上不是独立的 WHO 证据；将它与 WHERE 融合反而
明显破坏了本已较强的几何匹配。不得为救此结果在已读 split 上扫描 token layer、fusion
weight 或 margin，也不得把它接到 BRTC/Kabsch。P2 不读取任何 independent confirmation
split；后续若重新考虑 appearance cue，必须使用新的、与本 token family 正交的可观测证据
和全新预注册 protocol。
