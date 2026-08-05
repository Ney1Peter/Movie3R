# V9 Decoder and Correct-Token Final Report

## Scope

This experiment tests whether V9 correction is limited by:

1. insufficient information in the decoder;
2. the semantic/alignment/momentum correct-token construction;
3. final token pooling and residual readout;
4. insufficient correction training data.

All work is isolated under `versions/v9/decoder_correct_token_probe/` and
`output/v9_decoder_correct_token_probe/`. Formal V9, V14, Human3R, their
checkpoints, geometry, and human heads were not modified.

## Protocol

- Formal read-only checkpoint: `checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth`
- Training sources: AvatarReX, THuman, MVHuman100, and MVHuman200
- Scale-up set: 96 camera-pair-disjoint cuts per source, 384 total
- Frozen evaluation: the existing ten cuts, never used to choose training steps or architecture
- Decoder probes: blocks 2, 5, 8, and 11
- Captured streams: semantic, alignment, momentum, their mean/concatenations,
  native pose token, image-token mean, and human-token mean
- Metrics: camera translation, rotation, composite, median, P90, per-source results,
  camera-pair grouped CV, and leave-one-source-out validation

## Stage 0: Formal V9 Behavior

Formal V9 is strongly source-dependent on the 384 training cuts.

| Source | Raw composite | Formal V9 | V9 win fraction |
|---|---:|---:|---:|
| AvatarReX | 0.992 | 0.620 | 80.2% |
| THuman | 0.264 | 0.152 | 86.5% |
| MVHuman100 | 2.963 | 2.918 | 59.4% |
| MVHuman200 | 2.649 | 2.630 | 49.0% |

The current residual is useful on AvatarReX and THuman but nearly neutral on
MVHuman. This is a distribution/generalization problem, not simply insufficient
residual magnitude.

## Stage 1: Decoder and Correct-Token Evidence

The 96-cut/source frozen probe produced the following key results.

| Readout | Pair-grouped CV | Leave-source-out | Frozen composite | Frozen P90 |
|---|---:|---:|---:|---:|
| Decoder L11 pose | **1.698** | **2.214** | **1.179** | 3.362 |
| Decoder L11 image mean | 1.916 | 2.716 | 1.349 | **2.495** |
| Decoder L8 semantic + alignment | 2.000 | 2.549 | 1.424 | 2.816 |
| Final correct-token mean | 1.782 | 2.500 | 1.602 | 3.219 |
| Raw-pose-only learned readout | 1.980 | 3.669 | 1.868 | 4.292 |
| Formal V9 head | - | - | 1.862 | 4.894 |

Main findings:

1. The decoder contains correction evidence. L11 pose is the strongest and most
   repeatable stream.
2. Pre-decoder prompt tokens are weaker than decoder-refined tokens.
3. Averaging semantic/alignment/momentum is suboptimal, but naive concatenation
   is also not a solution. The bottleneck is not mean pooling alone.
4. Multi-depth concatenation does not improve over one deep token.
5. The native pose token is a better correction carrier than the hand-designed
   correct-token mean. Correction evidence is routed into the native decoding
   path even when the dedicated residual readout does not use it effectively.

Therefore, early DINO/CUT3R features and a larger pre-decoder correct token are
not the next priority. The supported direction is a small final-decoder pose
relation readout.

## Stage 2: Pose-Relation Residual Head

The tested head preserves four roles separately:

```text
pre-cut L11 pose token
post-cut L11 pose token
normalized token difference
elementwise token interaction
+ raw/formal camera context
-> independent low-rank projections
-> one bounded SE(3) residual around formal V9
```

It has 175,114 parameters. Human3R, decoder, prompt, original heads, LoRA, and
human reconstruction remain frozen.

### Training progression

| Stage | Training cuts | Formal V9 | Structured head | Conclusion |
|---|---:|---:|---:|---|
| Single overfit | 1 | 0.391 | approximately 0 | Optimization path is correct |
| Small | 10 | 1.862 | 2.242 | Too little data; no generalization |
| Full | 384 | 1.862 | 1.624 | Mean improves, safety fails |

The flat 420k-parameter MLP reached 1.663 composite and 4.356 P90. The selected
175k structured head reached 1.624 composite and 3.945 P90, so preserving the
relation roles is both smaller and better than flat concatenation.

### Full frozen result

| Metric | Formal V9 | Selected relation head |
|---|---:|---:|
| Mean composite | 1.862 | **1.624** |
| Median composite | **0.451** | 0.516 |
| P90 composite | 4.894 | **3.945** |

The mean improves by 12.8% and P90 by 19.4%, but median performance becomes
slightly worse.

| Source | Formal V9 | Selected relation head |
|---|---:|---:|
| AvatarReX | 0.243 | **0.219** |
| THuman | 0.208 | **0.203** |
| MVHuman100 | 4.617 | **3.533** |
| MVHuman200 | **1.811** | 2.288 |

The decisive failure is a frozen MVHuman200 cut:

```text
formal V9 composite: 0.511
new residual:        3.871
```

This is a new catastrophic error and prevents promotion.

## Stage 3: Safety Tests

The following training-only controls were tested and rejected:

1. global residual scaling;
2. rotation/translation residual bounds;
3. five-head ensemble disagreement;
4. a learned "formal V9 has failed" gate.

Global scaling selected full application. Strict bounds reduced useful large
corrections but did not remove harmful ones. Ensemble uncertainty overlapped
between useful and harmful MVHuman cases. The learned gate separated dataset
families rather than hard cases within one family, and accepted the same bad
MVHuman200 cut.

## Final Decision

The experiment reaches a clear architectural conclusion but does not produce a
deployable replacement head.

### Keep

- Decoder L11 pose is the most useful correction representation found so far.
- Role-preserving pose-relation projection is more efficient than flat pooling.
- Four-source training is necessary; one or ten sequences can only demonstrate
  capacity, not generalization.
- A future correction head should operate near the final decoder pose path and
  predict a one-shot residual around the formal/coarse camera result.

### Do not promote

- No new correct token is inserted into formal V9/V14.
- The 175k pose-relation head remains an experimental checkpoint only.
- No uncertainty gate or residual bound is added to the main method.
- Formal V9 remains the active implicit coarse-correction checkpoint.

### Bottleneck

The current bottleneck is not missing early encoder information. It is reliable
cross-source residual generalization: the decoder exposes enough evidence to
reduce average and tail errors, but the available supervision does not let a
small head distinguish every already-correct MVHuman case from a failed one.

More architecture search on the same 384 cuts is unlikely to solve this safely.
The next valid attempt would require broader, camera-pair-diverse correction
supervision or an independently verifiable geometric acceptance rule. It should
not add more early-image features or a larger ad hoc correct token.

## Artifacts

- Frozen 24/source report: `output/v9_decoder_correct_token_probe/full24/robust_analysis.md`
- Frozen 96/source report: `output/v9_decoder_correct_token_probe/full96/robust_analysis.md`
- Architecture search: `output/v9_decoder_correct_token_probe/pose_relation_head/train_only_arch_search.json`
- Selected checkpoint/report: `output/v9_decoder_correct_token_probe/pose_relation_head_selected/full_structured/`
- Residual-bound audit: `output/v9_decoder_correct_token_probe/pose_relation_head_selected/bound_search.json`
