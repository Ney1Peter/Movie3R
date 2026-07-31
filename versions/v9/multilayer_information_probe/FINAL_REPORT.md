# V9 Multi-Layer Information Feasibility Result

Date: 2026-07-31

## Question

This isolated experiment tests two hypotheses:

1. V9 correction is limited because Human3R only exposes late, semantically compressed
   encoder features.
2. A small direct residual head or an additional V9 evidence token could recover useful
   camera-cut information from earlier or multiple layers.

This is an information-capacity probe, not a replacement Movie3R method.

## Isolation

- No file under `src/`, existing config, checkpoint, or output was modified.
- Human3R and V9 checkpoints remain frozen and read-only.
- All new code is under `versions/v9/multilayer_information_probe/`.
- All generated artifacts are under `output/v9_multilayer_information_probe/`.
- The probe uses forward hooks and saves compact relation descriptors, not feature maps.

## Architecture Audit

The information bottleneck concern is partly correct:

- The DINO wrapper returns one default late representation from
  `get_intermediate_layers(x)[0]`.
- The CUT3R image encoder executes all 24 blocks but returns only its normalized final
  feature map.
- The pose path obtains an image-level cue by averaging all final image tokens.
- V9 uses one learned relation query to compress current image, pose, and human context.
- Its default residual head mean-pools the refined semantic, alignment, and momentum
  correction tokens before predicting a pose-latent residual.

The main Human3R recurrent decoder itself is not restricted to a single pooled image
token: it still attends over the final patch-token map. The strongest compression occurs
in the pose/correction routing around that decoder.

## Protocol

### Frozen models

- Original Human3R: `src/human3r_896L.pth`
- Formal V9: `checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth`

### Features

- CUT3R encoder blocks: 5, 11, 17, 23
- DINOv2 blocks: 5, 11, 17, 23
- Human3R decoder blocks: 2, 5, 8, 11
- Individual layers, same-backbone multi-layer concatenation, and cross-backbone
  concatenation

### Data

- Frozen evaluation: 10 wide-view cuts from AvatarReX, THuman, MVHuman100, and
  MVHuman200
- Small training probe: 94 valid cuts, with frozen camera pairs excluded
- Scaled training probe: 382 valid cuts, with frozen camera pairs and their reverse
  direction absent
- Two incomplete MVHuman records were detected and skipped before indexing; no labels
  were shifted

Each sample contains the second pre-cut frame and first post-cut frame. No future
post-cut frame is used.

### Targets

Two targets were tested:

1. `absolute`: predict the GT relative camera transform directly.
2. `residual`: predict a correction composed with raw Human3R. The raw relative pose is
   included for every residual feature group.

The residual target is the relevant V9 diagnostic. Camera composite is
`translation_m + 0.02 * rotation_deg`.

### Capacity controls

All tested layer descriptors can fit the single `lbn1_1192` cut to numerical zero,
and most can fit the ten frozen cuts to near zero when those same ten labels are used
for training. This confirms that the small heads have enough capacity, but it also means
single-case overfit cannot tell us which representation generalizes. The decision below
therefore uses only camera-pair-disjoint held-out results and repeated learning curves.

## Core Results

### Formal checkpoint baseline on the same ten cuts

| Method | Translation m | Rotation deg | Composite | Median | P90 |
|---|---:|---:|---:|---:|---:|
| Original Human3R | 1.919 | 114.77 | 4.214 | 4.940 | 6.325 |
| V9 full | 0.799 | 53.15 | 1.862 | 0.451 | 4.894 |
| V9 without correction prompt | 1.973 | 115.10 | 4.275 | 5.001 | 6.313 |

Disabling the human latent correction leaves these camera numbers unchanged, as
expected: that branch changes SMPL output, not the camera head. The V9 correction prompt
is therefore genuinely responsible for its camera improvement.

V9 is not uniformly robust across sources. Its mean composite is approximately `0.243`
on AvatarReX, `0.208` on THuman, `4.617` on MVHuman100, and `1.811` on MVHuman200.

### Camera-pair-CV residual probe, 382 training cuts

Ridge regularization was selected using five-fold camera-pair-grouped CV on training
data only. The ten frozen cuts were not used to choose regularization.

| Feature | Translation m | Rotation deg | Composite | P90 |
|---|---:|---:|---:|---:|
| Decoder block 8 | 1.162 | 43.32 | **2.029** | 3.653 |
| Decoder block 5 | 1.179 | 46.07 | 2.101 | 4.163 |
| Decoder final block 11 | 1.285 | 43.14 | 2.148 | 3.567 |
| Decoder multi-layer | 1.119 | 52.67 | 2.173 | 3.952 |
| DINO block 5 | 1.385 | 45.26 | 2.290 | 4.316 |
| CUT3R final block 23 | 1.384 | 48.99 | 2.364 | 4.556 |
| Raw-pose-only learned control | 1.586 | 38.91 | 2.364 | 3.864 |
| DINO final block 23 | 1.422 | 50.68 | 2.435 | 4.180 |
| CUT3R block 5 | 1.636 | 48.68 | 2.610 | 4.956 |
| All layers and backbones | 1.346 | 59.27 | 2.531 | 3.823 |

Decoder block 8 improves the learned raw-pose control by `0.335` composite and wins on
7/10 cuts. However, the paired bootstrap 95% interval is `[-0.185, 1.002]`, which crosses
zero. This is useful evidence, but not enough to claim a statistically secure final
method.

The decoder-block-8 probe is more balanced than formal V9: its source composites are
approximately `2.269`, `1.895`, `1.879`, and `2.027` in the same source order. It removes
V9's MVHuman100 tail but gives back V9's very strong AvatarReX/THuman performance.

## Data-Scale Result

The data hypothesis is only partially supported.

At 48 to 90 training cuts per source:

| Feature | 48/source | 90/source |
|---|---:|---:|
| Raw-pose-only | 2.432 +/- 0.219 | 2.431 +/- 0.033 |
| DINO block 5 | 2.456 +/- 0.226 | 2.309 +/- 0.063 |
| Decoder block 5 | 2.277 +/- 0.230 | 2.112 +/- 0.042 |
| Decoder block 8 | 2.158 +/- 0.248 | **2.029 +/- 0.032** |
| Decoder final block 11 | 2.242 +/- 0.206 | 2.153 +/- 0.028 |

More data improves the decoder probes and reduces seed variance, but it does not make
early DINO or naive multi-layer concatenation dominant. Raw quantity is therefore not
the only bottleneck. Camera-pair, source, and capture coverage matter more than simply
repeating more frames from the same rigs.

## Answers

### Is early DINO information clearly stronger?

No. DINO block 5 is slightly better than DINO block 23 at scale, but it does not beat
the formal V9 checkpoint and only modestly beats the learned raw-pose control. This does
not justify routing large early DINO maps into the recurrent decoder.

### Is the CUT3R encoder over-compressed?

Not in the simple form hypothesized. Its early block is not better than its final block
on the robust residual test. Multi-layer CUT3R concatenation is also not stable.

### Is useful information lost elsewhere?

Probably. Middle Human3R decoder states are consistently stronger than the final
decoder state and encoder-only features. Combined with V9's one-query context
compression and final correction-token mean pooling, the evidence points more toward a
pose/correction routing bottleneck than a missing early-image-feature bottleneck.

### Should DINO features directly predict the explicit correction?

No, not as the main method. The direct head is useful as a capacity diagnostic, but its
held-out result is below formal V9 and its evidence is not strong enough to bypass
decoder refinement.

### Should a multi-layer evidence token be added to V9 now?

No. The predeclared condition was that a multi-layer group must clearly and stably beat
final-only features. It did not: naive multi-layer and all-backbone groups are generally
worse than a single middle decoder layer. Training an evidence token after this result
would be an unjustified architecture expansion.

## Final Decision

Do not add early-DINO concatenation or a direct DINO-to-SE(3) branch to V14.

The next architecture experiment, if this direction is resumed, should be narrowly
defined as a middle-decoder pose/correction readout or layerwise pose adapter, with:

- one bounded residual around raw Human3R/V14 B0;
- no independent human/camera transforms;
- no full early feature maps in recurrent state;
- camera-pair-disjoint and source-disjoint evaluation;
- formal V9 and B0 as required baselines.

This probe does not prove that more pretraining data is useless. It proves that, under
the available four-source data and causal first-post-cut protocol, the proposed early
DINO/multi-layer concatenation is not the missing fix. The strongest actionable signal
is in Human3R's middle decoder, while the strongest deployable baseline remains formal
V9 followed by the explicit V14 alignment path.

## Artifacts

- `run_probe.py`: frozen extraction and absolute/residual probes
- `robust_analysis.py`: camera-pair CV, repeated learning curves, paired bootstrap
- `evaluate_v9_checkpoint.py`: formal V9 ablations on identical cuts
- `output/v9_multilayer_information_probe/full/report.json`
- `output/v9_multilayer_information_probe/scale96/robust_analysis.json`
- `output/v9_multilayer_information_probe/v9_checkpoint_eval.json`
