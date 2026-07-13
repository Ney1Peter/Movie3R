# V10 Human3R-Specific Streaming Global Alignment Plan

Date: 2026-07-13

## 1. Core Position

HumanMM gives a useful lesson: multi-shot reconstruction should not be solved by
one end-to-end correction module that learns everything at once. It decomposes
shot detection, per-shot initialization, cross-shot orientation alignment, pose
recovery, velocity/contact prediction, and trajectory refinement.

Movie3R should adopt the decomposition idea, but not copy HumanMM's target or
implementation. HumanMM focuses on global human motion recovery. Our target is
different:

> Given Human3R's streaming monocular output, align camera, human, and point
> cloud from different shots into one causal global coordinate system.

The key module is therefore not a human-motion transformer or an offline
trajectory optimizer. The key module is a streaming segment-to-global gauge
alignment layer.

## 2. Why This Is Not Just HumanMM

HumanMM:

- estimates per-shot camera using Masked LEAP-VO;
- initializes human motion using GVHMR;
- aligns cross-shot human orientation with an explicit geometry module;
- refines the whole human motion sequence with ms-HMR and trajectory modules;
- is not designed as a strict Human3R-style camera/human/scene reconstruction
  system.

Movie3R / V10:

- uses frozen original Human3R as the local reconstructor;
- keeps Human3R's recurrent within-shot state for stable continuous frames;
- resets or forks a local state only when a shot boundary is detected;
- maintains a separate causal global state across shots;
- estimates one segment-to-global transform and applies it consistently to
  camera, SMPLX/human, and point cloud;
- avoids global BA and avoids future-frame sequence optimization.

This gives a clearer novelty:

> Human3R reconstructs local shots well but lacks a cross-shot global state.
> V10 adds a causal global state alignment layer that turns local Human3R
> reconstructions into a globally consistent multi-shot reconstruction.

## 3. Current Problem Diagnosis

The V9 correct-token route proved that token information is useful, but it also
showed the limitation of local frame correction:

- stable AA frames can be over-corrected;
- A/B shot gauges remain inconsistent;
- direct learned SE(3) regression from human anchors overfits;
- token-only segment alignment contains useful signals but is not stable enough
  to be the main alignment mechanism.

The current best interpretation is:

1. Human3R should remain responsible for within-shot reconstruction.
2. Shot discontinuity should trigger a state operation, not per-frame universal
   correction.
3. Alignment is a strong geometry problem, so geometry should propose the main
   transform.
4. Learning should decide reliability, anchor weights, gates, and small
   residuals, not invent the full transform from scratch.

## 4. Proposed V10 Modules

### M0. Frozen Human3R Local Reconstructor

Use strict original Human3R for every frame. For continuous frames, keep its
normal recurrent state and do not apply extra correction.

Output per frame:

- camera pose;
- SMPLX / human joints;
- point cloud or depth payload;
- optional pose/human/state token summaries for reliability prediction.

### M1. Streaming Shot Boundary Detector

Predict whether the current frame starts a new local segment.

First training target:

- image-only adjacent-frame detector;
- label sequence like `[0, 0, 1, 0]` for AABB;
- later add bbox / pose / Human3R-output features as teacher signals.

Detector and alignment should be trained separately at first. During alignment
development, use oracle boundaries to isolate the problem.

### M2. Local State Manager

If no boundary:

- continue Human3R local state;
- apply current cached segment-to-global transform;
- update global state causally.

If boundary:

- reset or fork Human3R local state;
- let the new shot produce a local reconstruction;
- ask the global alignment module how to attach this new local segment to the
  historical global state.

This is the main difference from V9. The operation happens at state/segment
level, not by correcting every frame token.

### M3. Causal Global State

The global state should summarize what the world looked like before the current
frame:

- recent aligned camera trajectory;
- recent aligned human anchors;
- estimated human motion trend;
- cached transform of the current segment;
- optional point-cloud / floor / scene anchors;
- confidence statistics for Human3R human output and detector output.

For static-human data, the state predicts a nearly fixed human anchor. For
motion data, the state should predict where the human should be now if the shot
had not changed.

### M4. Geometry Proposal

At a boundary, compute a strong initial transform `T_geo` from reliable anchors.

Candidate anchors:

- pelvis/root;
- hip pair;
- torso/spine;
- head;
- feet;
- optional floor/upright cue;
- optional sparse scene anchors if reliable.

`T_geo` is not the final contribution. It is a constrained proposal. It provides
the stable coarse alignment that a small network should not have to rediscover.

### M5. Learned Reliability / Residual Head

The learned module should be conservative.

Inputs:

- global-state anchor prediction;
- current local Human3R anchors;
- `T_geo`;
- residual after applying `T_geo`;
- Human3R pose/human/state token summaries;
- detector confidence;
- optional motion features from recent global state.

Outputs:

- anchor weights or anchor reliability;
- alignment gate;
- small SE(3) residual around `T_geo`;
- optional state update gate.

The network should not output a free full SE(3) transform as the main path. That
was the unstable direct-regression version.

### M6. Causal Segment Integration

After estimating `T_final`, cache it for the current segment.

For following frames in the same segment:

- keep using Human3R's local recurrent state;
- transform camera/human/point cloud with the cached segment transform;
- optionally update the transform slowly with a causal filter if confidence is
  high.

This is the V10 version of trajectory refinement. It is causal and state-based,
not an offline whole-sequence optimizer.

## 5. Training Strategy

### Stage A. Detector Training

Train separately from alignment.

Data construction:

- continuous pairs as negative samples;
- cross-view or cross-shot pairs as positive samples;
- AABB / ABAB / ABBA / AABC / ABCD patterns for boundary labels;
- image-only first, then add optional teacher features.

Loss:

- binary cross entropy for boundary;
- emphasize low false-positive rate on stable frames.

Reason:

false positives are costly because they reset state and may damage frames that
Human3R already reconstructs correctly.

### Stage B. Geometry Proposal Baseline

No learning.

Use oracle boundary and evaluate:

- original Human3R local reset;
- fixed `T_geo`;
- cached `T_geo` for the rest of the segment.

This is the strong baseline. Any learned method must not be worse than it.

### Stage C. Learned Anchor Reliability

Train a small module to choose anchor weights, not to predict the whole transform.

Training target:

- weighted Procrustes result after learned weights should improve camera/human
  metrics versus fixed weights;
- weights should not collapse to one joint;
- stable cases should stay close to fixed geometry.

Loss:

- final camera rotation/translation loss;
- human anchor loss;
- body frame / body vector loss;
- anchor weight regularization;
- proposal non-degradation loss.

This is currently the most defensible learned component because it explains what
the network contributes beyond direct SMPLX alignment: it learns which anchors
are reliable.

### Stage D. Small Residual Around Geometry

Only after Stage C is stable, add a small residual.

Constraint:

- residual rotation should be bounded, e.g. 5-10 degrees;
- residual translation should be bounded, e.g. 0.2-0.5 m;
- default residual should be identity.

Loss:

- final alignment loss;
- residual prior;
- residual target loss if a target transform is available;
- proposal improvement loss so `T_final` does not damage `T_geo`.

This stage is where token/state features can be useful. They should guide a
small correction, not replace geometry.

### Stage E. Motion-Aware Global State

This is the key extension beyond static-person post-processing.

Data does not have to be true multi-shot GT. We can construct training pairs
from continuous monocular videos:

1. Run frozen Human3R on continuous video.
2. Treat its aligned continuous output as pseudo-global state.
3. Split the sequence into artificial segments.
4. Apply random SE(3) gauge perturbations to the later segment.
5. Train V10 to recover the original continuous gauge causally.

For static data, the expected anchor is almost fixed. For motion data, the
expected anchor is predicted from recent velocity/state. This makes the method
more meaningful than simply aligning the current SMPLX root to the previous
root.

Loss:

- predicted-anchor loss from state;
- segment-to-global transform loss;
- final camera/human/point-cloud consistency;
- no-op loss for non-boundary frames;
- causal smoothness of cached segment transform.

## 6. Dataset Usage

Use different data for different modules.

Detector:

- can use many unlabeled or weakly labeled videos;
- labels can be synthetic from constructed patterns;
- does not require SMPLX/camera GT.

Alignment geometry / reliability:

- use 4-source AABB data with GT where available;
- use AIST/H36M style multi-view clips for validation;
- use synthetic random gauge perturbation to enlarge training variety.

Motion-aware state:

- can use ordinary monocular motion videos;
- Human3R outputs provide pseudo camera/human/scene;
- no need for full ground-truth multi-shot SMPLX/camera for every sample.

The important shift is:

> We do not train only on scarce real multi-shot GT. We train the alignment
> module by perturbing local segment gauges and asking the model to restore a
> causal global state.

## 7. Minimal Experimental Path

1. Keep current `fixed_geo` as the strong baseline.
2. Run `learned anchor weights` on a medium four-source set.
3. Add token/state summaries only as reliability features.
4. Test held-out AIST/H36M and motion clips, not only training clips.
5. Build the perturbation-based pseudo-training set for motion-aware state.
6. Compare:
   - original Human3R;
   - local reset only;
   - fixed geometry alignment;
   - learned anchor reliability;
   - learned anchor reliability + state-aware residual.

Success criteria:

- stable frames remain close to original Human3R;
- boundary frames align A/B camera, human, and point cloud better;
- moving-person clips are not collapsed by naive human-root alignment;
- learned component beats fixed geometry on at least some held-out cases without
  hurting the average.

## 8. How To Present The Innovation

Recommended wording:

> Existing Human3R-style streaming reconstruction has strong local reconstruction
> ability but lacks a causal global state for shot-discontinuous input. We propose
> a Human3R-specific streaming global alignment framework. It keeps the original
> model intact within each shot, detects shot boundaries, resets the local state,
> and attaches each new local reconstruction to a maintained global state through
> geometry-constrained, reliability-aware segment alignment.

The contribution is not just "use human as anchor". The contribution is:

1. Define shot-discontinuous monocular Human3R reconstruction as a segment gauge
   alignment problem.
2. Separate local Human3R state from cross-shot global state.
3. Align camera, human, and point cloud jointly through one segment transform.
4. Use explicit geometry as proposal and learning for reliability/state-aware
   refinement.
5. Train with synthetic segment-gauge perturbations, reducing dependence on rare
   real multi-shot GT.
