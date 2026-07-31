# V14 Causal B0 + Automatic-ID Multi-Human Status

Date: 2026-07-29

## 1. Status

The first causal automatic multi-human loop is now operational:

```text
pre-cut Human3R tracks
-> one V14 shadow correction at the first post-cut frame
-> fresh-reset Human3R post-cut reconstruction
-> learned coarse Boundary B0
-> B0-assisted anonymous identity assignment
-> frozen per-human explicit geometry
-> uniform multi-human consensus
-> ONE fixed shared Boundary for the full post-cut segment
```

This is an initial feasibility baseline, not a finished deployable method. The
controlled same-visibility path is working; identity ambiguity, variable human
count, and final Boundary accuracy still need improvement.

## 2. Checkpoint Provenance

The active V14.1 checkpoint is:

```text
/dev/shm/movie3r_v14_1/
v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth
```

Its actual initialization chain is:

```text
original Human3R
-> formal V9 mixed AvatarReX + THuman training
-> V14.1 event-only fine-tuning on one AvatarReX lbn1_1192 event
```

V14.1 therefore does not learn cross-view geometry from one sample from
scratch. It inherits the Human3R representation and V9 correction prior, then
adapts that correction to the first-post-cut event contract.

## 3. Current Causal Runtime

### 3.1 Pre-cut state

Human3R runs normally inside the current shot. Native within-shot human tracks
provide stable anonymous track labels such as `track_0`, `track_1`, and
`track_2`.

### 3.2 Cut event

The current experiments receive an explicit cut index. An automatic cut
detector is not part of this baseline.

At the cut, two paths are evaluated:

```text
shadow path:
    pre-cut history + first post-cut event frame
    V14 correction enabled
    state discarded after B0 is read

raw path:
    first post-cut frame with fresh Human3R state
    state retained for all later post-cut frames
```

The one-shot coarse Boundary is:

```text
B0 = C_shadow_post @ inverse(C_raw_post)
```

No future post-cut frame contributes to `B0`.

### 3.3 Automatic identity association

The first post-cut detections are anonymous. The current controlled matcher
uses the same normalized root-plus-torso cost before and after B0:

```text
direct:
    pre geometry vs raw post geometry

B0-assisted:
    pre geometry vs B0-mapped post geometry

assignment:
    Hungarian one-to-one matching
```

GT identity is used only for audit. It is not used in the cost matrix,
Hungarian assignment, per-human candidate generation, or Boundary solve.

The current matcher assumes the same detected human set on both sides. It does
not yet implement dustbin, entry/exit, or unmatched tracks.

### 3.4 Frozen multi-human geometry

After automatic identity assignment, every accepted track independently
produces one frozen Phase-2 candidate:

```text
Fixed Explicit initializer
-> local pointmap refinement
-> V16 torso residual with 20 degree bound
-> root-anchor translation
-> (R_i, t_i)
```

The final multi-human Boundary is unchanged:

```text
R = equal-weight SO(3) mean(R_i)
t = arithmetic mean(t_i)
B = [R, t]
```

In the current automatic-ID ladder, `B0` is used to make identity matching
well-posed. The frozen multi-human solver then estimates the final Boundary
from the matched raw-reset humans. It does not read GT identity or GT camera.

### 3.5 Segment propagation

The final `B` is estimated once and left-multiplied into every post-cut:

- camera;
- world pointmap;
- human mesh through the shared camera/world gauge.

The raw-reset recurrent state is the only committed state. The shadow state is
never committed. The same fixed Boundary is used for the full segment.

## 4. Current Evidence

### 4.1 Controlled identity matching

| Sequence | Eligible cuts | Direct all-correct | B0 all-correct |
|---|---:|---:|---:|
| `three` | 41 | 46.3% | 100.0% |
| `dance` | 61 | 65.6% | 100.0% |
| `box` | 78 | 65.4% | 98.7% |

The one remaining controlled failure is:

```text
box_t0630_c0_c3_k8
direct: 0/2
B0:     0/2
GT camera mapping: 0/2
```

This is a motion/local-human-reconstruction identity ambiguity, not a coarse
camera-alignment failure.

### 4.2 Long 24-frame causal visual probes

Each probe contains four pre-cut frames and twenty post-cut frames. Identity
IDs remain stable from the first through the last displayed post-cut frame.

| Case | Humans | Direct ID | B0 ID | B0 camera error | Final multi camera error |
|---|---:|---:|---:|---:|---:|
| `dance_t0600_c1_c4_k1` | 2 | 0/2 | 2/2 | 0.427 m / 2.24 deg | 0.568 m / 4.93 deg |
| `box_t0470_c1_c4_k8` | 2 | 0/2 | 2/2 | 0.390 m / 2.81 deg | 0.466 m / 5.40 deg |
| `three_t0900_c3_c4_k0` | 3 | 0/3 | 3/3 | 0.113 m / 3.85 deg | 0.408 m / 2.26 deg |

These probes establish that the automatic-ID multi-human path can run for a
full causal segment. They also expose the next accuracy problem: uniform
multi-human geometry improves identity/layout and can improve rotation, but
unconditionally replacing the learned B0 translation makes camera translation
worse in all three probes.

## 5. What Is and Is Not Solved

### Established

- V14.1 produces a useful cross-dataset coarse shot Boundary.
- B0 must run before geometry-based cross-shot identity matching.
- B0 resolves wide-view permutation failures in the controlled two- and
  three-person cases.
- Correct automatic identity can drive the frozen uniform multi-human solver.
- One shared Boundary can be propagated through a longer reset-state segment.
- The runtime remains causal, one-shot, streaming, and fixed-budget for two or
  three people.

### Not established

- automatic cut detection;
- entry, exit, missed detection, and reappearance;
- dustbin and precision-first acceptance;
- appearance/beta support for motion-induced identity ambiguity;
- a broadly trained V14.1 checkpoint;
- final multi-human translation that consistently improves over B0;
- multi-cut state evolution across `A -> B -> C`;
- end-to-end frozen evaluation including every candidate cut;
- robust cross-dataset catastrophic-rate guarantees.

The controlled evaluation excluded changing detection sets:

```text
three: 22/63 excluded
dance: 29/90 excluded
box: 12/90 excluded
```

Those cuts are unresolved, not successful results.

## 6. Main Technical Diagnosis

The current pipeline contains two different useful estimates:

```text
B0:
    strong shared camera-gauge estimate
    good enough to disambiguate identity

uniform multi-human Boundary:
    strong human continuity/layout estimate
    useful rotation ambiguity reduction
    translation can inherit Human3R root-depth bias
```

The next version should not assume that the old Phase-2 translation remains
optimal after a much stronger learned B0 is available. Previous experiments
also showed that blindly adding V12/V16 residuals to an accurate B0 can
over-correct it.

## 7. Prioritized Next Work

### Step 1: Freeze this baseline

Keep the current runner and report contract as the reproducible baseline:

```text
versions/v14/run_v14_autoid_visual_ladder.py
```

Do not change identity and Boundary modules simultaneously in the next
ablation.

### Step 2: Improve V14.1 coverage

Train the same event-only architecture progressively:

```text
one event
-> ten sequences
-> broader AvatarReX/THuman/MVHuman training
```

Keep the original Human3R raw-reset branch frozen and preserve the one-shot
shadow-state contract.

### Step 3: Improve WHO after B0

Add frozen appearance and beta cues after B0, with local pose used only as a
short-term compatibility cue. Add mutual matching, margin, dustbin, and
precision-first acceptance. This targets the `box k=8` failure and the
variable-visibility cuts.

### Step 4: Improve WHERE around B0

Treat B0 as a strong prior and test a bounded multi-human residual rather than
unconditionally replacing its translation. Compare at minimum:

```text
B0 only
B0 rotation + uniform multi translation
bounded multi rotation residual around B0
bounded multi translation residual around B0
full frozen Phase-2 uniform multi
```

Report camera, human root, layout, P90/P95, and catastrophic rate. A new method
must improve visible human/layout consistency without degrading camera tails.

### Step 5: Complete the streaming lifecycle

Add the fixed fallback hierarchy:

```text
>=2 safe identities: multi-human consensus
1 safe identity:     single-human Boundary
0 safe identities:   strongest identity-independent Boundary
```

Then test entry/exit, missed detections, reappearance, and real multi-cut
streams without GT reinitialization.

## 8. Current Conclusion

The multi-human route is worth continuing. The primary architectural ordering
is now supported:

```text
learned causal B0
-> automatic WHO
-> explicit multi-human WHERE
-> ONE shared Boundary
```

The next objective is no longer to prove that multi-human execution is
possible. It is to improve precision, visibility coverage, and final Boundary
accuracy while preserving the causal fixed-budget state split.
