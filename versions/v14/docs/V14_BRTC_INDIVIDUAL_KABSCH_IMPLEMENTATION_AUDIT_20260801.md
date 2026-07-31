# V14 BRTC individual Kabsch implementation audit

Date: 2026-08-01

## Audit conclusion

Final implementation verdict:

`IMPLEMENTATION_AUDIT_PASS_WITH_EXPLICIT_DUAL_STATE_CONTRACT`

The person-local Kabsch rotation itself, its SO(3) direction, accepted-person
application, rejected/unmatched fallback, camera invariance, native-root
invariance, causal propagation, and frozen-probe parity are now internally
consistent. The deployable function can reproduce the EgoHumans evaluator only
when the caller explicitly maintains two causal states:

1. `pre_people`: the unrotated frozen-BRTC translation reference state;
2. `orientation_pre_people`: the already-rotated causal pose state used only by
   Kabsch.

This is a required deployment contract, not an optional evaluator convenience.
Before this API was added, feeding the previous complete candidate state back
through the one-state runtime changed the next cut's BRTC root by as much as
`0.0090556 m` relative to frozen BRTC v1. Feeding only the unrotated BRTC state
avoided that root drift but discarded the previous cut's causal orientation.
The old API therefore could not simultaneously reproduce both advertised
properties and was `BLOCKED_AS_STANDALONE_DEPLOYABLE_RUNTIME`.

The dual-state repair closes this implementation gap without changing the
frozen orientation policy or any reported metric.

## Geometry convention

The implementation uses row-vector points. For corresponding root-centred
torso points `source` and `target`, Kabsch estimates `R` such that:

```text
source @ R.T -> target
```

With covariance `source.T @ target = U S V.T`, the code uses
`R = V U.T`, with the final singular vector flipped when needed to ensure
`det(R) = +1`. A known nontrivial SO(3) unit test recovers the planted rotation
to `1e-12`, including the forward mapping `source @ R.T == target`.

For an accepted person, the bounded rotation is applied around the already
BRTC-corrected native root:

```text
p' = (p - root) @ R.T + root
```

Optional orientation metadata follows the matching column-vector convention:

```text
torso'         = R @ torso
root_rotation' = R @ root_rotation
```

The native root is copied exactly and camera matrices are never mutated.

## Runtime path after repair

At each shot boundary the deployable function now performs the complete path:

```text
unrotated translation pre-state
  -> frozen BRTC-LC v1
  -> accepted/rejected decision and translated root

rotated causal orientation pre-state + current B0 post pose
  -> torso4 Kabsch
  -> frozen angle/fraction/observable gate
  -> rotate only accepted person's body-local geometry

output
  -> BRTC root + optional person-local rotation
  -> same shift/rotation propagated through the post shot
```

The Ego evaluator now obtains its first-post candidate directly from
`refine_matched_people_orientation_kabsch(...)` with both state arguments. An
independent frozen-BRTC replay remains only as a parity oracle. The evaluator no
longer constructs its reported first-post candidate by manually calling BRTC
and then manually calling the orientation helper.

The runtime checks that both state lists have the same length. When track IDs
are present, it also checks that `global_track_id` and `native_track_id` agree at
each matched pre index. Omitting `orientation_pre_people` preserves the original
single-state behavior for backward compatibility.

## Fallback and propagation audit

- BRTC-rejected matched people: exact B0 `root`, `joints`, `vertices`, `torso`,
  and `root_rotation`.
- Unmatched post people: exact B0 for the same fields.
- Accepted people: only frozen BRTC translation plus the bounded root-centred
  SO(3) rotation.
- Camera: exact B0.
- Native stored root: exact frozen BRTC v1.
- Later post-shot frames: repeat the boundary person's frozen translation and
  rotation action by native track ID.
- Second cut: BRTC reads the unrotated translation reference, while Kabsch reads
  the rotated last-pre candidate.

On the EgoHumans replay, `6/7` second-cut tracks have a nonzero inherited
orientation difference, proving that the orientation state is actually causal
rather than silently reset.

## Verification results

Commands:

```bash
.venv/bin/python -m py_compile \
  versions/v14/b0_person_triangulation_orientation_kabsch.py \
  versions/v14/probe_brtc_global_orientation_kabsch.py \
  versions/v14/eval_brtc_global_orientation_kabsch_egohumans.py \
  versions/v14/tests/test_b0_person_triangulation_orientation_kabsch.py

.venv/bin/python -m pytest -q \
  versions/v14/tests/test_b0_person_triangulation.py \
  versions/v14/tests/test_b0_person_triangulation_orientation_kabsch.py

.venv/bin/python \
  versions/v14/probe_brtc_global_orientation_kabsch.py --phase validate

.venv/bin/python \
  versions/v14/eval_brtc_global_orientation_kabsch_egohumans.py --self_test

.venv/bin/python \
  versions/v14/eval_brtc_global_orientation_kabsch_egohumans.py
```

Results:

- compile: pass;
- unit tests: `12 passed`;
- self-test: pass;
- MultiHuman three-offset1 deployable/probe parity: exact zero for root, joints,
  vertices, rotation, applied decision, camera, and rejected fallback;
- Ego causal accepted-person deployable/probe parity: exact zero;
- frozen BRTC first-frame geometry parity: `8.882e-16` maximum absolute delta;
- deployable first frame versus shot propagation: `8.882e-16` maximum absolute
  delta;
- rejected/unmatched exact-B0 maximum change: `0`;
- native root versus frozen BRTC v1 maximum change: `0`;
- camera versus B0 maximum change: `0`;
- rotation maximum orthogonality error: `3.331e-16`;
- rotation maximum determinant error: `4.441e-16`;
- frozen policy canonical checksum validation: pass.

The `8.882e-16` values are normal float64 roundoff and are below the declared
`1e-12` parity threshold.

## Scientific result remains unchanged

The implementation repair does not promote the method to a strict winner.
The reproduced EgoHumans candidate remains:

```text
W-MPJPE / WA-MPJPE                 312.769 / 200.029 mm
pelvis MPJPE / MPVPE               101.526 / 119.928 mm
fixed-world joint / vertex         383.933 / 383.791 mm
world root / joint Accel           115.698 / 123.167 mm/frame^2
mapped-pelvis fixed-root delta     +0.034 mm vs BRTC v1
native stored-root max delta       0
camera max delta                   0
joint/vertex harm over 5 cm        0%
```

Therefore the scientific decisions stay:

```text
strict exact-non-regression:
NO_GO_GLOBAL_ORIENTATION_KABSCH_EGOHUMANS

secondary explicit 0.1 mm mapped-pelvis proxy tolerance:
QUALIFIED_GLOBAL_ORIENTATION_KABSCH_CANDIDATE
```

The second label is only a qualified engineering candidate under an explicitly
declared proxy tolerance. It is not the original strict winner rule, is not the
official unpublished Multi-THuMBS protocol, and must not be presented as the
final fine-alignment solution.

## Hashes requiring manifest refresh

The frozen numeric policy and MultiHuman validation evidence did not change.
The implementation/evaluator hashes changed because of the required runtime
repair and added audits:

```text
runtime_sha256
92c7f83a7388ba2fbc336ae77a8cf9a0f9cf1a55b26f18ebbe86f8a6597bd481

probe_sha256
c12a29c5483eb8865d237d4e00c462f7b53acbc7e61ff03a63c6971eeeba9f71

egohumans_evaluator_sha256
8af6eb985d0a53690ab68741683edf7997d12eac4d5e98d05457dc8e068c4790

policy_file_sha256 (unchanged)
cd51d67ef2779f7959b08a445ae9879fc9244285b099fd72a2348adc12718111

multihuman_validation_sha256 (unchanged)
8d9c14056e6642e03fc38044b147d29a39409ed80d2213ba9eb2c7813ef7b8eb

egohumans_report_sha256
89551e227110a328f38d04d34bcd4db0aa96071ea6da45b1ff5f4a771742156f
```

The report hash changed only because the report now records the explicit
dual-state runtime and first-frame propagation parity fields; the evaluation
metrics and dual scientific decision are unchanged.
