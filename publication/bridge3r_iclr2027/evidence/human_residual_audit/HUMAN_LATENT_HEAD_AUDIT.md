# Human-latent head formal-90 audit

## Verdict

`--ablation-disable-human-latent-head` is correctly parsed and switches off
`model.enable_v8_human_latent_corr`.  The frozen checkpoint contains nonzero
weights for this head, and a causal event-frame probe confirms that the head
injects a nonzero residual into the human token and changes decoded SMPL
parameters.

However, the published EgoHumans formal-90 transaction does not consume the
event-frame human prediction.  It uses only the event shadow camera to form
the coarse gauge transform, then obtains every post-cut person from a separate
no-event clean-reset rollout.  Consequently, enabling or disabling the learned
human-latent head cannot change any packed formal result under the current
transaction.  The existing `Full token, human residual off` row is therefore
not a meaningful efficacy ablation and must not be used to claim that this
head improves the reported benchmark metrics.

## Code-path evidence

1. The queue passes `--ablation-disable-human-latent-head` in
   `versions/v19/egohumans/run_formal_ablation_queue.py:50-53`.
2. `configure_inference_ablation` assigns
   `enable_v8_human_latent_corr = not args.ablation_disable_human_latent_head`
   in `versions/v15/harmony4d/run_harmony_case.py:304-340`.
3. The model checks this flag and, when active, applies the residual before the
   Human3R human head in `src/dust3r/model.py:3128-3155` and
   `src/dust3r/model.py:4035-4057`.
4. `run_transaction` executes an event shadow prefix and a no-event raw-post
   rollout separately (`run_harmony_case.py:832-848`).
5. `compose_transaction` forms `b0_transform` from
   `shadow_first_post["camera"]`, but maps the people from `raw_post`
   (`run_harmony_case.py:688-704`).  It never reads
   `shadow_first_post["people"]`.

Thus the learned human correction exists inside the discarded shadow person
prediction, while the final transaction contains the unaffected reset people.

## Formal-artifact evidence

- All 90 head-off runtime records contain the requested command-line flag.
- All 90 record
  `inference_ablation.explicit_human_latent_head_enabled = false`.
- For all 90 matched cases, the full and head-off packed NPZ files have the
  same recorded SHA-256 digest (90/90 byte-identical).
- After normalizing only the method label, all 90 case-level metric rows are
  exactly identical, not merely equal after table rounding.

One instant read-only verification is:

```bash
Movie3R/.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('Movie3R/output/bridge3r_egohumans_ablation_v1')
def rows(route):
    out = {}
    for path in (root / route / 'test/predictions').glob('*/*.runtime.json'):
        payload = json.loads(path.read_text())
        out[payload['record']['case_id']] = payload['cache_sha256']
    return out
full = rows('formal90_full_replay')
off = rows('formal90_human_residual_off')
common = set(full) & set(off)
print(len(common), sum(full[key] == off[key] for key in common))
PY
```

Expected output: `90 90`; runtime is below one second and no GPU is used.

## Three-case tensor probe

The audit uses one frozen test capture and three preregistered angle strata.
Each probe reads exactly 50 pre-cut frames and the first post-cut frame.  It
does not read GT, calibration, evaluator identities, or future post-cut RGB.

| Stratum | Angle | People | Injected token residual L2 | Max SMPL translation change | Max camera change | Final formal NPZ |
|---|---:|---:|---:|---:|---:|---|
| Small | 26.48 deg | 3 | 1.6597 | 0.6967 | 0 | byte-identical |
| Medium | 80.25 deg | 4 | 0.5710 | 0.2868 | 0 | byte-identical |
| Extreme | 179.63 deg | 3 | 1.4130 | 0.3803 | 0 | byte-identical |

For all three cases:

- the event route and gate are both exactly one;
- the residual reported by the head matches the observed on-minus-off human
  token within `4.2e-7` maximum absolute error;
- decoded human parameters change, confirming that the runtime switch and
  learned head are functional;
- the event camera is unchanged, as expected for a human-only head;
- the final full/head-off formal caches remain byte-identical, confirming that
  the changed event human output is discarded later by transaction assembly.

Machine-readable evidence is in `tensor_probe/summary.json`.  Each case also
has a compact `*.tensor_probe.npz` containing on/off human tokens, the reported
residual and gate, decoded SMPL fields, camera pose, and the two formal-cache
hashes.

Reproduction command (about 90 seconds of forward time plus model loading and
image preprocessing on one 46-GB GPU):

```bash
Movie3R/.venv/bin/python \
  Movie3R/publication/bridge3r_iclr2027/audit_human_latent_head.py \
  --extracted-root \
  data/EgoHuman_human_residual_audit_v1/staging/fencing__002_fencing-002 \
  --output-dir \
  Movie3R/publication/bridge3r_iclr2027/evidence/human_residual_audit/tensor_probe \
  --device cuda:1
```

## Experimental decision

Do **not** rerun all 90 cases with the current head-off flag: the composition
algebra guarantees the same output, and the completed run already verifies
that fact exactly.

For the current paper version, remove this row from the efficacy table and do
not attribute benchmark gains to the learned human-latent head.  A future
meaningful ablation first requires an explicitly specified transaction that
consumes the corrected event person proposal (for example, a bounded person
residual propagated from the shadow event into the clean-reset post-cut
people).  Validate that new route on 1--3 cases before committing to a
formal-90 run.  It constitutes a method change and must not be silently mixed
with the currently locked Bridge3R results.
