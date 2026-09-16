# Shot3R

Shot3R performs streaming reconstruction of cameras, visible people, and scene
geometry across shot transitions. The repository directory retains the
historical name `Movie3R`, but the current paper method is **Shot3R**.

At a transition, the previous recurrent state is read temporarily to infer the
inter-shot relation, while a separately reinitialized state becomes the only
state propagated through the new shot. Geometry-based person association and
a shared world transform place the new-shot outputs in the common coordinate
frame without propagating the old recurrent state.

## Start here

- Disaster recovery and server migration: [`docs/recovery/README.md`](docs/recovery/README.md)
- Dataset and protocol inventory: [`docs/recovery/DATASETS_AND_PROTOCOLS.md`](docs/recovery/DATASETS_AND_PROTOCOLS.md)
- Model-weight manifest: [`docs/recovery/WEIGHTS.md`](docs/recovery/WEIGHTS.md)
- Experiment/result map: [`docs/recovery/EXPERIMENTS.md`](docs/recovery/EXPERIMENTS.md)
- External baseline commits and patches: [`docs/recovery/BASELINES.md`](docs/recovery/BASELINES.md)
- Paper-facing method/code map: [`publication/bridge3r_iclr2027/METHOD_TO_CODE_FACT_AUDIT_20260829.md`](publication/bridge3r_iclr2027/METHOD_TO_CODE_FACT_AUDIT_20260829.md)

## Current reproducibility snapshot

The lightweight snapshot under
[`publication/shot3r_reproducibility_20260916/`](publication/shot3r_reproducibility_20260916/)
contains frozen numerical summaries, per-case metrics used by the paper,
training logs, runtime measurements, traditional-registration controls, and
the audited causal cut detector. It deliberately excludes raw datasets, full
per-frame predictions, virtual environments, licensed body models, and
multi-gigabyte reconstruction checkpoints.

The final Shot3R checkpoint is identified by SHA-256
`de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265`.
Consult the weight manifest before downloading or replacing any checkpoint.

## Main implementation

| Component | Location |
|---|---|
| Recurrent backbone and typed alignment representations | `src/dust3r/model.py` |
| Training and loss implementation | `src/train.py`, `src/dust3r/losses.py` |
| Frozen final training configuration | `config/train_v14_1_cut_first_cross_source_multihuman_p0.yaml` |
| Shot-transition detector | `versions/v14/causal_image_detector.py` |
| Streaming state transition and shared transform | `versions/v20/egobody/deployment_runtime.py` |
| Cross-shot identity association | `versions/v19/egohumans/causal_identity.py` |
| Dataset-independent boundary transaction | `publication/bridge3r_iclr2027/bridge3r.py` |
| Traditional-registration controls | `experiments/registration_baselines/` |

Some directories and identifiers retain the earlier `bridge3r` or `Movie3R`
names for path compatibility. They are historical implementation names, not
separate paper methods.

## Environment

The frozen server used Python 3.10.19, PyTorch 2.4.0+cu124, CUDA 12.4, and
eight NVIDIA L20 GPUs. Recreate the environment from
`requirements_Movie3R.txt` and follow `docs/recovery/ENVIRONMENT.md`; do not
copy the local `.venv` directory.

## Data and weights

No raw benchmark data or multi-gigabyte checkpoint is stored in Git. Download
datasets from their official sources and restore the paths documented in
`docs/recovery/DATASETS_AND_PROTOCOLS.md`. Licensed SMPL/SMPL-X assets must be
obtained under their original licenses.

## License

This project extends Human3R/CUT3R-related code and retains the applicable
upstream license requirements. Dataset, body-model, and external-baseline
licenses remain separate.
