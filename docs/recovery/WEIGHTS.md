# Weight manifest

All paths are relative to the `Movie3R` repository. Hugging Face URLs can be
added after the external upload; the checksums below are the immutable
identifiers.

| Artifact | Required for | Restore path | Bytes | SHA-256 | Git status |
|---|---|---|---:|---|---|
| Final Shot3R full checkpoint | Paper method inference | `output/v14_cut_first_cross_source/v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth` | 4,930,639,378 | `de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265` | External/Hugging Face |
| V9 60-hour initializer | Reproducing final six-epoch training | `checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth` | 4,831,184,406 | `3fb2799420f7fd3caa63a47c9cde73090a6f93383520363484eb5158e446fceb` | External/Hugging Face |
| Original Human3R 896L | Strict Human3R baseline | `src/human3r_896L.pth` | 4,670,554,642 | `1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377` | External/upstream |
| Causal transition detector | Automatic boundary proposal | `publication/shot3r_reproducibility_20260916/weights/SELECTED_MODEL.pt` | 24,910 | `cb84b0da620878515e94f08b30d757206b41c4de82e2ff4091fe2a6e519e498f` | Tracked in Git |

The final Shot3R checkpoint contains the complete model state and is the only
large learned artifact required for paper-method inference. The V9 checkpoint
is required only to reproduce the final training initialization. The original
Human3R checkpoint is required for strict baseline comparisons and traditional
registration controls.

Local source files before external upload:

```text
/data/wangzheng/iJCV-CODE/Movie3R/output/v14_cut_first_cross_source/v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth
/data/wangzheng/iJCV-CODE/Movie3R/checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth
/data/wangzheng/iJCV-CODE/Movie3R/src/human3r_896L.pth
```

After downloading a weight, verify it with:

```bash
sha256sum <downloaded-file>
python scripts/verify_shot3r_recovery.py --hash-large-weights
```

Do not upload SMPL/SMPL-X licensed model files to the public repository. Obtain
them from the official SMPL family distribution and place them under
`src/models/smpl/` and `src/models/smplx/`.

