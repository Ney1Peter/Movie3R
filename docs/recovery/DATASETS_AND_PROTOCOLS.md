# Dataset and protocol inventory

Raw datasets are deliberately excluded from Git. The local archive sizes below
allow a restored download to be checked before extraction. Whole-archive hashes
were not recomputed for the three hundreds-of-gigabytes ZIP files; extraction
code validates selected members and ZIP integrity.

## Primary multi-person evaluation

| Dataset/protocol | Local source on frozen server | Size | Frozen evaluation |
|---|---|---:|---|
| EgoBody-CS150 | `/data/wangzheng/iJCV-CODE/data/EgoBody.zip` | 359,069,231,957 bytes | 129 test cases, 43 recordings, 150 frames |
| EgoHumans-CS100 | `/data/wangzheng/iJCV-CODE/data/EgoHuman.zip` | 237,157,095,656 bytes | 90 test cases, 27 captures, 100 frames |
| Harmony4D-CS150 | `/data/wangzheng/iJCV-CODE/data/Harmony4D.zip` | 351,667,292,193 bytes | 88 evaluator-valid cases from the frozen 100-case protocol |

Official paper/project references:

- EgoBody: <https://arxiv.org/abs/2112.07642>
- EgoHumans: <https://arxiv.org/abs/2305.16487>
- Harmony4D: <https://arxiv.org/abs/2410.20294>

The exact splits, selected archive members, and integrity records are encoded
in `versions/v19/egohumans/`, `versions/v20/egobody/`,
`versions/v15/harmony4d/`, and the frozen result artifacts in
`publication/shot3r_reproducibility_20260916/results/`.

## Additional evaluation

| Dataset | Protocol | Code/manifest location | Purpose |
|---|---|---|---|
| AIST++ | CS150, MC150-3, MC150-4 | `versions/v21/aist_singleperson/` | Single-person and repeated-transition supplement |
| MVHuman | MVH150 | `versions/v25/mvhuman_heldout/` | Large-view held-out and OnlineHMR supplement |
| Harmony4D | three-shot MC150 | `publication/bridge3r_iclr2027/multicut/` | Accumulated-transition analysis |

AIST++ should be obtained from its official release. The locally derived card
and checksums are produced by `versions/v21/aist_singleperson/finalize_dataset.py`.
MVHuman input construction is defined by
`versions/v25/mvhuman_heldout/build_protocol.py`; do not substitute a different
frame sampling policy while reusing the recorded metrics.

## Final module training data

The frozen configuration is
`config/train_v14_1_cut_first_cross_source_multihuman_p0.yaml`. Each epoch uses
96 events from each of five sources (480 events total): AvatarReX, THuman,
MVHuman100, MVHuman200, and the MultiHuman camera-supervision set. The exact
manifests are under `versions/v14/cut_first_cross_source/manifests/train96ps/`
and `config/manifests/v14_multihuman_camera_supervision_20260803.json`.

The training-data root is intentionally not reconstructed from filenames. On a
new server, place or link each licensed/downloaded source at the paths supplied
to the frozen config, then audit every manifest entry before training.

## Extraction policy

Do not extract the complete large archives by default. The formal runners stage
only the members referenced by a frozen runtime/evaluator manifest. Keep raw
archives read-only, use a disposable staging directory, and delete staging only
after aggregate metrics and provenance files have been copied into the
reproducibility snapshot.

