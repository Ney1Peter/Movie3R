# OnlineHMR extension evidence

This directory contains paper-facing LaTeX fragments generated from the
fixed-denominator OnlineHMR extension campaign under
`data/OnlineHMR_work_v1/extensions/formal/`.  The numerical fragments are not
manually transcribed.

The campaign covers five frozen protocols: Harmony4D three-shot (4 cases),
AIST++ CS150 (100), AIST++ MC150-3 (100), AIST++ MC150-4 (100), and MVHuman
MVH150 (50).  Runtime inference receives ordered RGB frames only.  Failed or
structurally invalid cases remain in the Completion and Coverage denominators;
conditional geometric errors always retain their finite support.

Generation command, run from `Movie3R/` after all five protocol states are
complete:

```bash
.venv/bin/python \
  publication/bridge3r_iclr2027/onlinehmr/finalize_onlinehmr_extension_campaign.py \
  --campaign-root ../data/OnlineHMR_work_v1/extensions

.venv/bin/python \
  publication/bridge3r_iclr2027/onlinehmr/export_onlinehmr_extension_latex.py \
  --campaign-root ../data/OnlineHMR_work_v1/extensions \
  --output-dir ../ICLR-paper/bridge3r_iclr2027/versions/\
v034_20260906_onlinehmr_extensions/manuscript/artifacts/onlinehmr_extensions
```

The accompanying `formal/` subdirectory is a release copy of the aggregate,
case-level, boundary-level, and paired JSON/CSV evidence used to generate the
tables.  Absolute paths stored in machine-readable provenance remain outside
the rendered manuscript and are not required by Overleaf.
