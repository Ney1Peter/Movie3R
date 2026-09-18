# H4D-CS150 full-system runtime

This directory contains the anonymous, publication-facing record of the
seven-method runtime comparison.

- `PROTOCOL.md` defines the included/excluded operations and aggregation.
- `selected_cases.json` freezes the five 150-frame multi-shot inputs.
- `runtime_summary.csv` contains the seven aggregate rows used in the paper.
- `end_to_end_runtime_table.tex` renders the supplementary table.

FPS is the micro-average over 750 frames. Seconds are the mean and population
standard deviation of the five process-level wall times. Per-case records and
hashes are retained outside the Overleaf package because they contain local
machine paths.
