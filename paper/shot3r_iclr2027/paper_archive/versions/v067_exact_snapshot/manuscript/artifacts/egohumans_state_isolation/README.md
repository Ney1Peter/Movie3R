# EgoHumans recurrent-state isolation

This directory contains the anonymous, publication-facing summary of a fixed
90-case analysis over 27 EgoHumans captures. The continuous and reinitialized
settings use the same original Human3R checkpoint, identical RGB frames after
the transition, and identical preprocessing. No Shot3R module is enabled.

Each shot uses a static camera. Camera drift is measured relative to the first
prediction after the transition, so any fixed transform between the two shot coordinate
systems cancels. Metrics are averaged within each case, then within capture,
and finally with equal weight over the 27 captures. The primary 16-frame
window and the 10/32/49-frame sensitivity windows were fixed before reading
the reinitialized results.

The complete per-case caches, runtime records, file hashes, evaluator output,
and machine-specific provenance are retained in the version's private audit
directory and are deliberately excluded from the anonymous Overleaf package.
