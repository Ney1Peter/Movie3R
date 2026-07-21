# V15 Wide-Baseline Boundary Bridge

V15 uses frozen VGGT-1B to test wide-baseline coarse localization before Human3R metric refinement on the same 180 real cross-camera cuts.

The experiment proves that wide-baseline vision materially expands the capture basin: on Fixed initial rotation errors above 60 degrees, 1+1 VGGT Coarse reduces rotation from `75.3` to `21.8` degrees and catastrophic failure from `100%` to `30%`. It also reduces catastrophic failure to `31.2%` on MVHuman100 and `25.0%` on MVHuman200.

The complete Hybrid does not pass. Human3R correspondence metrification and ICP erase most MVHuman gains, 3+3 is worse than 1+1, and AvatarReX/THuman regress. Do not distill a full pose or metric-SE(3) Shot Bridge yet. Retain VGGT only as a possible coarse rotation/direction teacher and study metrification separately.

Main report:

- [V15 Wide-Baseline Boundary Bridge Feasibility](V15_WIDE_BASELINE_BOUNDARY_BRIDGE_FEASIBILITY_20260719.md)

Code:

```text
scripts/v15_wide_baseline_boundary_bridge_candidates.py
scripts/v15_wide_baseline_boundary_bridge_eval.py
```
