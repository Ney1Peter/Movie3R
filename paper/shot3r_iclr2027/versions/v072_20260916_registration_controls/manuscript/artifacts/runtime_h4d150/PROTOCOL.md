# Seven-method runtime protocol (H4D-CS150)

## Purpose

This benchmark measures the practical latency of the complete prediction path
needed by each released method.  It is intended for the Shot3R ICLR 2027
efficiency comparison, not for selecting a method or tuning a threshold.

## Frozen inputs

Five 150-frame RGB streams are fixed before timing.  Each stream contains two
temporally adjacent 75-frame monocular shots and one camera change.  The set
covers five Harmony4D actions and was selected without inspecting runtime:

| Manifest line | Action | Viewpoint | Frames |
|---:|---|---|---:|
| 2 | hugging | large | 150 |
| 5 | grappling2 | extreme | 150 |
| 33 | sword2 | extreme | 150 |
| 69 | mma4 | extreme | 150 |
| 73 | mma5 | extreme | 150 |

The immutable case identifiers and local RGB directories are recorded in
`selected_cases.json`.  Every directory contains the ordered files
`000000.jpg` through `000149.jpg`.

## Timing contract

- Hardware: one NVIDIA L20 per run; no multi-GPU inference.
- Configuration: each method's released inference configuration used for the
  paper's accuracy experiment.  Rendering and qualitative visualization are
  disabled.
- Included: model construction/checkpoint loading and every method-specific
  component required to produce its native prediction, including detection,
  tracking, camera recovery, temporal/global optimization, and native result
  serialization when these are part of the released entry point. The retained
  PromptHMR and TRACE runners also include their lightweight per-case hardlink
  staging; no image decoding or re-encoding is performed in that step.
- Excluded: dataset download/archive extraction, metric evaluation,
  paper-format conversion, and rendering.
- Unit: accumulated wall-clock seconds from launching the required prediction
  stage process or processes until the native prediction has been saved. When
  an official release exposes multiple required stages (as for JOSH), every
  stage includes its own process launch and model loading and the stage times
  are summed.
- Aggregation: all five clips have equal length, so the reported throughput is
  the micro-average `FPS = 750 / sum(runtime_seconds)`.  We additionally report
  mean and standard deviation of seconds per 150-frame clip.
- Failures are not silently omitted.  A method must produce a native result on
  all five inputs to receive a numerical FPS.

This wall-clock FPS is distinct from the controlled reconstruction-component
benchmark in the appendix, which excludes input preparation and transition
detection.  The two values must not be mixed.

## Existing-run reuse

PromptHMR, OnlineHMR, TRACE, and JOSH were already executed on these exact RGB
streams on NVIDIA L20 GPUs using the configurations evaluated in the paper.
Their immutable per-case runtime records are audited and aggregated directly;
rerunning them would only repeat hours of identical inference.  Human3R and
Shot3R are rerun through dedicated single-method entry points because the old
Harmony4D runner jointly materialized multiple ablations and therefore did not
provide an unbiased process-level time.  TRAM is run with its official three
prediction stages (camera/track estimation followed by VIMO human recovery;
rendering is excluded).
