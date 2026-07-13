# V10 End-to-End Evaluation Protocol

Date: 2026-07-08

## Goal

Use one fixed protocol to compare V10 against previous routes after each
alignment or detector variant finishes.

The key question is not only whether B frames move closer to GT.  The system
also has to avoid damaging stable continuous frames.  Therefore the evaluation
separates two decisions:

1. Boundary detection: should the current frame start a new local segment?
2. Segment alignment: if yes, can the new local segment be attached to the
   historical global coordinate system?

## Methods To Compare

The final table should include these rows when their outputs are available:

- Original Human3R local output.
- V9 correction-token model.
- V10 oracle-boundary alignment.
- V10 predicted-boundary alignment.
- V10 ablations:
  - no boundary reset/alignment;
  - wrong or over-triggered boundary;
  - detector-only variants.

Current available automatic rows:

- `raw`: strict original Human3R local-reset output saved inside each V10
  alignment probe.
- `aligned`: V10 oracle-boundary aligned output from the same probe.
- detector summaries from image-only, Human3R-output, and combined probes.

## Core Metrics

Alignment metrics from `metrics_flat.csv`:

- `cam_rot_deg`: post-boundary camera rotation error.
- `cam_trans_m`: post-boundary camera translation error.
- `human_post_m`: post-boundary human joint/anchor error.
- `Amean_B0_m`: distance from historical A human anchor to B0.
- `Amean_B1_m`: distance from historical A human anchor to B1.
- `BB_m`: B0-B1 internal human consistency.

Detector metrics:

- `F1`: overall boundary detection balance.
- `precision`: how many predicted changes are true changes.
- `recall`: how many real changes are detected.
- `stable FPR`: false positive rate on stable frames. This matters most for
  no-op safety because false positives will reset or align frames that should
  have stayed original.

## Report Builder

Script:

```bash
.venv/bin/python scripts/v10_build_end_to_end_eval_report.py
```

Default output:

```text
output/v10_end_to_end_eval/report.md
output/v10_end_to_end_eval/alignment_summary.csv
output/v10_end_to_end_eval/alignment_by_source.csv
output/v10_end_to_end_eval/detector_summary.csv
```

Extra runs can be added without editing the script:

```bash
.venv/bin/python scripts/v10_build_end_to_end_eval_report.py \
  --alignment_run my_v10_run:/path/to/run_dir \
  --detector_run my_detector:/path/to/method_results.csv
```

## Medium Quick Alignment

A medium V10 W5 run is used to get a faster trend before the full GPU7 large run
finishes.

Launcher:

```bash
GPU=6 SAMPLES_PER_SOURCE=200 STEPS=2500 \
  bash scripts/training/run_v10_static_alignment_4source_medium_w5.sh --start
```

Default output:

```text
output/v10_static_alignment_probe/medium_4source_angle60_w5_s200
```

After it finishes, regenerate the report:

```bash
.venv/bin/python scripts/v10_build_end_to_end_eval_report.py
```

## Reading The Results

The V10 alignment route is considered promising only if:

- post-boundary camera and human errors improve versus raw local Human3R;
- B0/B1 internal consistency does not get worse;
- by-source metrics do not reveal one source being consistently damaged;
- predicted-boundary evaluation stays close to oracle-boundary evaluation;
- detector stable FPR is low enough that continuous frames are not frequently
  over-reset.
