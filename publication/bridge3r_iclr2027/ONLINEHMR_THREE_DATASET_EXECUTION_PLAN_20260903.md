# OnlineHMR three-dataset execution plan

Date: 2026-09-03  
Status: active long-running goal  
Method role: strong external camera--human reconstruction baseline

## 1. Target evidence

Run the official OnlineHMR multi-person path on exactly the final BRIDGE3R
test cases:

| Dataset | Final cases | Independent aggregation unit | Viewpoint strata | Frames |
|---|---:|---|---|---:|
| EgoBody | 129 | 43 recordings | small / medium / extreme | 150 |
| EgoHumans | 90 | 27 captures | small / medium / large / extreme | 100 |
| Harmony4D | 88 | 25 captures | small / medium / large / extreme | 150 |

The OnlineHMR row must use the same ordered RGB frames, cut location, camera
pair, GT visibility rules, common SMPL-6890/SMPL24 topology, 2 m native-camera
Hungarian threshold, failure denominator and dataset-level macro aggregation as
the existing methods. No GT camera, identity, boxes, masks, depth, person count
or cut labels are exposed to inference.

Report W-MPJPE, WA-MPJPE, MPJPE, PA-MPJPE, MPVPE, IDF1, coverage, detection
precision, camera ATE-Sim3/ATE-SE3, camera RPE, boundary camera seam, human
seam, completion and runtime wherever the native outputs support them.
Conditional human errors must always be accompanied by coverage and
availability.

## 2. Execution phases

1. Rebuild immutable OnlineHMR runtime/evaluator manifests by exact case-ID
   intersection with the retained final evidence. Verify EgoBody's reconstructed
   runtime manifest against its historical SHA-256.
2. Add a benchmark-only image-folder entry to OnlineHMR so it consumes the
   exact staged JPEGs without video re-encoding. Keep model logic and weights
   unchanged; isolate DEVA intermediates per case for safe parallel execution.
3. Convert native SMPL predictions and the independent MASt3R-SLAM camera
   trajectory to the common prediction cache. Preserve native track identities;
   no GT-assisted track repair or post-cut geometric correction is allowed.
4. Run a 12-case availability pilot: four cases per dataset, stratified by
   viewpoint. EgoBody contributes small, medium, and two independent extreme
   cases because its frozen protocol has no separate `large` label.
5. Pilot gate: at least 10/12 complete automatically; both shots have finite
   predictions and non-zero evaluated coverage; camera/world convention checks
   pass; one configuration works across all strata. Compatibility fixes may be
   made globally, but no Test-metric tuning or per-case exceptions are allowed.
6. Freeze code commit, environment, weight manifest, runtime options, case
   manifests and failure policy.
7. Run full inference disk-bounded by archive group. Use at most five idle GPUs,
   normally one OnlineHMR case per GPU; retain raw outputs, logs and checksums,
   while reclaiming only reproducible staging data after evaluation.
8. Aggregate by the existing dataset-specific statistical unit, produce
   all-view and angle-stratified tables, paired comparisons, bootstrap
   uncertainty, availability and failure inventories.
9. Add OnlineHMR to the external-baseline table and supplementary details only
   after audit. It is not part of the same-backbone ablation table. Describe its
   semi-online tracker/global camera backend accurately rather than asserting
   BRIDGE3R's strict causal contract.

## 3. Current archive audit

- `EgoHuman.zip`: central directory and Zip64 end records are present.
- `Harmony4D.zip`: central directory and Zip64 end records are present.
- `EgoBody.zip`: incomplete as uploaded on 2026-09-03. Its size is
  253,431,283,712 bytes, while the local header declares
  `kinect_color.zip` alone as 352,747,775,690 uncompressed bytes and
  352,801,606,863 packed bytes. Both `unzip` and `7z` report a missing/corrupt
  central directory. The available file ends inside that payload, so its
  formal four pilot cases cannot be staged.

Work therefore proceeds immediately with the eight EgoHumans/Harmony4D pilot
cases and all method-independent adapter tests. The four frozen EgoBody pilot
cases are queued without changing selection; they run after a complete archive
is restored. The incomplete archive must not be silently treated as valid data.

## 4. Reproducibility and stopping rules

- Existing outputs are reused only when case ID, manifest hash, code commit,
  weight manifest and command match.
- A crash or empty output remains a zero-coverage case; it is never removed.
- Any retry retains the failed attempt and reason. Only infrastructure failures
  may be retried automatically; prediction quality is not a retry criterion.
- Pilot or full results never alter the formal case set, matcher, cut index or
  aggregation denominator.
- Paper values are generated from machine-readable aggregate artifacts, not
  manually copied from terminal output.

