# V14 BRTC person-to-scene contact cache availability audit

Decision: **NO_GO_CURRENT_CACHE_FOR_PERSON_SCENE_CONTACT_RESIDUAL**.

No model, GPU, image, or GT geometry was loaded. This is an availability audit, not a candidate evaluation.

## Current cache verdict

The current caches are insufficient for a defensible foot-local scene residual. The Ego cache has no pointmap/depth/confidence/scene/foot-visibility fields. The MultiHuman caches retain only a sparse background cloud; the generator explicitly removes every complete human bbox plus an 8% margin, then drops confidence and pixel coordinates. It therefore removes exactly the local support region needed for a foot-contact gate.

| Split | Cases | Post cloud nonempty | Post points median | Foot nearest <=10cm | <=25cm | <=50cm |
|---|---:|---:|---:|---:|---:|---:|
| three_dev_offset0 | 41 | 92.7% | 144.0 | 2.0% | 7.7% | 26.3% |
| three_heldout_offset1 | 42 | 92.9% | 144.0 | 2.3% | 6.3% | 23.7% |
| dance | 61 | 98.4% | 144.0 | 0.1% | 0.7% | 8.9% |
| box | 78 | 100.0% | 220.5 | 7.4% | 29.7% | 54.5% |

## EgoHumans cache

Frames/person-frames: `45` / `121`.
Frame keys: `camera_c2w, camera_name, dataset_frame, people`.
Person keys: `detection_index, gt_label_evaluator_only, joints, native_track_id, root, root_rotation, torso, vertices`.
Scene/visibility fields present: `[]`.

A fallback-only run would be numerically identical to frozen BRTC but have zero contact coverage. That is not a valid GO result, and spatial/Accel effects of a nonzero candidate cannot be reported from this cache.

## Prior failure controls

- V11.2 forced contact with a mean `0.515 m` root correction and caused `112.1 px` mean reprojection displacement (`252.4 px` P95).
- The earlier three-dev scene probe changed camera composite from `0.3385` to `0.4546` (ICP) or `0.3962` (bounded mutual translation).
- Human3R pointmap and SMPL-X come from the same forward pass. Their agreement is a consistency proxy, not independent metric evidence.

## Saved exceptions are not split coverage

The original-demo depth/conf export covers `1` selected case; the virtual ray-query artifact covers `1` case. Neither supplies ordinary B0 foot-local evidence for MultiHuman validation and Ego chains.

## Minimal cache extension

Reuse the existing Human3R forward and retain compact camera-local 33x33 patches around the projected feet, including pointmap, confidence, UV, validity, human mask, foot depth/visibility, camera, and intrinsics. Do not store only world points: the frozen B0 camera must be applied at runtime.

Foot-local masks must use a 33x33 patch, keep a 4..16 px support annulus, and remove only the emitted human mask dilated by 3 px. The current whole-bbox removal is unsuitable. Preserve raw confidence and exact UV rather than only the selected 3D points.

The first candidate should preserve the last-pre signed foot-to-surface offset, not force distance to zero. It must be BRTC-accepted-only, plane-quality/visibility gated, bounded initially to 30 mm, camera-free, strictly causal, and exact fallback for every unobservable/rejected/unmatched person.

### First deterministic development gate

Require at least 24 valid samples, three UV quadrants, 5 cm 3D extent, <=2 cm weighted plane residual, <=25 deg pre/post normal disagreement, <=20 cm contact-like signed distance, and <=2 cm left/right proposal disagreement. Apply only if the predicted contact residual improves by at least 10%; the action is `clip(0.5 * proposal, 30 mm)`. These are initial dev values and must be frozen and checksummed before any held-out run.

The reference offset must be supported by both stable past-only observations and the raw current-post Human3R person/scene pair. This prevents blindly treating a moving or airborne foot as contact. Because both outputs are still same-forward Human3R, the gate remains a consistency safeguard, not independent metric depth evidence.

### CPU replay

Extend the existing CPU-only Ego cache builder and write to a new path under `/data`; do not overwrite the frozen cache. After that one forward pass, all policy scans and held-out evaluation must load the `.pt` with `map_location=cpu`, access no images or model, form actions before GT is loaded, and record cache/checkpoint/policy SHA256 plus per-person fallback reasons.

After extending the cache, freeze on three offset0 and validate unchanged on three offset1, dance, box, and full Ego chains with spatial, layout, reprojection, harm, root/joint Accel, camera, and fallback audits.
