# V14.1 Initial Event-Only Correction Results

Date: 2026-07-27

## 1. Scope

This report covers two capacity probes:

1. one-event overfit;
2. 10-event multi-source pilot.

Both use:

```text
A(t-1) -> A(t) -> B(t)
shot_labels = [0, 0, 1]
```

Only the event frame is corrected and supervised. These are training-event results,
not held-out generalization results.

## 2. Architecture

```text
V9 checkpoint initialization
+ semantic correct token
+ alignment correct token
+ full decoder attention on event frame
+ pose latent residual
+ human latent residual
+ event-only pose/human head LoRA
```

Momentum, previous correction residual/history, learned gate, AABB/AAAA classification,
future frames and segment loss are excluded.

## 3. One-Event Overfit

Sample:

```text
lbn1/22053926 frame 1191
lbn1/22053926 frame 1192
lbn1/22010716 frame 1192
view angle = 132.853 degrees
```

Training:

```text
40 epochs
GPU 2
2m26s
```

| Metric | Initial/raw | Corrected |
|---|---:|---:|
| Loss | 1.1098 | 0.0099 |
| Camera translation | 1.1773 m | 0.0891 m |
| Camera rotation | 17.245 deg | 0.0969 deg |
| Human translation | 0.5238 m | 0.0086 m |
| Event gate | 1.0 | 1.0 |

Checkpoint:

```text
output/v14_1/v14_1_cut_event_single_lbn1_1192/checkpoint-best.pth
```

Conclusion: the event-only correction branch has enough capacity to fit one hard
wide-view cut jointly for camera and human.

## 4. Ten-Event Dataset

| Source | Events |
|---|---:|
| AvatarReX | 3 |
| THuman | 2 |
| MVHuman100 | 3 |
| MVHuman200 | 2 |
| Total | 10 |

The first attempted config used `1 @ Dataset` for each source and therefore evaluated
only four events. That run was stopped. The valid run uses `3/2/3/2` for both train
and test datasets and evaluates all 10 events.

Training:

```text
12 epochs
10 mixed updates per epoch
each update consumes one event from every source
GPU 4
13m47s
peak GPU memory about 10.4 GB
```

## 5. Final Epoch-12 Results

Values are per-source means over all events in that source.

### 5.1 Camera Translation

| Source | Raw | Corrected | Reduction |
|---|---:|---:|---:|
| AvatarReX | 1.3564 m | 0.1348 m | 90.1% |
| THuman | 0.4488 m | 0.0480 m | 89.3% |
| MVHuman100 | 0.6039 m | 0.1052 m | 82.6% |
| MVHuman200 | 0.5111 m | 0.1043 m | 79.6% |
| Weighted 10-event mean | 0.7801 m | 0.1025 m | 86.9% |

### 5.2 Camera Rotation

| Source | Raw | Corrected | Reduction |
|---|---:|---:|---:|
| AvatarReX | 35.3508 deg | 0.2562 deg | 99.3% |
| THuman | 4.8252 deg | 0.0560 deg | 98.8% |
| MVHuman100 | 21.1010 deg | 0.1496 deg | 99.3% |
| MVHuman200 | 10.4828 deg | 0.1650 deg | 98.4% |
| Weighted 10-event mean | 19.9971 deg | 0.1659 deg | 99.2% |

### 5.3 Human Translation

| Source | Raw | Corrected | Reduction |
|---|---:|---:|---:|
| AvatarReX | 0.4624 m | 0.0032 m | 99.3% |
| THuman | 0.2496 m | 0.0047 m | 98.1% |
| MVHuman100 | 1.3627 m | 0.0051 m | 99.6% |
| MVHuman200 | 0.5968 m | 0.0157 m | 97.4% |
| Weighted 10-event mean | 0.7168 m | 0.0066 m | 99.1% |

### 5.4 Loss Progression

| Epoch | AvatarReX mean/median | THuman mean/median | MVHuman100 mean/median | MVHuman200 mean/median |
|---:|---:|---:|---:|---:|
| 0 | 3.4366 / 1.1098 | 0.4318 / 0.3116 | 12.6596 / 11.8502 | 3.9860 / 2.2153 |
| 4 | 3.5730 / 0.1390 | 0.0850 / 0.0624 | 0.4545 / 0.2042 | 0.2832 / 0.1502 |
| 8 | 0.2291 / 0.0633 | 0.0253 / 0.0219 | 0.0835 / 0.0966 | 0.0803 / 0.0793 |
| 12 | 0.0274 / 0.0231 | 0.0055 / 0.0049 | 0.0156 / 0.0183 | 0.0168 / 0.0081 |

At epoch 4 one AvatarReX event remained a rotation outlier: source mean rotation was
40.08 degrees although the median was 1.54 degrees. By epoch 8 this outlier was
reduced and the source mean rotation reached 2.55 degrees. At epoch 12 all four
source means were strongly below their raw baselines.

## 6. Visualization

All viewers use the same `lbn1` input and `cut_indices=2`:

| Port | Model |
|---:|---|
| 8091 | Original Human3R |
| 8092 | Formal V9 |
| 8093 | V14.1 one-event overfit upper bound |
| 8094 | V14.1 10-event pilot |

The viewer comparison is required because the current loss does not supervise
pointmap/scene alignment. A numerically good camera/human result is insufficient if
the pointmap, camera and human do not remain visually coherent.

## 7. Artifacts

Persistent single-event checkpoint:

```text
output/v14_1/v14_1_cut_event_single_lbn1_1192/checkpoint-best.pth
```

Persistent 10-event logs:

```text
output/v14_1/v14_1_cut_event_ten_sequences/log.txt
output/v14_1/v14_1_cut_event_ten_sequences/metrics_epoch.jsonl
output/v14_1/v14_1_cut_event_ten_sequences/train_steps.jsonl
```

The data volume was full during this run. The 10-event full checkpoint is temporarily
stored at:

```text
/dev/shm/movie3r_v14_1/v14_1_cut_event_ten_full/checkpoint-best.pth
```

This path is volatile and must not be treated as durable storage. Before a reboot,
either free at least 5 GB on `/data` and copy it, or implement a compact
base-checkpoint-plus-trainable-delta format.

## 8. Interpretation

Supported:

1. Two correct tokens are sufficient for strong first-post-cut fitting capacity.
2. Momentum and previous correction history are not required for this capacity probe.
3. Camera and human can be corrected jointly on the event frame.
4. The result is not limited to one source within the 10-event training set.

Not supported yet:

1. held-out event generalization;
2. no-cut output equivalence over long streams;
3. post-cut later-frame stability;
4. corrected-state commit versus raw-reset-state commit;
5. extraction of one explicit SE(3) Boundary;
6. scene/pointmap consistency;
7. automatic cut detection;
8. multi-human identity and V13 consensus.

## 9. Next Experiment

The next minimal experiment should freeze this event-frame model and test unseen cut
events from the same four sources, then compare:

```text
A. commit corrected recurrent state
B. commit raw hard-reset recurrent state and use corrected output only
C. derive one explicit first-frame SE(3) Boundary and apply it to later raw frames
```

Use at least 4-8 post-cut frames for this evaluation. Do not expand training scale
until the subjective viewer confirms that camera, pointmap and human move coherently.
