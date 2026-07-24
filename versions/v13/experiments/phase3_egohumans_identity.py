#!/usr/bin/env python3
"""Apply the frozen V13 Phase-3 WHO rule to an EgoHumans multi-cut stream.

This is a cross-data identity/TTL/fallback audit. The legacy EgoHumans probe
does not implement the full MultiHuman V16 geometry evaluator, so this script
does not report its geometry smoke test as a final V13 Boundary result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13.identity_bridge import (  # noqa: E402
    CausalIdentityMemory,
    MatchConfig,
    evaluate_assignment,
)
from versions.v13.native_token_probe import jsonable, tensor_numpy  # noqa: E402
from versions.v13.experiments.phase3_cross_shot_identity import (  # noqa: E402
    aggregate_identity,
    feature_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe_dir", type=Path, default=ROOT / "output/v13/egobody"
    )
    parser.add_argument(
        "--frozen_config",
        type=Path,
        default=ROOT / "output/v13/phase3_identity/three/frozen_identity_config.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v13/phase3_identity/egohumans_001_legoassemble",
    )
    return parser.parse_args()


def majority(votes: dict[int, list[int]]) -> dict[int, int]:
    output = {}
    for track_id, labels in votes.items():
        valid = np.asarray([value for value in labels if value >= 0], dtype=np.int64)
        if not len(valid):
            continue
        values, counts = np.unique(valid, return_counts=True)
        output[int(track_id)] = int(values[int(np.argmax(counts))])
    return output


def native_ids(debug: dict, count: int) -> np.ndarray:
    return (
        np.full(count, -1, dtype=np.int64)
        if debug.get("smpl_ids") is None
        else tensor_numpy(debug["smpl_ids"])[0].astype(np.int64)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = args.probe_dir / "v13_egobody_compact_tokens.pt"
    report_path = args.probe_dir / "v13_egobody_three_person_probe.json"
    if not compact_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "Run versions/v13/egobody_probe.py before this frozen-rule audit"
        )
    if not args.frozen_config.is_file():
        raise FileNotFoundError(f"Missing frozen three rule: {args.frozen_config}")
    compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    config = MatchConfig(**frozen["match_config"])
    predictions = compact["predictions"]
    debug = compact["token_debug"]
    labels = [np.asarray(value, dtype=np.int64) for value in compact["labels"]]
    frames = [feature_frame(prediction, row) for prediction, row in zip(predictions, debug)]
    cuts = {int(value) for value in source_report["stream"]["cuts"]}

    memory = CausalIdentityMemory(
        ttl=int(frozen.get("track_ttl", 8)), prototype_window=5
    )
    identity_votes: dict[int, list[int]] = defaultdict(list)
    native_to_external: dict[int, int] = {}
    cut_rows = []
    snapshots = []
    for frame_index, (frame, frame_labels, debug_row) in enumerate(
        zip(frames, labels, debug)
    ):
        current_native = native_ids(debug_row, int(frame["count"]))
        if frame_index == 0:
            external = memory.bootstrap(frame, timestamp=frame_index, use_native_ids=True)
            native_to_external = {
                int(native): int(track)
                for native, track in zip(current_native, external)
                if int(native) >= 0
            }
        elif frame_index in cuts:
            # Camera-cut reset starts a new native-ID namespace. External IDs
            # survive only through this explicit, frozen identity bridge.
            result = memory.tentative_match(frame, config, timestamp=frame_index)
            bank_labels = majority(identity_votes)
            metrics = evaluate_assignment(result, bank_labels, frame_labels)
            before_commit = memory.snapshot()
            external = memory.commit(frame, result, timestamp=frame_index)
            after_commit = memory.snapshot()
            native_to_external = {
                int(native): int(track)
                for native, track in zip(current_native, external)
                if int(native) >= 0
            }
            cut_rows.append(
                {
                    "frame_index": frame_index,
                    "camera": source_report["stream"]["segments"][
                        sum(frame_index >= value for value in cuts)
                    ]["camera"],
                    "identity": metrics,
                    "cost": result["cost"],
                    "accepted_pairs": result["accepted_pairs"],
                    "unmatched_source": result["unmatched_source"],
                    "unmatched_target": result["unmatched_target"],
                    "memory_before_commit": before_commit,
                    "memory_after_commit": after_commit,
                }
            )
        else:
            external_values = []
            for native in current_native:
                native = int(native)
                if native not in native_to_external:
                    native_to_external[native] = memory.next_track_id
                    memory.next_track_id += 1
                external_values.append(native_to_external[native])
            external = np.asarray(external_values, dtype=np.int64)
            memory.observe(frame, external, timestamp=frame_index)
        for track_id, label in zip(external, frame_labels):
            if int(label) >= 0:
                identity_votes[int(track_id)].append(int(label))
        snapshots.append(memory.snapshot())

    aggregate = aggregate_identity([row["identity"] for row in cut_rows])
    report = {
        "experiment": "V13 Phase 3 EgoHumans frozen identity bridge audit",
        "dataset": "EgoHumans 001_legoassemble",
        "scope": "cross-data WHO, multi-cut memory, dustbin and TTL; not final V13 geometry benchmark",
        "selected_config": frozen,
        "candidate_gt_usage": {
            "matching": False,
            "memory_update": False,
            "identity_scoring": True,
            "legacy_geometry_smoke_not_reused_as_final_result": True,
        },
        "cuts": cut_rows,
        "identity": aggregate,
        "memory_timeline": snapshots,
        "native_tracker_reference": source_report.get("native_tracker"),
    }
    output = args.output_dir / "v13_phase3_egohumans_identity.json"
    output.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(
        1, max(1, len(cut_rows)), figsize=(5 * max(1, len(cut_rows)), 4), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, row in zip(axes, cut_rows):
        cost = np.asarray(row["cost"])
        image = axis.imshow(cost, cmap="magma") if cost.size else None
        axis.set_title(f"EgoHumans cut {row['frame_index']}")
        axis.set_xlabel("post detection")
        axis.set_ylabel("external track")
        if image is not None:
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(args.output_dir / "egohumans_frozen_identity_matrices.png", dpi=160)
    plt.close(figure)
    print(f">> EgoHumans Phase 3 identity audit: {output}", flush=True)


if __name__ == "__main__":
    main()
