#!/usr/bin/env python3
"""Export frozen-B0 and B0+BRTC-LC original Human3R demo payloads.

This is a CPU-only post-processing utility.  It copies an already saved
``demo.py --save`` payload and adds the deployable BRTC-LC world translation
for each accepted post-cut identity.  Cameras, depth, confidence, RGB, masks,
shape and pose are never recomputed.

The BRTC-LC report names GT identities for evaluation.  A saved demo payload,
on the other hand, stores persistent ``smpl_id`` values.  The required
``--identity_to_smpl_id`` mapping therefore comes from the already audited
automatic after-B0 association, not from GT at export time.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = "three_t0900_c3_c4_k0"
DEFAULT_SOURCE = Path(
    "/dev/shm/movie3r_v14_visual_long/three_t0900_c3_c4_k0/"
    "03_v14_learned_b0"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/b0_two_view_person_triangulation/"
    "dev_three.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/brtc_lc_original_demo" / DEFAULT_CASE
)
DEFAULT_IDENTITY_MAP = ("person0=1", "person1=0", "person2=2")
REQUIRED_SUBDIRS = ("camera", "color", "conf", "depth", "smpl")
IMMUTABLE_SUBDIRS = ("camera", "color", "conf", "depth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_payload", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--case_key", default=DEFAULT_CASE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cut_index", type=int, default=4)
    parser.add_argument(
        "--identity_to_smpl_id",
        nargs="+",
        default=DEFAULT_IDENTITY_MAP,
        metavar="IDENTITY=ID",
        help="Audited after-B0 identity to persistent demo smpl_id mapping.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_identity_map(values: list[str] | tuple[str, ...]) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected IDENTITY=ID, got {value!r}")
        identity, raw_smpl_id = value.split("=", 1)
        identity = identity.strip()
        if not identity or identity in output:
            raise ValueError(f"Invalid or repeated identity in {value!r}")
        output[identity] = int(raw_smpl_id)
    if len(set(output.values())) != len(output):
        raise ValueError(f"smpl_id values must be unique: {output}")
    return output


def load_case(path: Path, key: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in report["cases"] if row["case"]["key"] == key]
    if len(matches) != 1:
        raise KeyError(f"Expected one case {key!r} in {path}, found {len(matches)}")
    return matches[0]


def contiguous_files(directory: Path, suffix: str) -> list[Path]:
    files = sorted(directory.glob(f"*{suffix}"))
    expected = [f"{index:06d}{suffix}" for index in range(len(files))]
    if [path.name for path in files] != expected:
        raise RuntimeError(f"Non-contiguous {suffix} files under {directory}")
    return files


def validate_source(path: Path) -> int:
    if not path.is_dir():
        raise FileNotFoundError(path)
    suffixes = {
        "camera": ".npz",
        "color": ".png",
        "conf": ".npy",
        "depth": ".npy",
        "smpl": ".npz",
    }
    counts = {}
    for subdir in REQUIRED_SUBDIRS:
        directory = path / subdir
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        counts[subdir] = len(contiguous_files(directory, suffixes[subdir]))
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise RuntimeError(f"Payload frame counts disagree: {counts}")
    return counts["camera"]


def replace_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".new")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def copy_payload(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        if destination.parent == destination or str(destination) in ("/", "/data"):
            raise ValueError(f"Refusing broad deletion target: {destination}")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def metric_mean(row: dict, section: str, metric: str) -> float:
    values = [float(person[section][metric]) for person in row["people"]]
    return float(np.mean(values))


def shifts_by_smpl_id(
    row: dict, identity_map: dict[str, int]
) -> tuple[dict[int, np.ndarray], dict[str, dict]]:
    report_people = {person["identity"]: person for person in row["people"]}
    if set(identity_map) != set(report_people):
        raise ValueError(
            "Identity map and BRTC-LC report disagree: "
            f"map={sorted(identity_map)}, report={sorted(report_people)}"
        )
    by_smpl_id = {}
    details = {}
    for identity, smpl_id in identity_map.items():
        person = report_people[identity]
        shift = np.asarray(person["consensus_shift_world"], dtype=np.float64)
        if shift.shape != (3,) or not np.isfinite(shift).all():
            raise ValueError(f"Invalid shift for {identity}: {shift}")
        if not bool(person["accepted"]) and not np.array_equal(shift, np.zeros(3)):
            raise ValueError(f"Rejected identity {identity} has a nonzero shift")
        by_smpl_id[int(smpl_id)] = shift
        details[identity] = {
            "smpl_id": int(smpl_id),
            "accepted": bool(person["accepted"]),
            "shift_world": shift.tolist(),
            "baseline": person["baseline"],
            "corrected": person["corrected"],
        }
    return by_smpl_id, details


def apply_brtc_lc(
    payload: Path,
    frame_count: int,
    cut_index: int,
    shifts: dict[int, np.ndarray],
) -> dict:
    max_world_residual = 0.0
    observed_ids: dict[int, list[int]] = {}
    frame_records = []
    for index in range(cut_index, frame_count):
        camera_path = payload / "camera" / f"{index:06d}.npz"
        smpl_path = payload / "smpl" / f"{index:06d}.npz"
        with np.load(camera_path) as camera:
            pose = np.asarray(camera["pose"], dtype=np.float64)
        with np.load(smpl_path, allow_pickle=True) as source:
            values = {key: source[key] for key in source.files}
        transl_before = np.asarray(values["transl"], dtype=np.float64)
        transl_after = transl_before.copy()
        smpl_ids = np.asarray(values["smpl_id"], dtype=np.int64).reshape(-1)
        if transl_before.shape != (len(smpl_ids), 3):
            raise RuntimeError(
                f"Frame {index}: transl {transl_before.shape} vs ids {smpl_ids.shape}"
            )
        if len(set(int(value) for value in smpl_ids)) != len(smpl_ids):
            raise RuntimeError(f"Frame {index}: repeated smpl_id values {smpl_ids}")

        per_id = {}
        for row_index, raw_smpl_id in enumerate(smpl_ids):
            smpl_id = int(raw_smpl_id)
            observed_ids.setdefault(smpl_id, []).append(index)
            shift_world = shifts.get(smpl_id, np.zeros(3, dtype=np.float64))
            delta_camera = pose[:3, :3].T @ shift_world
            transl_after[row_index] += delta_camera
            reconstructed_world = pose[:3, :3] @ (
                transl_after[row_index] - transl_before[row_index]
            )
            residual = float(np.max(np.abs(reconstructed_world - shift_world)))
            max_world_residual = max(max_world_residual, residual)
            per_id[str(smpl_id)] = {
                "shift_world": shift_world.tolist(),
                "delta_camera": delta_camera.tolist(),
                "world_reconstruction_max_abs_error": residual,
            }

        values["transl"] = transl_after.astype(values["transl"].dtype, copy=False)
        if "verts_world" in values:
            vertices = np.asarray(values["verts_world"])
            if vertices.shape[0] != len(smpl_ids):
                raise RuntimeError(
                    f"Frame {index}: verts_world {vertices.shape} vs ids {smpl_ids.shape}"
                )
            vertices = vertices.copy()
            for row_index, raw_smpl_id in enumerate(smpl_ids):
                vertices[row_index] += shifts.get(
                    int(raw_smpl_id), np.zeros(3, dtype=np.float64)
                ).astype(vertices.dtype)
            values["verts_world"] = vertices
        replace_npz(smpl_path, values)
        frame_records.append({"frame_index": index, "people": per_id})

    missing = sorted(set(shifts) - set(observed_ids))
    if missing:
        raise RuntimeError(f"Mapped smpl_id values never appear after cut: {missing}")
    return {
        "max_world_reconstruction_abs_error": max_world_residual,
        "observed_post_frames_by_smpl_id": {
            str(key): value for key, value in sorted(observed_ids.items())
        },
        "frames": frame_records,
    }


def compare_files(left: Path, right: Path) -> bool:
    return filecmp.cmp(left, right, shallow=False)


def verify_export(
    source: Path,
    baseline: Path,
    corrected: Path,
    frame_count: int,
    cut_index: int,
    shifts: dict[int, np.ndarray],
) -> dict:
    # The frozen B0 copy must be byte-for-byte identical to its source.
    baseline_differences = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        copied = baseline / relative
        if not copied.is_file() or not compare_files(source_file, copied):
            baseline_differences.append(str(relative))
    if baseline_differences:
        raise RuntimeError(f"Frozen B0 copy differs: {baseline_differences[:10]}")

    # BRTC-LC is an SMPL translation-only edit.  All scene/camera files remain
    # byte-identical and pre-cut SMPL files remain byte-identical.
    immutable_differences = []
    for subdir in IMMUTABLE_SUBDIRS:
        for source_file in sorted((source / subdir).iterdir()):
            copied = corrected / subdir / source_file.name
            if not compare_files(source_file, copied):
                immutable_differences.append(str(copied.relative_to(corrected)))
    for index in range(cut_index):
        source_file = source / "smpl" / f"{index:06d}.npz"
        copied = corrected / "smpl" / source_file.name
        if not compare_files(source_file, copied):
            immutable_differences.append(str(copied.relative_to(corrected)))
    if immutable_differences:
        raise RuntimeError(
            f"Immutable BRTC-LC payload files differ: {immutable_differences[:10]}"
        )

    max_residual = 0.0
    for index in range(cut_index, frame_count):
        with np.load(source / "camera" / f"{index:06d}.npz") as camera:
            rotation = np.asarray(camera["pose"], dtype=np.float64)[:3, :3]
        with np.load(
            source / "smpl" / f"{index:06d}.npz", allow_pickle=True
        ) as before, np.load(
            corrected / "smpl" / f"{index:06d}.npz", allow_pickle=True
        ) as after:
            if before.files != after.files:
                raise RuntimeError(f"Frame {index}: SMPL keys changed")
            ids = np.asarray(before["smpl_id"], dtype=np.int64).reshape(-1)
            for key in before.files:
                if key in ("transl", "verts_world"):
                    continue
                if not np.array_equal(before[key], after[key], equal_nan=True):
                    raise RuntimeError(f"Frame {index}: unexpected SMPL edit to {key}")
            delta = np.asarray(after["transl"], dtype=np.float64) - np.asarray(
                before["transl"], dtype=np.float64
            )
            for row_index, raw_smpl_id in enumerate(ids):
                expected = shifts.get(int(raw_smpl_id), np.zeros(3))
                residual = float(
                    np.max(np.abs(rotation @ delta[row_index] - expected))
                )
                max_residual = max(max_residual, residual)
    if max_residual > 5e-7:
        raise RuntimeError(f"World-shift verification failed: {max_residual}")
    return {
        "baseline_source_byte_exact": True,
        "corrected_camera_depth_conf_color_byte_exact": True,
        "corrected_pre_cut_smpl_byte_exact": True,
        "corrected_post_cut_nontranslation_smpl_arrays_exact": True,
        "max_world_shift_abs_error": max_residual,
    }


def main() -> None:
    args = parse_args()
    source = args.source_payload.resolve()
    report_path = args.report.resolve()
    output_root = args.output_root.resolve()
    if str(output_root) in ("/", "/data"):
        raise ValueError(f"Refusing broad output target: {output_root}")
    if REPO_ROOT not in output_root.parents:
        raise ValueError(f"Output must stay under repository /data tree: {output_root}")
    frame_count = validate_source(source)
    cut_index = int(args.cut_index)
    if not 0 < cut_index < frame_count:
        raise ValueError(f"cut_index={cut_index} is invalid for {frame_count} frames")

    row = load_case(report_path, args.case_key)
    identity_map = parse_identity_map(args.identity_to_smpl_id)
    shifts, identity_details = shifts_by_smpl_id(row, identity_map)

    output_root.mkdir(parents=True, exist_ok=True)
    baseline = output_root / "b0_frozen"
    corrected = output_root / "b0_brtc_lc"
    copy_payload(source, baseline, bool(args.overwrite))
    copy_payload(source, corrected, bool(args.overwrite))
    application = apply_brtc_lc(
        corrected, frame_count, cut_index, shifts
    )
    verification = verify_export(
        source, baseline, corrected, frame_count, cut_index, shifts
    )

    metrics = {
        "human_root_error_m": {
            "b0": metric_mean(row, "baseline", "root_error_m"),
            "b0_brtc_lc": metric_mean(row, "corrected", "root_error_m"),
        },
        "human_joint_error_m": {
            "b0": metric_mean(row, "baseline", "joint_error_m"),
            "b0_brtc_lc": metric_mean(row, "corrected", "joint_error_m"),
        },
        "human_vertex_error_m": {
            "b0": metric_mean(row, "baseline", "vertex_error_m"),
            "b0_brtc_lc": metric_mean(row, "corrected", "vertex_error_m"),
        },
        "layout": row["layout"],
    }
    manifest = {
        "format": "two independent original Human3R demo.py saved payloads",
        "method": "frozen B0 camera + BRTC-LC per-person world translation",
        "cpu_only_postprocess": True,
        "model_inference_run": False,
        "case": row["case"],
        "frame_count": frame_count,
        "cut_index": cut_index,
        "identity_to_smpl_id": identity_map,
        "people": identity_details,
        "metrics": metrics,
        "source_payload": str(source),
        "source_report": str(report_path),
        "outputs": {
            "b0_frozen": str(baseline),
            "b0_brtc_lc": str(corrected),
        },
        "application_audit": application,
        "verification": verification,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> frozen B0: {baseline}", flush=True)
    print(f">> B0+BRTC-LC: {corrected}", flush=True)
    print(f">> frames: {frame_count}; people: {len(identity_map)}", flush=True)
    print(
        ">> root mean: "
        f"{metrics['human_root_error_m']['b0']:.6f} -> "
        f"{metrics['human_root_error_m']['b0_brtc_lc']:.6f} m",
        flush=True,
    )
    print(f">> manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
