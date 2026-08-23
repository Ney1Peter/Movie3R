#!/usr/bin/env python3
"""Boundary-free RGB-only deployment path for the frozen EgoBody candidate.

The module intentionally has no evaluator or supervision adapter.  A case is
represented only by its ordered RGB paths; the causal detector chooses the
event used by the current-model transaction.  Heavy CUDA/model imports are
lazy so provenance and contract tests never initialize a GPU.

This implementation is a benchmark candidate, not a proven replacement for
the formal inference path.  Until an independent numerical equivalence audit
is frozen, every report produced here is marked ``reportable_fps=false``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


FINAL_SCHEMA = "Bridge3R-EgoBody-frozen-final-candidate-v1"
PROTOCOL = "Bridge3R-EgoBody-CS150-v1"
CASE_REPORT_SCHEMA = "Bridge3R-EgoBody-deployment-runtime-case-v1"
EXPECTED_FINAL_SHA256 = "9d9f4c21b37fb3b53f889bd445a540a037c5f03e3f54f4beffa10580d9ddf58e"
EXPECTED_CANDIDATE_NAME = "v19_ungated_translation_b050"
PROVENANCE_BINDINGS = (
    ("development_summary", "development_summary_sha256"),
    ("holdout_summary", "holdout_summary_sha256"),
    ("holdout_recording_metrics", "holdout_recording_metrics_sha256"),
    ("holdout_candidates", "holdout_candidates_sha256"),
)


@dataclass(frozen=True)
class FrozenCandidate:
    path: Path
    sha256: str
    name: str
    candidate: dict[str, Any]
    payload: dict[str, Any]
    provenance_files: tuple[dict[str, str], ...]


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def validate_frozen_candidate(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_FINAL_SHA256,
) -> FrozenCandidate:
    """Validate schema, self-binding, selected config, and frozen provenance."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed_sha = file_sha256(resolved)
    if observed_sha != expected_sha256:
        raise ValueError(
            f"Final-candidate SHA-256 mismatch: {observed_sha} vs {expected_sha256}"
        )
    payload = _read_json_object(resolved)
    if payload.get("schema_version") != FINAL_SCHEMA:
        raise ValueError("Final candidate has an incompatible schema")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Final candidate has an incompatible protocol")
    if Path(str(payload.get("frozen_artifact_path", ""))).resolve() != resolved:
        raise ValueError("Final candidate is not self-bound to its frozen path")
    if payload.get("frozen_before_test") is not True:
        raise ValueError("Final candidate was not frozen before Test")
    if payload.get("test_metrics_read") is not False:
        raise ValueError("Final candidate does not certify unread Test metrics")

    rows = payload.get("candidates")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Final artifact must contain exactly one candidate")
    candidate = dict(rows[0])
    selected = str(payload.get("source_candidate_name", ""))
    if not selected or str(candidate.get("name", "")) != selected:
        raise ValueError("Final candidate name is not self-consistent")
    if selected != EXPECTED_CANDIDATE_NAME:
        raise ValueError(f"Unsupported frozen deployment candidate: {selected!r}")
    geometry = candidate.get("geometry")
    expected_geometry = {
        "name": EXPECTED_CANDIDATE_NAME,
        "camera_alpha": 1.0,
        "boundary_kind": "translation",
        "boundary_blend": 0.5,
    }
    if geometry != expected_geometry:
        raise ValueError(f"Frozen geometry differs from the audited config: {geometry}")
    if candidate.get("identity") is not None or candidate.get("person") is not None:
        raise ValueError("This deployment runtime supports the frozen geometry-only candidate")
    qualified = payload.get("qualified_holdout_candidates")
    if not isinstance(qualified, list) or selected not in qualified:
        raise ValueError("Selected candidate is absent from frozen qualification results")
    if payload.get("fallback_to_parent") is not False:
        raise ValueError("Unexpected final fallback selection")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Final candidate lacks frozen provenance")
    bound_files: list[dict[str, str]] = []
    for path_key, sha_key in PROVENANCE_BINDINGS:
        declared = provenance.get(path_key)
        expected = str(provenance.get(sha_key, ""))
        if not isinstance(declared, str) or not declared or len(expected) != 64:
            raise ValueError(f"Incomplete final provenance binding: {path_key}/{sha_key}")
        source = Path(declared).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = file_sha256(source)
        if observed != expected:
            raise ValueError(f"Frozen provenance SHA-256 mismatch for {path_key}")
        bound_files.append({"role": path_key, "path": str(source), "sha256": observed})
    return FrozenCandidate(
        path=resolved,
        sha256=observed_sha,
        name=selected,
        candidate=candidate,
        payload=payload,
        provenance_files=tuple(bound_files),
    )


def deployment_input_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a manifest row onto the only fields visible to deployment.

    Shot/evaluation annotations are deliberately not part of this projection.
    Consequently, changing any such annotation cannot change this interface.
    """

    case_id = str(record.get("case_id", ""))
    recording = str(record.get("recording", ""))
    if not case_id or not recording:
        raise ValueError("Deployment record requires case_id and recording")
    raw_paths = record.get("image_paths") or record.get("image_members")
    if not isinstance(raw_paths, list) or len(raw_paths) < 2:
        raise ValueError("Deployment record requires at least two ordered RGB paths")
    paths = tuple(str(value) for value in raw_paths)
    if any(not value for value in paths):
        raise ValueError("Deployment RGB paths must be non-empty")
    return {
        "case_id": case_id,
        "recording": recording,
        "image_paths": paths,
        "frame_count": len(paths),
    }


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe staged RGB path: {value!r}")
    return path


def resolve_staged_paths(deployment_input: Mapping[str, Any], staged_root: Path) -> list[Path]:
    root = Path(staged_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = [
        (root / _safe_relative_path(str(value))).resolve()
        for value in deployment_input["image_paths"]
    ]
    if any(root not in path.parents for path in output):
        raise ValueError("Resolved RGB path escapes the staged root")
    missing = [path for path in output if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return output


def process_peak_rss_bytes() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _frames_to_arrays(frames: list[dict[str, Any]], topology: Any, np: Any) -> dict[str, Any]:
    """Convert one deployment branch to the same common-SMPL array semantics."""

    frame_count = len(frames)
    people_max = max((len(frame["people"]) for frame in frames), default=0)
    cameras = np.stack([frame["camera"] for frame in frames]).astype(np.float32)
    vertices = np.full((frame_count, people_max, 6890, 3), np.nan, dtype=np.float32)
    joints = np.full((frame_count, people_max, 24, 3), np.nan, dtype=np.float32)
    persistent = np.full((frame_count, people_max), -1, dtype=np.int32)
    native = np.full((frame_count, people_max), -1, dtype=np.int32)
    valid = np.zeros((frame_count, people_max), dtype=np.uint8)
    for frame_index, frame in enumerate(frames):
        for person_index, person in enumerate(frame["people"]):
            smpl = topology.smplx_vertices_to_smpl(
                np.asarray(person["vertices"])[None]
            )[0]
            vertices[frame_index, person_index] = smpl
            joints[frame_index, person_index] = topology.joints_from_smpl(smpl)
            persistent[frame_index, person_index] = int(person["persistent_id"])
            native[frame_index, person_index] = int(person["native_id"])
            valid[frame_index, person_index] = 1
    return {
        "cameras_c2w": cameras,
        "vertices_world": vertices,
        "joints_world": joints,
        "persistent_ids": persistent,
        "native_ids": native,
        "valid": valid,
    }


class DeploymentRuntime:
    """One loaded current-model/detector instance reused across benchmark cases."""

    def __init__(self) -> None:  # pragma: no cover - instances come from load()
        raise RuntimeError("Use DeploymentRuntime.load()")

    @classmethod
    def load(
        cls,
        candidate: FrozenCandidate,
        *,
        device_name: str,
        size: int = 512,
        current_checkpoint: Path | None = None,
    ) -> "DeploymentRuntime":
        if not device_name.startswith("cuda:") or not device_name[5:].isdigit():
            raise ValueError("Deployment benchmark requires an explicit cuda:<index>")
        if int(size) != 512:
            raise ValueError("Frozen benchmark contract requires input size 512")

        # Heavy imports remain inside the real load path; contract tests can
        # import this module without touching CUDA.
        import numpy as np
        import torch
        from dust3r.model import ARCroco3DStereo
        from dust3r.utils.smpl_layer import SMPL_Layer
        from versions.v14.causal_image_detector import CausalGRUShotDetector
        from versions.v14.run_v14_2_single_sequence import configure_model
        from versions.v15.harmony4d import run_harmony_case as frozen
        from versions.v15.harmony4d.topology import CommonTopology
        from versions.v16.harmony4d.causal_stabilization import Candidate, apply_candidate

        device = torch.device(device_name)
        torch.cuda.set_device(device)
        default_current, _ = frozen.default_checkpoints()
        checkpoint = (current_checkpoint or default_current).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha = frozen.verified_artifact_sha256(checkpoint)
        detector_path = Path(frozen.DETECTOR_PATH).resolve()
        detector_sha = frozen.verified_artifact_sha256(detector_path)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        topology = CommonTopology.load()
        detector = CausalGRUShotDetector(detector_path)
        model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
        flags = configure_model(model)
        model.eval()
        layer = SMPL_Layer(
            type="smplx",
            gender="neutral",
            num_betas=10,
            kid=False,
            person_center="head",
        ).to(device).eval()
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - started

        instance = object.__new__(cls)
        instance._np = np
        instance._torch = torch
        instance._frozen = frozen
        instance._apply_candidate = apply_candidate
        instance._candidate_object = Candidate(**dict(candidate.candidate["geometry"]))
        instance._device = device
        instance._size = int(size)
        instance._topology = topology
        instance._detector = detector
        instance._model = model
        instance._layer = layer
        instance._candidate = candidate
        instance.load_metrics = {
            "seconds": float(load_seconds),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "forward_calls": 0,
            "forward_frames": 0,
            "branch": "model_detector_topology_load",
        }
        instance.artifacts = {
            "current_checkpoint": str(checkpoint),
            "current_checkpoint_sha256": checkpoint_sha,
            "detector": str(detector_path),
            "detector_sha256": detector_sha,
            "current_flags": flags,
            "topology": topology.metadata(),
        }
        return instance

    def _run_no_event(self, paths: list[Path]) -> dict[str, Any]:
        frames, forward = self._frozen.run_no_event(
            self._model,
            self._layer,
            self._topology,
            paths,
            self._device,
            self._size,
            "deployment_detector_miss",
        )
        frame_count = len(frames)
        del frames
        return {
            "branch": "detector_missed_lazy_exact_parent",
            "proposal_index": None,
            "association_pair_count": 0,
            "candidate_postprocess": "exact_parent_reference",
            "lazy_parent_fallback": True,
            "forward_calls": 1,
            "forward_frames": int(forward.get("frames", frame_count)),
            "forward_detail": [forward],
            "output_frames": frame_count,
        }

    def _run_detected(self, paths: list[Path], proposal: int) -> dict[str, Any]:
        frozen = self._frozen
        np = self._np
        if proposal <= 0 or proposal >= len(paths):
            raise ValueError(f"Invalid causal proposal {proposal} for {len(paths)} frames")
        pre_paths, post_paths = paths[:proposal], paths[proposal:]
        pre_views = frozen.gt_helpers.prepare_full_square_input(
            self._model, pre_paths, SimpleNamespace(size=self._size)
        )
        post_views = frozen.gt_helpers.prepare_full_square_input(
            self._model, post_paths, SimpleNamespace(size=self._size)
        )
        shadow_views = frozen.set_event_indices(
            copy.deepcopy(pre_views + post_views[:1]), {proposal}
        )
        raw_post_views = frozen.set_event_indices(copy.deepcopy(post_views), set())
        shadow_predictions, shadow_returned, shadow_debug, shadow_runtime = frozen.run_forward(
            self._model, shadow_views, self._device, "deployment_shadow"
        )
        shadow = frozen.decode_sequence(
            shadow_predictions,
            shadow_returned,
            shadow_debug,
            self._layer,
            self._topology,
        )
        del shadow_predictions, shadow_returned, shadow_debug, shadow_views
        raw_predictions, raw_returned, raw_debug, raw_runtime = frozen.run_forward(
            self._model, raw_post_views, self._device, "deployment_raw_post"
        )
        raw_post = frozen.decode_sequence(
            raw_predictions,
            raw_returned,
            raw_debug,
            self._layer,
            self._topology,
        )
        del raw_predictions, raw_returned, raw_debug, raw_post_views, pre_views, post_views
        if len(shadow) != proposal + 1 or len(raw_post) != len(paths) - proposal:
            raise RuntimeError("Deployment forward returned an unexpected frame count")

        b0_transform = np.asarray(shadow[-1]["camera"]) @ np.linalg.inv(
            np.asarray(raw_post[0]["camera"])
        )
        b0_post = frozen.map_frames(raw_post, b0_transform)
        source_frames = shadow[:-1] + b0_post
        association = frozen.anonymous_match(
            source_frames[proposal - 1]["people"],
            source_frames[proposal]["people"],
        )
        pairs = [tuple(map(int, pair)) for pair in association.get("pairs", [])]
        source_arrays = _frames_to_arrays(source_frames, self._topology, np)
        output_arrays, debug = self._apply_candidate(
            source_arrays,
            proposal,
            pairs,
            self._candidate_object,
        )
        runtime_contract = debug.get("runtime_contract", {})
        if runtime_contract.get("exact_m15_fallback") is True:
            raise RuntimeError(
                "Unexpected gated-parent fallback for the frozen ungated candidate"
            )
        output_frames = int(len(output_arrays["valid"]))
        forward_detail = [shadow_runtime, raw_runtime]
        forward_frames = sum(int(row.get("frames", 0)) for row in forward_detail)
        if forward_frames != len(paths) + 1:
            raise RuntimeError(
                f"Transaction forward-frame contract changed: {forward_frames}"
            )
        boundary_debug = debug.get("boundary", {})
        del output_arrays, source_arrays, source_frames, b0_post, raw_post, shadow
        return {
            "branch": "detected_b0_ungated_translation_b050",
            "proposal_index": int(proposal),
            "association_pair_count": len(pairs),
            "candidate_postprocess": str(boundary_debug.get("policy", "unknown")),
            "candidate_registration_accepted": bool(
                boundary_debug.get("accepted", False)
            ),
            "lazy_parent_fallback": False,
            "forward_calls": 2,
            "forward_frames": forward_frames,
            "forward_detail": forward_detail,
            "output_frames": output_frames,
        }

    def _execute(self, paths: list[Path]) -> dict[str, Any]:
        labels, detector_rows = self._detector.predict_sequence(paths)
        if len(labels) != len(paths):
            raise RuntimeError("Detector label count differs from RGB frame count")
        proposal = self._frozen.first_positive([int(value) for value in labels])
        derived = next((index for index, value in enumerate(labels) if int(value)), None)
        if proposal != derived:
            raise RuntimeError("Detector first-positive contract is inconsistent")
        branch = (
            self._run_no_event(paths)
            if proposal is None
            else self._run_detected(paths, int(proposal))
        )
        return {
            **branch,
            "input_frames": len(paths),
            "detector_positive_count": sum(bool(value) for value in labels),
            "detector_pair_count": len(detector_rows),
        }

    def measure_case(
        self,
        *,
        case_id: str,
        paths: list[Path],
        phase: str,
        repeat_index: int | None,
    ) -> dict[str, Any]:
        if phase not in {"warmup", "steady"}:
            raise ValueError(phase)
        if not paths or any(not Path(path).is_file() for path in paths):
            raise FileNotFoundError("One or more deployment RGB inputs are missing")
        torch = self._torch
        torch.cuda.synchronize(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)
        started = time.perf_counter()
        execution = self._execute(paths)
        torch.cuda.synchronize(self._device)
        seconds = time.perf_counter() - started
        return {
            "case_id": str(case_id),
            "phase": phase,
            "repeat_index": repeat_index,
            "seconds": float(seconds),
            "cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self._device)
            ),
            "cuda_max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self._device)
            ),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            **execution,
        }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(payload))
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if output.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace deployment report: {output}")
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--final-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [path.resolve() for path in args.images]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    candidate = validate_frozen_candidate(args.final_candidate)
    runtime = DeploymentRuntime.load(
        candidate,
        device_name=args.device,
        size=args.size,
        current_checkpoint=args.current_checkpoint,
    )
    measurement = runtime.measure_case(
        case_id=args.case_id,
        paths=paths,
        phase="steady",
        repeat_index=0,
    )
    report = {
        "schema_version": CASE_REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete_nonreportable",
        "reportable_fps": False,
        "nonreportable_reason": (
            "No frozen numerical equivalence audit against the formal path exists."
        ),
        "candidate": {
            "path": str(candidate.path),
            "sha256": candidate.sha256,
            "name": candidate.name,
        },
        "contract": {
            "rgb_only": True,
            "evaluation_annotation_consumed": False,
            "single_candidate_single_path": True,
        },
        "load": runtime.load_metrics,
        "measurement": measurement,
        "artifacts": runtime.artifacts,
    }
    _write_new_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "reportable_fps": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
