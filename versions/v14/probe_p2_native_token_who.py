#!/usr/bin/env python3
"""P2: evaluate native Human3R tokens as a precision-first WHO certificate.

All candidate assignments are created from predicted token/geometry data before
GT identity assignment.  This script has no person-root or camera update.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from versions.v13 import gt_id_consensus as gt  # noqa: E402
from versions.v14.probe_p1_foot_scene_observability import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_MANIFEST, anonymous_match, decode_people,
    ensure_workspace, jsonable, sha256, transform_person,
)
from versions.v14.run_v14_2_single_sequence import camera_matrix, configure_model, set_event_indices  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/p2_native_token_who"
TOKEN_KEYS = ("refined_human_tokens", "human_head_tokens", "mhmr_head_tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes()).hexdigest()


def token_rows(debug: dict[str, Any], key: str, count: int) -> np.ndarray | None:
    value = debug.get(key)
    if value is None:
        return None
    array = np.asarray(gt.tensor_numpy(value), dtype=np.float32)
    while array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1 and count == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[0] != count or not np.isfinite(array).all():
        return None
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        return None
    return array / norms


def cosine_cost(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None or first.shape[1] != second.shape[1]:
        return None
    return 1.0 - np.clip(first @ second.T, -1.0, 1.0)


def hungarian(cost: np.ndarray | None) -> list[tuple[int, int]] | None:
    if cost is None or cost.ndim != 2 or not cost.size or not np.isfinite(cost).all():
        return None
    rows, columns = linear_sum_assignment(cost)
    return [(int(row), int(column)) for row, column in zip(rows, columns)]


def matrix_normalize(cost: np.ndarray) -> np.ndarray:
    values = cost[np.isfinite(cost) & (cost > 1e-9)]
    scale = float(np.median(values)) if len(values) else 1.0
    return cost / max(scale, 1e-9)


def certificate_pairs(token_cost: np.ndarray | None, fused_cost: np.ndarray | None) -> list[tuple[int, int]]:
    pairs = hungarian(fused_cost)
    if token_cost is None or pairs is None or token_cost.shape[0] < 2 or token_cost.shape[1] < 2:
        return []
    accepted = []
    for row, column in pairs:
        row_order = np.argsort(token_cost[row])
        col_order = np.argsort(token_cost[:, column])
        if int(row_order[0]) != column or int(col_order[0]) != row:
            continue
        row_first, row_second = token_cost[row, row_order[0]], token_cost[row, row_order[1]]
        col_first, col_second = token_cost[col_order[0], column], token_cost[col_order[1], column]
        row_margin = float((row_second - row_first) / max(row_first, 1e-6))
        col_margin = float((col_second - col_first) / max(col_first, 1e-6))
        if min(row_margin, col_margin) >= 0.10:
            accepted.append((row, column))
    return accepted


def evaluator_labels(
    people: list[dict[str, Any]], pose: np.ndarray, camera: int, frame: int, gt_args: SimpleNamespace
) -> tuple[dict[int, str], dict[str, Any]]:
    height, width = [int(value) for value in gt.tensor_numpy(gt_args._views_shape)[0]]
    assigned, audit = gt.assign_gt_identities(gt_args, people, pose, int(camera), int(frame), height, width)
    return {int(person["detection_index"]): str(identity) for identity, person in assigned.items()}, audit


def cache_case(model: ARCroco3DStereo, layer: SMPL_Layer, record: dict[str, Any], gt_args: SimpleNamespace, device: torch.device, size: int) -> dict[str, Any]:
    frame, pre_camera, post_camera = (int(record[key]) for key in ("frame", "pre_camera", "post_camera"))
    inputs = [gt.extract_video_frame(gt_args, pre_camera, frame - 1), gt.extract_video_frame(gt_args, pre_camera, frame), gt.extract_video_frame(gt_args, post_camera, frame)]
    views = gt.prepare_full_square_input(model, inputs, SimpleNamespace(size=int(size)))
    shadow_views, raw_views = set_event_indices(copy.deepcopy(views), {2}), set_event_indices(copy.deepcopy(views[2:]), set())
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow_predictions, shadow_returned, shadow_debug = model.forward_recurrent_lighter(shadow_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True)
        raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(raw_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True)
    pre_pose = camera_matrix(shadow_predictions[1]).astype(np.float64)
    raw_pose = camera_matrix(raw_predictions[0]).astype(np.float64)
    shadow_pose = camera_matrix(shadow_predictions[2]).astype(np.float64)
    b0 = boundary_from_camera_predictions(shadow_predictions[2], raw_predictions[0])[0].detach().float().cpu().numpy().astype(np.float64)
    b0_pose = b0 @ raw_pose
    parity = float(np.max(np.abs(shadow_pose - b0_pose)))
    if parity > 1e-5:
        raise RuntimeError(f"B0 camera parity failure {parity}")
    pre_people = decode_people(shadow_predictions[1], shadow_returned[1], shadow_debug[1], layer)
    raw_people = decode_people(raw_predictions[0], raw_returned[0], raw_debug[0], layer)
    b0_people = [transform_person(b0, person) for person in raw_people]
    descriptors = {
        key: {"pre": token_rows(shadow_debug[1], key, len(pre_people)), "post": token_rows(raw_debug[0], key, len(raw_people))}
        for key in TOKEN_KEYS
    }
    geometry = anonymous_match(pre_people, b0_people)
    runtime = {"record": dict(record), "pre_camera_c2w": pre_pose, "b0_camera_c2w": b0_pose, "b0_camera_sha256": array_sha256(b0_pose), "b0": b0,
               "pre_people": pre_people, "raw_post_people": raw_people, "b0_post_people": b0_people, "descriptors": descriptors, "geometry": geometry,
               "runtime_contract": {"gt_used": False, "future_post_frames_used": 0, "camera_update": "none", "shadow_state_committed": False}}
    # Runtime action/candidates above are complete.  Everything below is evaluator-only.
    gt_args._views_shape = shadow_returned[1]["true_shape"]
    pre_labels, pre_audit = evaluator_labels(pre_people, pre_pose, pre_camera, frame, gt_args)
    gt_args._views_shape = raw_returned[0]["true_shape"]
    post_labels, post_audit = evaluator_labels(raw_people, raw_pose, post_camera, frame, gt_args)
    return {"status": "ok", "runtime": runtime, "evaluator": {"pre_labels_by_detection": pre_labels, "post_labels_by_detection": post_labels, "pre_assignment": pre_audit, "post_assignment": post_audit}}


def build(args: argparse.Namespace) -> None:
    payload = json.loads(args.manifest.read_text(encoding="utf-8")); records = list(payload["dev"])
    if args.max_cases: records = records[:int(args.max_cases)]
    cases_dir = args.output_dir / "cases"; cases_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device); flags = configure_model(model)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    gt_args = SimpleNamespace(data_root=args.data_root, sequence="three", output_dir=args.output_dir / "frame_cache", size=int(args.size)); gt.IDENTITIES = gt.SEQUENCE_IDENTITIES["three"]
    paths, failures = [], []
    try:
        for number, record in enumerate(records, 1):
            path = cases_dir / f"{record['event_id']}.pt"
            if path.is_file() and not args.overwrite: row = torch.load(path, map_location="cpu", weights_only=False)
            else:
                try: row = cache_case(model, layer, record, gt_args, device, int(args.size))
                except Exception as error: row = {"status": "failed", "record": record, "error": repr(error), "traceback": traceback.format_exc()}
                torch.save(row, path)
            if row["status"] == "ok": paths.append(path); print(f"[{number:02d}/{len(records):02d}] {record['event_id']} pre/post={len(row['runtime']['pre_people'])}/{len(row['runtime']['raw_post_people'])}", flush=True)
            else: failures.append({"event_id": record["event_id"], "error": row["error"]}); print(f"[{number:02d}/{len(records):02d}] FAILED {record['event_id']}: {row['error']}", flush=True)
            if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        del layer, model
        if device.type == "cuda": torch.cuda.empty_cache()
    index = {"schema": 1, "checkpoint": str(args.model_path), "checkpoint_sha256": sha256(args.model_path), "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest), "flags": flags, "case_paths": [str(path) for path in paths], "failures": failures, "token_keys": TOKEN_KEYS, "runtime_before_gt": True}
    (args.output_dir / "P2_CACHE_INDEX.json").write_text(json.dumps(jsonable(index), indent=2) + "\n", encoding="utf-8")


def pair_stats(pairs: list[tuple[int, int]] | None, runtime: dict[str, Any], evaluator: dict[str, Any]) -> dict[str, Any]:
    pairs = pairs or []; pre_labels, post_labels = evaluator["pre_labels_by_detection"], evaluator["post_labels_by_detection"]
    correct, evaluable = 0, 0
    records = []
    for pre_index, post_index in pairs:
        pre_id = pre_labels.get(int(runtime["pre_people"][pre_index]["detection_index"]))
        post_id = post_labels.get(int(runtime["raw_post_people"][post_index]["detection_index"]))
        known = pre_id is not None and post_id is not None
        ok = bool(known and pre_id == post_id)
        evaluable += int(known); correct += int(ok)
        records.append({"pre_index": pre_index, "post_index": post_index, "evaluable": known, "correct_evaluator_only": ok})
    return {"pairs": records, "accepted": len(pairs), "evaluable": evaluable, "correct": correct}


def diagnose(args: argparse.Namespace) -> Path:
    index_path = args.output_dir / "P2_CACHE_INDEX.json"; index = json.loads(index_path.read_text(encoding="utf-8"))
    methods = {name: [] for name in ("G", "T_refined", "T_head", "T_mhmr", "TG", "TG_cert")}; token_parity = {key: [0, 0] for key in TOKEN_KEYS}; camera_hashes = {}
    for path_text in index["case_paths"]:
        cached = torch.load(path_text, map_location="cpu", weights_only=False); runtime, evaluator = cached["runtime"], cached["evaluator"]
        if runtime["runtime_contract"]["gt_used"] or runtime["runtime_contract"]["future_post_frames_used"]: raise RuntimeError("invalid P2 cache contract")
        if array_sha256(runtime["b0_camera_c2w"]) != runtime["b0_camera_sha256"]: raise RuntimeError("camera mutation")
        camera_hashes[runtime["record"]["event_id"]] = runtime["b0_camera_sha256"]
        g_cost = np.asarray(runtime["geometry"]["cost"], dtype=np.float64); g_pairs = hungarian(g_cost)
        costs = {key: cosine_cost(runtime["descriptors"][key]["pre"], runtime["descriptors"][key]["post"]) for key in TOKEN_KEYS}
        for key, cost in costs.items():
            token_parity[key][0] += 1; token_parity[key][1] += int(cost is not None)
        fused = None if costs["refined_human_tokens"] is None else matrix_normalize(g_cost) + matrix_normalize(costs["refined_human_tokens"])
        candidates = {"G": g_pairs, "T_refined": hungarian(costs["refined_human_tokens"]), "T_head": hungarian(costs["human_head_tokens"]), "T_mhmr": hungarian(costs["mhmr_head_tokens"]), "TG": hungarian(fused), "TG_cert": certificate_pairs(costs["refined_human_tokens"], fused)}
        for name, pairs in candidates.items(): methods[name].append({"event_id": runtime["record"]["event_id"], **pair_stats(pairs, runtime, evaluator)})
    summary = {}
    for name, cases in methods.items():
        accepted, evaluable, correct = (sum(row[key] for row in cases) for key in ("accepted", "evaluable", "correct"))
        exact = [all(item["correct_evaluator_only"] for item in row["pairs"] if item["evaluable"]) and any(item["evaluable"] for item in row["pairs"]) for row in cases]
        summary[name] = {"case_count": len(cases), "accepted": accepted, "evaluable": evaluable, "correct": correct, "precision": float(correct / max(evaluable, 1)), "coverage_vs_g_evaluable": float(evaluable / max(sum(sum(item["evaluable"] for item in row["pairs"]) for row in methods["G"]), 1)), "all_evaluable_pairs_correct_case_rate": float(np.mean(exact)) if exact else float("nan"), "cases": cases}
    gate = {"all_tg_cert_accepted_correct": summary["TG_cert"]["correct"] == summary["TG_cert"]["evaluable"], "tg_cert_coverage_at_least_20pct": summary["TG_cert"]["coverage_vs_g_evaluable"] >= .20, "tg_accuracy_not_below_g": summary["TG"]["precision"] >= summary["G"]["precision"], "token_row_parity_100pct": all(total == valid for total, valid in token_parity.values()), "camera_bit_exact": True}
    report = {"experiment": "v14_p2_native_token_who", "status": "GO_TO_FROZEN_CONFIRMATION" if all(gate.values()) else "NO_GO_NATIVE_TOKEN_WHO_CERTIFICATE", "cache_index": str(index_path), "cache_index_sha256": sha256(index_path), "methods": summary, "token_descriptor_case_parity": {key: {"total": total, "valid": valid} for key, (total, valid) in token_parity.items()}, "runtime_invariants": {"camera_sha256_by_event": camera_hashes, "all_candidates_before_gt": True, "external_pretrained_models": []}, "gate": gate}
    destination = args.output_dir / "P2_NATIVE_TOKEN_WHO_REPORT.json"; destination.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8"); return destination


def main() -> None:
    args = parse_args()
    for path in (args.model_path, args.manifest, args.output_dir): ensure_workspace(path)
    if not args.diagnose_only:
        if not args.model_path.is_file(): raise FileNotFoundError(args.model_path)
        build(args)
    report = diagnose(args); print(report, flush=True)


if __name__ == "__main__": main()
