#!/usr/bin/env python3
"""V13 stage-2 frozen descriptor keyframe and world-anchor matching probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.image import pad_image  # noqa: E402
from v10_latent_activation_patching_probe import camera_matrix, run_branch  # noqa: E402
from v10_latent_token_cache import LayerwiseCollector  # noqa: E402
from v11_gauge_neutral_first_write_oracle import fixed_explicit_transform  # noqa: E402
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    configure_views,
    read_jsonl,
    record_spec,
    texture_score,
)
from v12_build_gauge_neutral_teacher_cache import old_a_dataset  # noqa: E402
from v13_scene_coordinate_oracle import (  # noqa: E402
    camera_points,
    confidence,
    direct_transform_error,
    human_mask,
    pose_error,
    robust_fit,
    transform_points,
    valid_points,
    weighted_similarity,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v13_world_coordinate_memory" / "stage2_frozen_world_memory"
DEFAULT_CANDIDATES = REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate_root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--patch_samples", type=int, default=256)
    parser.add_argument("--match_radius", type=float, default=0.50)
    parser.add_argument("--ransac_steps", type=int, default=192)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    value = value - value.mean(axis=1, keepdims=True)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def selected_layers(count: int, requested: int) -> list[int]:
    if count <= requested:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, requested).round().astype(int).tolist()))


def capture_descriptors(model, views: list[dict], patch_samples: int, seed: int, device: torch.device):
    for view in views:
        view["img_mhmr"] = pad_image(view["img"], int(model.mhmr_img_res))
    collector = LayerwiseCollector(model, boundary=1, patch_samples=patch_samples, seed=seed)
    collector.patch_frames = set(range(len(views)))
    started = time.perf_counter()
    with collector:
        with torch.no_grad():
            predictions, _ = model.forward_recurrent_lighter(
                views,
                str(device),
                ret_state=False,
                use_ttt3r=False,
                return_token_debug=False,
            )
    return predictions, collector.frame_data, time.perf_counter() - started


def descriptor_table(frame_data: dict) -> dict[str, np.ndarray]:
    table = {}
    encoder = frame_data["encoder_patch"]
    decoder = frame_data["decoder_patch"]
    for layer in selected_layers(len(encoder), 6):
        table[f"encoder_l{layer:02d}"] = np.asarray(encoder[layer], dtype=np.float32)
    for layer in selected_layers(len(decoder), 5):
        table[f"decoder_l{layer:02d}"] = np.asarray(decoder[layer], dtype=np.float32)
    dino = frame_data.get("dino_patch")
    if dino is not None and np.asarray(dino).size:
        table["dino_mhmr"] = np.asarray(dino, dtype=np.float32)
    return table


def frame_anchors(
    prediction: dict,
    view: dict,
    frame_data: dict,
    teacher_prediction: dict | None,
    old_from_raw: np.ndarray | None,
) -> dict:
    positions = np.asarray(frame_data["positions"], dtype=np.float32)
    patch_ids = np.asarray(frame_data["patch_ids"], dtype=np.int64)
    selected = positions[patch_ids]
    points = camera_points(prediction)
    shape = tuple(prediction["pts3d_in_self_view"].shape)
    height, width = int(shape[-3]), int(shape[-2])
    yy = np.clip(np.round((selected[:, 0] + 0.5) * 16.0).astype(np.int64), 0, height - 1)
    xx = np.clip(np.round((selected[:, 1] + 0.5) * 16.0).astype(np.int64), 0, width - 1)
    pixel_ids = yy * width + xx
    conf = confidence(prediction, len(points))[pixel_ids]
    mask = human_mask(view, len(points))[pixel_ids]
    camera = points[pixel_ids]
    valid = valid_points(camera, conf, mask, True)
    descriptors = {name: value[valid] for name, value in descriptor_table(frame_data).items()}
    pose = camera_matrix(prediction)
    output = {
        "world": transform_points(pose, camera[valid]),
        "confidence": conf[valid],
        "pixels": np.stack([yy[valid], xx[valid]], axis=1),
        "descriptors": descriptors,
    }
    if teacher_prediction is not None and old_from_raw is not None:
        teacher_points = camera_points(teacher_prediction)
        teacher_camera = teacher_points[pixel_ids][valid]
        gt_pose = gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float32)
        output["teacher_target_old"] = transform_points(old_from_raw @ gt_pose, teacher_camera)
    return output


def descriptor_names(frames: list[dict]) -> list[str]:
    return sorted(set.intersection(*(set(frame["descriptors"]) for frame in frames)))


def global_descriptor(frame: dict, name: str) -> np.ndarray:
    value = normalize_rows(frame["descriptors"][name])
    pooled = value.mean(axis=0)
    return pooled / max(np.linalg.norm(pooled), 1e-8)


def query_arrays(frames: list[dict], name: str | None = None):
    source = np.concatenate([frame["world"] for frame in frames], axis=0)
    target = np.concatenate([frame["teacher_target_old"] for frame in frames], axis=0)
    confidence_rows = np.concatenate([frame["confidence"] for frame in frames], axis=0)
    descriptors = None if name is None else np.concatenate([frame["descriptors"][name] for frame in frames], axis=0)
    return source, target, confidence_rows, descriptors


def memory_arrays(frames: list[dict], indices: list[int], name: str | None = None):
    world = np.concatenate([frames[index]["world"] for index in indices], axis=0)
    confidence_rows = np.concatenate([frames[index]["confidence"] for index in indices], axis=0)
    frame_ids = np.concatenate(
        [np.full(len(frames[index]["world"]), index, dtype=np.int32) for index in indices], axis=0
    )
    descriptors = None if name is None else np.concatenate([frames[index]["descriptors"][name] for index in indices], axis=0)
    return world, confidence_rows, frame_ids, descriptors


def oracle_keyframe(query_target: np.ndarray, old_frames: list[dict], radius: float) -> tuple[int, list[dict]]:
    diagnostics = []
    for index, frame in enumerate(old_frames):
        distances, _ = cKDTree(frame["world"]).query(query_target, k=1, workers=-1)
        diagnostics.append(
            {
                "frame": index,
                "overlap_020": float(np.mean(distances < 0.20)),
                "overlap_050": float(np.mean(distances < radius)),
                "median_distance_m": float(np.median(distances)),
            }
        )
    diagnostics.sort(key=lambda row: (-row["overlap_050"], row["median_distance_m"]))
    return int(diagnostics[0]["frame"]), diagnostics


def ransac_fit(
    source: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    steps: int,
    seed: int,
    device: torch.device,
) -> dict | None:
    if len(source) < 6:
        return None
    sample_size = min(4, len(source))
    source_gpu = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_gpu = torch.as_tensor(target, dtype=torch.float32, device=device)
    weight_gpu = torch.as_tensor(np.maximum(weight, 1e-6), dtype=torch.float32, device=device)
    probability = weight_gpu / weight_gpu.sum().clamp_min(1e-8)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    sample_ids = torch.multinomial(
        probability.expand(int(steps), -1),
        num_samples=sample_size,
        replacement=False,
        generator=generator,
    )
    sampled_source = source_gpu[sample_ids]
    sampled_target = target_gpu[sample_ids]
    sampled_weight = weight_gpu[sample_ids]
    sampled_weight = sampled_weight / sampled_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    source_mean = (sampled_weight[..., None] * sampled_source).sum(dim=1)
    target_mean = (sampled_weight[..., None] * sampled_target).sum(dim=1)
    source_centered = sampled_source - source_mean[:, None]
    target_centered = sampled_target - target_mean[:, None]
    covariance = torch.matmul(
        (sampled_weight[..., None] * target_centered).transpose(1, 2), source_centered
    )
    u, _, vh = torch.linalg.svd(covariance)
    sign = torch.ones((len(sample_ids), 3), dtype=torch.float32, device=device)
    sign[:, -1] = torch.where(torch.linalg.det(torch.matmul(u, vh)) < 0, -1.0, 1.0)
    rotation = torch.matmul(torch.matmul(u, torch.diag_embed(sign)), vh)
    translation = target_mean - torch.einsum("bij,bj->bi", rotation, source_mean)
    predicted = torch.einsum("nj,bkj->bnk", source_gpu, rotation) + translation[:, None]
    residual = torch.linalg.vector_norm(predicted - target_gpu[None], dim=-1)
    inlier = residual < 0.20
    inlier_count = inlier.sum(dim=1)
    weighted_inlier = (inlier * weight_gpu[None]).sum(dim=1)
    score = weighted_inlier - 0.05 * residual.median(dim=1).values
    score = torch.where(inlier_count >= 6, score, torch.full_like(score, -torch.inf))
    best_index = int(torch.argmax(score).item())
    if not torch.isfinite(score[best_index]):
        return robust_fit(source, target, weight, False)
    best_inlier = inlier[best_index].detach().cpu().numpy().astype(bool)
    refined = robust_fit(source[best_inlier], target[best_inlier], weight[best_inlier], False)
    if refined is None:
        return None
    predicted = source @ refined["rotation"].T + refined["translation"]
    refined["residual"] = np.linalg.norm(predicted - target, axis=1)
    refined["active"] = refined["residual"] < 0.20
    return refined


def fit_metrics(fit: dict | None, source: np.ndarray, target: np.ndarray, pred_pose: np.ndarray, target_pose: np.ndarray) -> dict:
    if fit is None:
        return {
            "camera_translation_error_m": float("nan"),
            "camera_rotation_error_deg": float("nan"),
            "correspondence_count": int(len(source)),
            "fit_failed": True,
        }
    residual = fit["residual"]
    return {
        **pose_error(fit, pred_pose, target_pose),
        "correspondence_count": int(len(source)),
        "fit_residual_mean_m": float(residual.mean()),
        "fit_residual_median_m": float(np.median(residual)),
        "inlier_ratio_0_20m": float(np.mean(residual < 0.20)),
        "fit_failed": False,
    }


def oracle_correspondence_method(
    query_frames: list[dict],
    old_frames: list[dict],
    keyframes: list[int],
    radius: float,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
) -> dict:
    source, true_target, query_conf, _ = query_arrays(query_frames)
    memory, memory_conf, _, _ = memory_arrays(old_frames, keyframes)
    distances, ids = cKDTree(memory).query(true_target, k=1, workers=-1)
    keep = distances < radius
    weight = np.sqrt(np.maximum(query_conf[keep] * memory_conf[ids[keep]], 1e-6))
    fit = robust_fit(source[keep], memory[ids[keep]], weight, False)
    result = fit_metrics(fit, source[keep], memory[ids[keep]], pred_pose, target_pose)
    result.update(
        {
            "oracle_correspondence": True,
            "keyframes": keyframes,
            "coverage_020": float(np.mean(distances < 0.20)),
            "coverage_050": float(np.mean(distances < 0.50)),
            "physical_match_mean_m": float(distances[keep].mean()) if keep.any() else None,
        }
    )
    return result


def frozen_match_method(
    query_frames: list[dict],
    old_frames: list[dict],
    keyframes: list[int],
    descriptor: str,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    ransac_steps: int,
    seed: int,
    device: torch.device,
) -> dict:
    source, true_target, query_conf, query_desc = query_arrays(query_frames, descriptor)
    memory, memory_conf, memory_frame, memory_desc = memory_arrays(old_frames, keyframes, descriptor)
    query_gpu = torch.as_tensor(normalize_rows(query_desc), dtype=torch.float32, device=device)
    memory_gpu = torch.as_tensor(normalize_rows(memory_desc), dtype=torch.float32, device=device)
    similarity_gpu = torch.matmul(query_gpu, memory_gpu.T)
    top_count = min(2, similarity_gpu.shape[1])
    top_values, top_ids = torch.topk(similarity_gpu, k=top_count, dim=1)
    best = top_ids[:, 0].detach().cpu().numpy()
    second = top_ids[:, 1].detach().cpu().numpy() if top_count > 1 else best
    similarity = similarity_gpu.detach().cpu().numpy()
    margin = similarity[np.arange(len(best)), best] - similarity[np.arange(len(best)), second]
    reverse = np.argmax(similarity, axis=0)
    mutual = reverse[best] == np.arange(len(best))
    candidate = np.nonzero(mutual)[0]
    if len(candidate) < 8:
        candidate = np.argsort(margin)[-min(max(32, len(candidate)), len(margin)) :]
    target = memory[best[candidate]]
    source_match = source[candidate]
    physical = np.linalg.norm(target - true_target[candidate], axis=1)
    cosine = similarity[candidate, best[candidate]]
    weight = np.sqrt(np.maximum(query_conf[candidate] * memory_conf[best[candidate]], 1e-6))
    weight *= np.maximum(cosine + 1.0, 1e-3) * np.maximum(margin[candidate] + 0.02, 0.01)
    fit = ransac_fit(source_match, target, weight, ransac_steps, seed, device)
    result = fit_metrics(fit, source_match, target, pred_pose, target_pose)
    result.update(
        {
            "oracle_correspondence": False,
            "keyframes": keyframes,
            "descriptor": descriptor,
            "mutual_match_count": int(mutual.sum()),
            "mean_cosine": float(cosine.mean()) if len(cosine) else None,
            "mean_margin": float(margin[candidate].mean()) if len(candidate) else None,
            "physical_match_mean_m": float(physical.mean()) if len(physical) else None,
            "physical_accuracy_010": float(np.mean(physical < 0.10)) if len(physical) else 0.0,
            "physical_accuracy_020": float(np.mean(physical < 0.20)) if len(physical) else 0.0,
            "physical_accuracy_050": float(np.mean(physical < 0.50)) if len(physical) else 0.0,
            "matched_memory_frame_histogram": np.bincount(
                memory_frame[best[candidate]], minlength=len(old_frames)
            ).tolist(),
        }
    )
    return result


def automatic_keyframes(query_frames: list[dict], old_frames: list[dict], descriptor: str) -> tuple[list[int], list[float]]:
    query = np.stack([global_descriptor(frame, descriptor) for frame in query_frames]).mean(axis=0)
    query = query / max(np.linalg.norm(query), 1e-8)
    old = np.stack([global_descriptor(frame, descriptor) for frame in old_frames])
    scores = old @ query
    order = np.argsort(-scores).astype(int).tolist()
    return order, scores.astype(float).tolist()


def run_case(record: dict, model, args: argparse.Namespace, device: torch.device, case_index: int) -> dict:
    spec = record_spec(record, args)
    reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, model.mhmr_img_res)
    teacher_views = configure_views(one_batch(build_dataset([spec], True, args)), device, model.mhmr_img_res)
    old_views = configure_views(one_batch(old_a_dataset(spec, args)), device, model.mhmr_img_res)
    reset_subset = reset_views[:3]
    reset_predictions, reset_data, reset_seconds = capture_descriptors(
        model, reset_subset, args.patch_samples, args.seed + case_index * 31, device
    )
    old_predictions, old_data, old_seconds = capture_descriptors(
        model, old_views, args.patch_samples, args.seed + case_index * 31 + 7, device
    )
    teacher_predictions, _, teacher_seconds, _ = run_branch(
        model, teacher_views, device, spec["warmup_count"], capture=False
    )
    teacher_post = teacher_predictions[spec["warmup_count"] : spec["warmup_count"] + 3]
    pred_pose0 = camera_matrix(reset_predictions[0])
    gt_pose0 = gt_pose_from_view(reset_subset[0]).detach().float().cpu().numpy().astype(np.float32)
    old_pred_pose = camera_matrix(old_predictions[-1])
    old_gt_pose = gt_pose_from_view(old_views[-1]).detach().float().cpu().numpy().astype(np.float32)
    old_from_raw = old_pred_pose @ np.linalg.inv(old_gt_pose)
    target_pose0 = old_from_raw @ gt_pose0
    boundary_gt = old_from_raw @ gt_pose0 @ np.linalg.inv(pred_pose0)
    explicit_raw, explicit_name = fixed_explicit_transform(args.candidate_root, record["pattern_id"])
    explicit_old = old_from_raw @ explicit_raw
    old_frames = [
        frame_anchors(old_predictions[index], old_views[index], old_data[index], None, None)
        for index in range(len(old_predictions))
    ]
    query_frames = [
        frame_anchors(
            reset_predictions[index],
            reset_subset[index],
            reset_data[index],
            teacher_post[index],
            old_from_raw,
        )
        for index in range(len(reset_predictions))
    ]
    names = descriptor_names(old_frames + query_frames)
    variants = {}
    retrieval = {}
    for frame_count in (1, 3):
        current = query_frames[:frame_count]
        _, query_target, _, _ = query_arrays(current)
        oracle_frame, oracle_diagnostics = oracle_keyframe(query_target, old_frames, args.match_radius)
        variants[f"oracle_keyframe_oracle_corr_{frame_count}f"] = oracle_correspondence_method(
            current, old_frames, [oracle_frame], args.match_radius, pred_pose0, target_pose0
        )
        retrieval[str(frame_count)] = {"oracle_frame": oracle_frame, "oracle_diagnostics": oracle_diagnostics}
        for descriptor in names:
            order, scores = automatic_keyframes(current, old_frames, descriptor)
            rank = order.index(oracle_frame) + 1
            retrieval[str(frame_count)][descriptor] = {
                "oracle_keyframe_rank": rank,
                "recall_at_1": rank <= 1,
                "recall_at_3": rank <= 3,
                "recall_at_5": rank <= 5,
                "order": order,
                "scores": scores,
            }
            variants[f"oracle_keyframe_{descriptor}_frozen_{frame_count}f"] = frozen_match_method(
                current,
                old_frames,
                [oracle_frame],
                descriptor,
                pred_pose0,
                target_pose0,
                args.ransac_steps,
                args.seed + case_index * 1009 + frame_count,
                device,
            )
            for topk in (1, 3, 5):
                selected = order[:topk]
                variants[f"auto_{descriptor}_top{topk}_oracle_corr_{frame_count}f"] = oracle_correspondence_method(
                    current, old_frames, selected, args.match_radius, pred_pose0, target_pose0
                )
                variants[f"auto_{descriptor}_top{topk}_frozen_{frame_count}f"] = frozen_match_method(
                    current,
                    old_frames,
                    selected,
                    descriptor,
                    pred_pose0,
                    target_pose0,
                    args.ransac_steps,
                    args.seed + case_index * 1009 + frame_count * 17 + topk,
                    device,
                )
    output = {
        "case_name": record["pattern_id"],
        "record": record,
        "descriptor_names": names,
        "texture_score": texture_score(reset_subset[0]),
        "memory": {
            "frame_count": len(old_frames),
            "anchor_count": int(sum(len(frame["world"]) for frame in old_frames)),
            "patch_limit_per_frame": int(args.patch_samples),
        },
        "baselines": {
            "hard_reset_no_alignment": direct_transform_error(np.eye(4, dtype=np.float32), pred_pose0, target_pose0),
            "fixed_explicit": {
                **direct_transform_error(explicit_old, pred_pose0, target_pose0),
                "name": explicit_name,
            },
            "boundary_oracle": direct_transform_error(boundary_gt, pred_pose0, target_pose0),
        },
        "retrieval": retrieval,
        "variants": variants,
        "timing_seconds": {
            "fresh_descriptor_capture": reset_seconds,
            "historical_memory_capture": old_seconds,
            "same_camera_teacher": teacher_seconds,
        },
    }
    del reset_views, teacher_views, old_views, reset_predictions, old_predictions, teacher_predictions
    torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V13 frozen world-memory probe requires CUDA Human3R inference")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"stage2_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if output.is_file() and not args.overwrite:
        print(f">> exists {output}")
        return
    records = read_jsonl(args.records)
    selected = [row for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    model = build_model(args)
    device = torch.device(args.device)
    cases = []
    started = time.perf_counter()
    for index, record in enumerate(selected):
        case = run_case(record, model, args, device, index)
        cases.append(case)
        best = min(
            (
                row for name, row in case["variants"].items()
                if "_frozen_3f" in name and math.isfinite(row["camera_rotation_error_deg"])
            ),
            key=lambda row: row["camera_rotation_error_deg"] + 5.0 * row["camera_translation_error_m"],
            default={},
        )
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} "
            f"bestT={best.get('camera_translation_error_m', float('nan')):.3f} "
            f"bestR={best.get('camera_rotation_error_deg', float('nan')):.2f}",
            flush=True,
        )
    report = {
        "experiment": "V13 Stage-2 Frozen World-Memory Probe",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
