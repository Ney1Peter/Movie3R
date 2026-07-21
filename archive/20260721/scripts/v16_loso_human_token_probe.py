#!/usr/bin/env python3
"""LOSO probe for human-token residual and confidence roles in V16."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from sklearn.metrics import brier_score_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_implicit_explicit_cross_shot_probe import background_cloud, history_background_cloud  # noqa: E402
from v10_oracle_candidate_selection_probe import predicted_poses  # noqa: E402
from v13_scene_coordinate_oracle import direct_transform_error  # noqa: E402
from v16_human_torso_candidates import make_transform, scene_translation_fixed_rotation  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
DEFAULT_V10_REPORT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "token_loso"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10_REPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_residual_deg", type=float, default=45.0)
    parser.add_argument("--cloud_points_per_frame", type=int, default=3000)
    parser.add_argument("--translation_iters", type=int, default=8)
    parser.add_argument("--translation_max_distance", type=float, default=0.60)
    parser.add_argument("--translation_min_distance", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


@dataclass
class Sample:
    case: dict
    old_token: np.ndarray
    new_token: np.ndarray
    local_dir: Path

    @property
    def name(self) -> str:
        return str(self.case["case_name"])

    @property
    def source(self) -> str:
        return str(self.case["record"]["source"])


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(cache_dir: Path, v10_report: Path) -> list[Sample]:
    paths = sorted(glob.glob(str(cache_dir / "v16_candidates_shard_*_of_*.json")))
    if not paths:
        raise FileNotFoundError(f"No V16 candidate shards in {cache_dir}")
    v10 = json.loads(v10_report.read_text(encoding="utf-8"))
    local_dirs = {case["case_name"]: Path(case["paths"]["human3r_local_reset"]) for case in v10["cases"]}
    samples = []
    for json_path in paths:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        feature_path = Path(json_path.replace("v16_candidates_", "v16_tokens_")).with_suffix(".npz")
        with np.load(feature_path) as features:
            names = [str(value) for value in features["case_names"]]
            old = features["old_human_token"].astype(np.float32)
            new = features["new_human_token"].astype(np.float32)
        cases = {str(case["case_name"]): case for case in payload["cases"]}
        for index, name in enumerate(names):
            samples.append(Sample(cases[name], old[index], new[index], local_dirs[name]))
    names = [sample.name for sample in samples]
    if len(names) != 180 or len(names) != len(set(names)):
        raise RuntimeError(f"Expected 180 unique V16 samples, got {len(names)}/{len(set(names))}")
    return samples


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1e-6)


def token_feature(sample: Sample) -> np.ndarray:
    old = sample.old_token.astype(np.float32)
    new = sample.new_token.astype(np.float32)
    old_mean, old_last, old_std = old.mean(axis=0), old[-1], old.std(axis=0)
    new_first, new_mean, new_std = new[0], new.mean(axis=0), new.std(axis=0)
    vectors = np.stack([old_mean, old_last, old_std, new_first, new_mean, new_std, new_first - old_last])
    vectors = normalize_rows(vectors)
    cosine = float(np.dot(normalize_rows(old_last[None])[0], normalize_rows(new_first[None])[0]))
    scalars = np.asarray(
        [
            cosine,
            np.linalg.norm(old_last),
            np.linalg.norm(new_first),
            np.linalg.norm(new_first - old_last),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vectors.reshape(-1), scalars])


def nested_get(row: dict, path: tuple[str, ...], default: float = 0.0) -> float:
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def geometry_feature(sample: Sample) -> np.ndarray:
    case = sample.case
    fixed = case["baselines"]["fixed_explicit"]
    torso = case["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]
    checked = case["fixed_candidates"]["fixed_torso_motion_1f_root_check"]
    motion = case["motion_diagnostics"]
    old_ground = case["ground_diagnostics"]["old"]
    new_ground = case["ground_diagnostics"]["new_1f"]
    solver = torso.get("translation_solver", {})
    iterations = solver.get("iterations", [])
    last_iteration = iterations[-1] if iterations else {}
    values = [
        nested_get(torso, ("raw_residual_deg",)) / 45.0,
        abs(nested_get(torso, ("raw_residual_deg",))) / 45.0,
        nested_get(torso, ("bounded_residual_deg",)) / 45.0,
        float(bool(torso.get("clipped", False))),
        nested_get(torso, ("angle_median_abs_deviation_deg",)) / 10.0,
        nested_get(motion, ("angular_speed_deg_per_frame",)) / 10.0,
        nested_get(motion, ("spread_deg",)) / 10.0,
        nested_get(motion, ("inlier_count",)) / max(nested_get(motion, ("count",), 1.0), 1.0),
        nested_get(fixed, ("human_root_jump_m",)),
        nested_get(fixed, ("human_torso_jump_deg",)) / 30.0,
        nested_get(torso, ("human_root_jump_m",)),
        nested_get(torso, ("human_torso_jump_deg",)) / 30.0,
        nested_get(checked, ("coarse_root_motion_error_m",)),
        nested_get(checked, ("corrected_root_motion_error_m",)),
        nested_get(solver, ("residual_from_t0_m",)),
        nested_get(last_iteration, ("pairs",)) / 1000.0,
        nested_get(last_iteration, ("median_distance_m",)),
        nested_get(old_ground, ("valid_frames",)) / 3.0,
        nested_get(old_ground, ("spread_deg",)) / 15.0,
        nested_get(new_ground, ("valid_frames",)),
        nested_get(new_ground, ("spread_deg",)) / 15.0,
        float(case.get("texture_score", 0.0)) * 10.0,
    ]
    return np.asarray(values, dtype=np.float32)


def transform_for(sample: Sample, group: str, key: str) -> np.ndarray:
    return np.asarray(sample.case[group][key]["transform"], dtype=np.float32)


def rotation_error(transform: np.ndarray, oracle: np.ndarray) -> float:
    delta = transform[:3, :3] @ oracle[:3, :3].T
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta.astype(np.float64)).as_rotvec())))


def oracle_transform(sample: Sample) -> np.ndarray:
    return transform_for(sample, "baselines", "boundary_oracle")


def fixed_transform(sample: Sample) -> np.ndarray:
    return transform_for(sample, "baselines", "fixed_explicit")


def torso_transform(sample: Sample) -> np.ndarray:
    return transform_for(sample, "fixed_candidates", "fixed_torso_motion_1f_resolve_t")


def target_residual(sample: Sample, maximum_deg: float) -> np.ndarray:
    fixed = fixed_transform(sample)
    oracle = oracle_transform(sample)
    rotvec = Rotation.from_matrix((oracle[:3, :3] @ fixed[:3, :3].T).astype(np.float64)).as_rotvec()
    maximum = math.radians(maximum_deg)
    norm = float(np.linalg.norm(rotvec))
    if norm > maximum:
        rotvec *= maximum / norm
    return rotvec.astype(np.float32)


def optimal_torso_gate(sample: Sample) -> float:
    fixed = fixed_transform(sample)
    torso = torso_transform(sample)
    oracle = oracle_transform(sample)
    delta = torso[:3, :3] @ fixed[:3, :3].T
    rotvec = Rotation.from_matrix(delta.astype(np.float64)).as_rotvec()
    best_gate, best_error = 0.0, float("inf")
    for gate in np.linspace(0.0, 1.0, 101):
        rotation = Rotation.from_rotvec(gate * rotvec).as_matrix() @ fixed[:3, :3]
        candidate = make_transform(rotation.astype(np.float32), fixed[:3, 3])
        error = rotation_error(candidate, oracle)
        if error < best_error:
            best_gate, best_error = float(gate), error
    return best_gate


class Projector:
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, components: torch.Tensor):
        self.mean = mean
        self.scale = scale
        self.components = components

    @classmethod
    def fit(cls, values: np.ndarray, dimension: int, device: torch.device) -> "Projector":
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        mean = tensor.mean(dim=0)
        centered = tensor - mean
        scale = centered.std(dim=0, unbiased=False).clamp_min(1e-4)
        normalized = centered / scale
        q = min(int(dimension), normalized.shape[0] - 1, normalized.shape[1])
        _, _, components = torch.pca_lowrank(normalized, q=q, center=False)
        return cls(mean, scale, components)

    def transform(self, values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.mean.device)
        return ((tensor - self.mean) / self.scale) @ self.components


class Standardizer:
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        self.mean = mean
        self.scale = scale

    @classmethod
    def fit(cls, values: np.ndarray, device: torch.device) -> "Standardizer":
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        return cls(tensor.mean(dim=0), tensor.std(dim=0, unbiased=False).clamp_min(1e-4))

    def transform(self, values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.mean.device)
        return (tensor - self.mean) / self.scale


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    output_dim: int,
    task: str,
    args: argparse.Namespace,
    seed: int,
) -> TinyMLP:
    seed_all(seed)
    model = TinyMLP(x.shape[1], int(args.hidden_dim), output_dim).to(x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    if task == "classification":
        positive = float(y.sum().item())
        negative = float(len(y) - positive)
        pos_weight = torch.tensor([negative / max(positive, 1.0)], device=x.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif task == "gate":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=math.radians(5.0))
    for _ in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        if task == "classification":
            loss = criterion(output[:, 0], y)
        elif task == "gate":
            loss = criterion(torch.sigmoid(output[:, 0]), y)
        else:
            loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    return model.eval()


def choose_threshold(probability: np.ndarray, labels: np.ndarray, base_error: np.ndarray, candidate_error: np.ndarray) -> float:
    best_threshold, best_objective = 1.0, float("inf")
    for threshold in np.linspace(0.05, 0.95, 19):
        accept = probability >= threshold
        selected = np.where(accept, candidate_error, base_error)
        false_rate = float(np.mean(accept & ~labels))
        objective = float(selected.mean() + 10.0 * max(false_rate - 0.10, 0.0))
        if objective < best_objective - 1e-8 or (abs(objective - best_objective) < 1e-8 and threshold > best_threshold):
            best_threshold, best_objective = float(threshold), objective
    return best_threshold


def apply_residual(fixed: np.ndarray, rotvec: np.ndarray, maximum_deg: float) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    maximum = math.radians(maximum_deg)
    norm = float(np.linalg.norm(vector))
    if norm > maximum:
        vector *= maximum / norm
    rotation = Rotation.from_rotvec(vector).as_matrix().astype(np.float32) @ fixed[:3, :3]
    return rotation.astype(np.float32)


def apply_gate(fixed: np.ndarray, torso: np.ndarray, gate: float) -> np.ndarray:
    delta = torso[:3, :3] @ fixed[:3, :3].T
    vector = Rotation.from_matrix(delta.astype(np.float64)).as_rotvec()
    return (Rotation.from_rotvec(float(np.clip(gate, 0.0, 1.0)) * vector).as_matrix() @ fixed[:3, :3]).astype(np.float32)


def evaluated_transform(sample: Sample, rotation: np.ndarray, args: argparse.Namespace, cloud_cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    fixed = fixed_transform(sample)
    if sample.name not in cloud_cache:
        target, _ = history_background_cloud(sample.local_dir, [0, 1], int(args.cloud_points_per_frame))
        source, _ = background_cloud(sample.local_dir, 2, int(args.cloud_points_per_frame), int(args.seed))
        cloud_cache[sample.name] = (source, target)
    source, target = cloud_cache[sample.name]
    translation, diagnostics = scene_translation_fixed_rotation(rotation, fixed[:3, 3], source, target, args)
    transform = make_transform(rotation, translation)
    pred_pose = predicted_poses(sample.local_dir)[2]
    target_pose = oracle_transform(sample) @ pred_pose
    return {
        **direct_transform_error(transform, pred_pose, target_pose),
        "fit_failed": False,
        "transform": transform.tolist(),
        "translation_solver": diagnostics,
    }


def stored_row(sample: Sample, group: str, key: str) -> dict:
    return dict(sample.case[group][key])


def confidence_metrics(probability: np.ndarray, labels: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        auroc = None
    else:
        auroc = float(roc_auc_score(labels, probability))
    return {
        "auroc": auroc,
        "brier": float(brier_score_loss(labels, probability)),
        "positive_rate": float(np.mean(labels)),
        "mean_probability": float(np.mean(probability)),
    }


def run_fold(
    samples: list[Sample],
    token_raw: np.ndarray,
    geometry_raw: np.ndarray,
    held_out: str,
    args: argparse.Namespace,
    fold_index: int,
) -> dict:
    device = torch.device(args.device)
    train_ids = np.asarray([index for index, sample in enumerate(samples) if sample.source != held_out])
    test_ids = np.asarray([index for index, sample in enumerate(samples) if sample.source == held_out])
    projector = Projector.fit(token_raw[train_ids], int(args.pca_dim), device)
    standardizer = Standardizer.fit(geometry_raw[train_ids], device)
    token_train, token_test = projector.transform(token_raw[train_ids]), projector.transform(token_raw[test_ids])
    geometry_train, geometry_test = standardizer.transform(geometry_raw[train_ids]), standardizer.transform(geometry_raw[test_ids])
    combined_train = torch.cat([geometry_train, token_train], dim=1)
    combined_test = torch.cat([geometry_test, token_test], dim=1)

    labels = np.asarray(
        [
            rotation_error(fixed_transform(sample), oracle_transform(sample))
            - rotation_error(torso_transform(sample), oracle_transform(sample))
            > 0.5
            for sample in samples
        ],
        dtype=np.float32,
    )
    residual_targets = np.stack([target_residual(sample, float(args.max_residual_deg)) for sample in samples])
    gate_targets = np.asarray([optimal_torso_gate(sample) for sample in samples], dtype=np.float32)
    label_train = torch.as_tensor(labels[train_ids], dtype=torch.float32, device=device)
    residual_train = torch.as_tensor(residual_targets[train_ids], dtype=torch.float32, device=device)
    gate_train = torch.as_tensor(gate_targets[train_ids], dtype=torch.float32, device=device)

    seed = int(args.seed) + fold_index * 100
    geometry_classifier = train_model(geometry_train, label_train, 1, "classification", args, seed + 1)
    token_classifier = train_model(token_train, label_train, 1, "classification", args, seed + 2)
    combined_classifier = train_model(combined_train, label_train, 1, "classification", args, seed + 3)
    token_residual = train_model(token_train, residual_train, 3, "residual", args, seed + 4)
    token_gate = train_model(token_train, gate_train, 1, "gate", args, seed + 5)
    combined_gate = train_model(combined_train, gate_train, 1, "gate", args, seed + 6)

    with torch.no_grad():
        probability_train = {
            "geometry": torch.sigmoid(geometry_classifier(geometry_train)[:, 0]).cpu().numpy(),
            "token": torch.sigmoid(token_classifier(token_train)[:, 0]).cpu().numpy(),
            "geometry_token": torch.sigmoid(combined_classifier(combined_train)[:, 0]).cpu().numpy(),
        }
        probability_test = {
            "geometry": torch.sigmoid(geometry_classifier(geometry_test)[:, 0]).cpu().numpy(),
            "token": torch.sigmoid(token_classifier(token_test)[:, 0]).cpu().numpy(),
            "geometry_token": torch.sigmoid(combined_classifier(combined_test)[:, 0]).cpu().numpy(),
        }
        residual_test = token_residual(token_test).cpu().numpy()
        token_gate_test = torch.sigmoid(token_gate(token_test)[:, 0]).cpu().numpy()
        combined_gate_test = torch.sigmoid(combined_gate(combined_test)[:, 0]).cpu().numpy()

    train_base_error = np.asarray([rotation_error(fixed_transform(samples[index]), oracle_transform(samples[index])) for index in train_ids])
    train_candidate_error = np.asarray([rotation_error(torso_transform(samples[index]), oracle_transform(samples[index])) for index in train_ids])
    thresholds = {
        key: choose_threshold(value, labels[train_ids].astype(bool), train_base_error, train_candidate_error)
        for key, value in probability_train.items()
    }

    rows = []
    cloud_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for local_index, sample_index in enumerate(test_ids):
        sample = samples[sample_index]
        fixed_row = stored_row(sample, "baselines", "fixed_explicit")
        torso_row = stored_row(sample, "fixed_candidates", "fixed_torso_motion_1f_resolve_t")
        oracle_row = stored_row(sample, "baselines", "boundary_oracle")
        methods = {
            "fixed_explicit": fixed_row,
            "torso_geometry": torso_row,
            "boundary_oracle": oracle_row,
        }
        for key in ("geometry", "token", "geometry_token"):
            accept = bool(probability_test[key][local_index] >= thresholds[key])
            methods[f"{key}_confidence_select"] = torso_row if accept else fixed_row
        token_rotation = apply_residual(fixed_transform(sample), residual_test[local_index], float(args.max_residual_deg))
        methods["token_direct_residual"] = evaluated_transform(sample, token_rotation, args, cloud_cache)
        token_gate_rotation = apply_gate(fixed_transform(sample), torso_transform(sample), float(token_gate_test[local_index]))
        methods["torso_token_gate"] = evaluated_transform(sample, token_gate_rotation, args, cloud_cache)
        combined_gate_rotation = apply_gate(fixed_transform(sample), torso_transform(sample), float(combined_gate_test[local_index]))
        methods["torso_geometry_token_gate"] = evaluated_transform(sample, combined_gate_rotation, args, cloud_cache)
        rows.append(
            {
                "case_name": sample.name,
                "source": sample.source,
                "label_torso_helpful": bool(labels[sample_index]),
                "probability": {key: float(value[local_index]) for key, value in probability_test.items()},
                "threshold": thresholds,
                "predicted_token_gate": float(token_gate_test[local_index]),
                "predicted_geometry_token_gate": float(combined_gate_test[local_index]),
                "optimal_torso_gate": float(gate_targets[sample_index]),
                "methods": methods,
            }
        )
    return {
        "held_out_source": held_out,
        "train_count": int(len(train_ids)),
        "test_count": int(len(test_ids)),
        "thresholds": thresholds,
        "confidence": {
            key: confidence_metrics(probability_test[key], labels[test_ids].astype(bool))
            for key in probability_test
        },
        "rows": rows,
    }


def failed(row: dict | None) -> bool:
    return row is None or bool(row.get("fit_failed", False)) or not np.isfinite(row.get("camera_rotation_error_deg", np.nan))


def catastrophic(row: dict | None) -> bool:
    return failed(row) or float(row["camera_translation_error_m"]) > 1.0 or float(row["camera_rotation_error_deg"]) > 30.0


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_rows(rows: list[dict]) -> dict:
    methods = sorted(rows[0]["methods"])
    output = {}
    for method in methods:
        values = [row["methods"][method] for row in rows]
        base = [row["methods"]["fixed_explicit"] for row in rows]
        rotation_gain = np.asarray(
            [float(a["camera_rotation_error_deg"]) - float(b["camera_rotation_error_deg"]) for a, b in zip(base, values)]
        )
        false = np.asarray(
            [
                float(a["camera_rotation_error_deg"]) < 10.0
                and float(b["camera_rotation_error_deg"]) > float(a["camera_rotation_error_deg"]) + 1.0
                for a, b in zip(base, values)
            ]
        )
        output[method] = {
            "count": len(values),
            "translation_m": distribution([float(row["camera_translation_error_m"]) for row in values]),
            "rotation_deg": distribution([float(row["camera_rotation_error_deg"]) for row in values]),
            "catastrophic_rate": float(np.mean([catastrophic(row) for row in values])),
            "rotation_gain_mean_deg": float(rotation_gain.mean()),
            "rotation_helpful_rate": float(np.mean(rotation_gain > 0.5)),
            "rotation_harmful_rate": float(np.mean(rotation_gain < -0.5)),
            "false_correction_rate_on_lt10": float(false.mean()),
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    methods = (
        "fixed_explicit",
        "torso_geometry",
        "geometry_confidence_select",
        "token_confidence_select",
        "geometry_token_confidence_select",
        "token_direct_residual",
        "torso_token_gate",
        "torso_geometry_token_gate",
        "boundary_oracle",
    )
    lines = [
        "# V16 LOSO Human-Token Probe",
        "",
        "| Method | T mean | R mean | R P90 | R P95 | Catastrophic | Helpful | Harmful | False correction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = report["overall"][method]
        lines.append(
            f"| {method} | {row['translation_m']['mean']:.3f} | {row['rotation_deg']['mean']:.2f} | "
            f"{row['rotation_deg']['p90']:.2f} | {row['rotation_deg']['p95']:.2f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% | {100.0 * row['rotation_helpful_rate']:.1f}% | "
            f"{100.0 * row['rotation_harmful_rate']:.1f}% | {100.0 * row['false_correction_rate_on_lt10']:.1f}% |"
        )
    lines.extend(["", "## Held-Out Sources", ""])
    for source, metrics in report["by_source"].items():
        fixed = metrics["fixed_explicit"]
        torso = metrics["torso_geometry"]
        token = metrics["geometry_token_confidence_select"]
        lines.append(
            f"- **{source}**: Fixed `{fixed['rotation_deg']['mean']:.2f} deg`; "
            f"torso `{torso['rotation_deg']['mean']:.2f} deg`; geometry+token select `{token['rotation_deg']['mean']:.2f} deg`."
        )
    lines.extend(
        [
            "",
            "## Confidence Generalization",
            "",
        ]
    )
    for fold in report["folds"]:
        lines.append(
            f"- **{fold['held_out_source']}**: geometry AUROC `{fold['confidence']['geometry']['auroc']}`; "
            f"token AUROC `{fold['confidence']['token']['auroc']}`; "
            f"geometry+token AUROC `{fold['confidence']['geometry_token']['auroc']}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V16 learned LOSO modules must train on CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(int(args.seed))
    samples = load_samples(args.cache_dir, args.v10_report)
    token_raw = np.stack([token_feature(sample) for sample in samples])
    geometry_raw = np.stack([geometry_feature(sample) for sample in samples])
    folds = [run_fold(samples, token_raw, geometry_raw, source, args, index) for index, source in enumerate(SOURCES)]
    rows = [row for fold in folds for row in fold["rows"]]
    overall = aggregate_rows(rows)
    by_source = {
        source: aggregate_rows([row for row in rows if row["source"] == source]) for source in SOURCES
    }
    report = {
        "experiment": "V16 LOSO Human-Token Residual and Confidence Probe",
        "case_count": len(rows),
        "protocol": {
            "leave_one_source_out": True,
            "human3r_frozen": True,
            "training_device": str(args.device),
            "gt_camera_use": "rotation target and evaluation only",
            "gt_depth_used": False,
            "translation_from_human_root": False,
            "token_pca_fit": "training sources only per fold",
            "threshold_fit": "training sources only per fold",
            "max_humans": 1,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "overall": overall,
        "by_source": by_source,
        "folds": folds,
    }
    path = args.output_dir / "v16_token_loso_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v16_token_loso_summary.md", report)
    print(json.dumps({"overall": overall, "by_source": by_source}, indent=2), flush=True)
    print(f">> wrote {path}", flush=True)


if __name__ == "__main__":
    main()
