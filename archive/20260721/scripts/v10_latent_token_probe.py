#!/usr/bin/env python3
"""Train frozen-feature Linear/MLP probes on the V10 latent token cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "output" / "v10_latent_token_probe" / "token_cache" / "cache_index.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_latent_token_probe" / "probe_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_index", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max_pca_dim", type=int, default=64)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--mlp_hidden", type=int, nargs="*", default=(64, 32))
    parser.add_argument("--mlp_max_iter", type=int, default=400)
    parser.add_argument("--skip_mlp", action="store_true")
    parser.add_argument("--skip_linear", action="store_true")
    parser.add_argument("--skip_patch_probe", action="store_true")
    parser.add_argument("--feature_names", nargs="*", default=None)
    parser.add_argument("--task_names", nargs="*", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class CachedCase:
    metadata: dict
    arrays: dict[str, np.ndarray]

    @property
    def source(self) -> str:
        return str(self.metadata["record"]["source"])

    @property
    def group(self) -> str:
        return str(self.metadata["record"].get("group", self.metadata["record"].get("seqA", "unknown")))


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    domain: str
    array_key: str | None = None
    layer: int | None = None
    components: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    kind: str
    target_key: str
    component: tuple[int, ...] | None = None


def load_cases(path: Path) -> list[CachedCase]:
    cases = []
    for metadata in read_jsonl(path):
        cache_path = Path(metadata["cache_path"])
        if not cache_path.is_file():
            continue
        with np.load(cache_path) as data:
            arrays = {key: data[key].astype(np.float32) for key in data.files}
        cases.append(CachedCase(metadata, arrays))
    if not cases:
        raise RuntimeError(f"No readable token caches in {path}")
    return cases


def split_cases(cases: list[CachedCase]) -> dict[str, set[int]]:
    zero = {idx for idx, case in enumerate(cases) if case.source == "avatarrex"}
    non_zero: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, case in enumerate(cases):
        if idx in zero:
            continue
        non_zero[case.source][case.group].append(idx)
    validation = set()
    train = set()
    split_debug = {}
    for source, groups in non_zero.items():
        names = sorted(groups)
        if not names:
            continue
        holdout = names[-1]
        validation.update(groups[holdout])
        for name in names[:-1]:
            train.update(groups[name])
        split_debug[source] = {"train_groups": names[:-1], "validation_groups": [holdout]}
    if not train or not validation:
        raise RuntimeError(
            f"Insufficient scene-level split: train={len(train)}, validation={len(validation)}, zero={len(zero)}"
        )
    return {"train": train, "validation": validation, "zero_shot_avatarrex": zero, "debug": split_debug}


def frame_feature_specs(cases: list[CachedCase]) -> list[FeatureSpec]:
    sample = cases[0].arrays
    specs = []
    for layer in range(sample["encoder_layer_pool"].shape[1]):
        specs.append(FeatureSpec(f"encoder_image_l{layer:02d}", "frame", "encoder_layer_pool", layer))
    for layer in range(sample["decoder_image_layer_pool"].shape[1]):
        specs.append(FeatureSpec(f"decoder_image_l{layer:02d}", "frame", "decoder_image_layer_pool", layer))
    for layer in range(sample["decoder_state_layer_pool"].shape[1]):
        specs.append(FeatureSpec(f"decoder_state_l{layer:02d}", "frame", "decoder_state_layer_pool", layer))
    specs.extend(
        [
            FeatureSpec("camera_initial", "frame", "camera_initial"),
            FeatureSpec("camera_refined", "frame", "camera_refined"),
            FeatureSpec("human_prompt", "frame", "human_prompt_pool"),
            FeatureSpec("human_refined", "frame", "human_refined_pool"),
            FeatureSpec("persistent_state", "frame", "persistent_state_pool"),
            FeatureSpec("new_state", "frame", "new_state_pool"),
            FeatureSpec("dino_global", "frame", "dino_pool"),
        ]
    )
    return specs


def boundary_feature_specs(cases: list[CachedCase]) -> list[FeatureSpec]:
    frame_specs = frame_feature_specs(cases)
    specs = [FeatureSpec(spec.name, "boundary", spec.array_key, spec.layer) for spec in frame_specs]
    specs.extend(
        [
            FeatureSpec("image_plus_state", "boundary", components=("decoder_image_l11", "new_state")),
            FeatureSpec("image_plus_human", "boundary", components=("decoder_image_l11", "human_refined")),
            FeatureSpec("state_plus_human", "boundary", components=("new_state", "human_refined")),
            FeatureSpec(
                "image_state_human",
                "boundary",
                components=("decoder_image_l11", "new_state", "human_refined"),
            ),
        ]
    )
    return specs


def patch_feature_specs(cases: list[CachedCase]) -> list[FeatureSpec]:
    sample = cases[0].arrays
    specs = []
    for layer in range(sample["encoder_layer_patch"].shape[1]):
        specs.append(FeatureSpec(f"encoder_patch_l{layer:02d}", "patch", "encoder_layer_patch", layer))
    for layer in range(sample["decoder_layer_patch"].shape[1]):
        specs.append(FeatureSpec(f"decoder_patch_l{layer:02d}", "patch", "decoder_layer_patch", layer))
    specs.append(FeatureSpec("dino_patch", "patch", "dino_patch"))
    return specs


FRAME_TASKS = (
    TaskSpec("camera_absolute_rotation", "frame", "regression", "camera_gt_euler_zyx_deg"),
    TaskSpec("camera_relative_rotation", "frame", "regression", "camera_gt_relative_euler_zyx_deg"),
    TaskSpec("camera_relative_translation", "frame", "regression", "camera_gt_relative_translation"),
    TaskSpec("camera_distance", "frame", "regression", "camera_gt_distance_from_first"),
    TaskSpec("human_world_root", "frame", "regression", "human_gt_world_root"),
    TaskSpec("human_torso_heading", "frame", "regression", "human_gt_torso_heading"),
    TaskSpec("human_root_velocity", "frame", "regression", "human_gt_root_velocity"),
    TaskSpec("human_angular_velocity", "frame", "regression", "human_gt_angular_velocity_deg"),
)

BOUNDARY_TASKS = (
    TaskSpec("boundary_rotation", "boundary", "regression", "boundary_euler_zyx_deg"),
    TaskSpec("boundary_translation_direction", "boundary", "regression", "boundary_translation_direction"),
    TaskSpec("boundary_translation_norm", "boundary", "regression", "boundary_translation_norm"),
    TaskSpec("explicit_translation_error", "boundary", "regression", "metadata:explicit_translation_error_m"),
    TaskSpec("explicit_rotation_error", "boundary", "regression", "metadata:explicit_rotation_error_deg"),
    TaskSpec("explicit_failure", "boundary", "classification", "metadata:explicit_failure_relaxed"),
    TaskSpec("explicit_catastrophic", "boundary", "classification", "metadata:explicit_catastrophic"),
)

PATCH_TASKS = (
    TaskSpec("scene_depth_pred", "patch", "regression", "patch_depth_pred"),
    TaskSpec("scene_camera_coordinate_pred", "patch", "regression", "patch_camera_point_pred"),
    TaskSpec("scene_world_coordinate_pred", "patch", "regression", "patch_world_point_pred"),
    TaskSpec("scene_surface_normal_pred", "patch", "regression", "patch_normal_pred"),
    TaskSpec("pointmap_confidence_pred", "patch", "regression", "patch_confidence_pred"),
    TaskSpec("scene_class_pseudo", "patch", "classification", "patch_scene_class_pseudo"),
    TaskSpec("static_background", "patch", "classification", "patch_static_background"),
    TaskSpec("alignment_suitable_pseudo", "patch", "classification", "patch_alignment_suitable_pseudo"),
)


def raw_frame_feature(case: CachedCase, spec: FeatureSpec) -> np.ndarray:
    array = case.arrays[spec.array_key]
    return array[:, spec.layer] if spec.layer is not None else array


def named_frame_feature(case: CachedCase, name: str) -> np.ndarray:
    for spec in frame_feature_specs([case]):
        if spec.name == name:
            return raw_frame_feature(case, spec)
    raise KeyError(name)


def pair_encode(values: np.ndarray, boundary: int) -> np.ndarray:
    before = values[boundary - 1].reshape(-1)
    after = values[boundary].reshape(-1)
    return np.concatenate([before, after, after - before, np.abs(after - before)]).astype(np.float32)


def feature_for_case(case: CachedCase, spec: FeatureSpec) -> np.ndarray:
    boundary = int(case.metadata["boundary"])
    if spec.domain == "frame":
        return raw_frame_feature(case, spec).reshape(len(case.arrays[spec.array_key]), -1).astype(np.float32)
    if spec.domain == "boundary":
        if spec.components:
            return np.concatenate([pair_encode(named_frame_feature(case, name), boundary) for name in spec.components])
        return pair_encode(raw_frame_feature(case, spec), boundary)
    array = case.arrays[spec.array_key]
    if spec.layer is not None:
        array = array[:, spec.layer]
    return array.reshape(-1, array.shape[-1]).astype(np.float32)


def target_for_case(case: CachedCase, task: TaskSpec) -> np.ndarray:
    if task.target_key.startswith("metadata:"):
        value = case.metadata.get(task.target_key.split(":", 1)[1], np.nan)
        return np.asarray([value])
    value = case.arrays[task.target_key]
    if task.domain == "patch":
        if value.ndim >= 3:
            return value.reshape(-1, value.shape[-1])
        return value.reshape(-1)
    if task.domain == "boundary":
        return np.asarray(value).reshape(1, -1) if np.asarray(value).ndim else np.asarray([value])
    return np.asarray(value)


def build_dataset(
    cases: list[CachedCase],
    case_ids: set[int],
    feature: FeatureSpec,
    task: TaskSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    owners = []
    for case_idx in sorted(case_ids):
        case = cases[case_idx]
        x = feature_for_case(case, feature)
        y = target_for_case(case, task)
        if task.domain == "boundary":
            x = x.reshape(1, -1)
            y = np.asarray(y).reshape(1, -1)
        elif task.domain == "frame":
            y = y.reshape(len(x), -1) if y.ndim > 1 else y.reshape(len(x), 1)
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
        if len(x) != len(y):
            raise ValueError(f"Feature/target length mismatch {feature.name}/{task.name}: {len(x)} != {len(y)}")
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        if task.kind == "classification":
            valid &= y[:, 0] >= 0
        features.append(x[valid])
        targets.append(y[valid])
        owners.append(np.full(int(valid.sum()), case_idx, dtype=np.int64))
    if not features or not any(len(value) for value in features):
        return np.empty((0, 0), np.float32), np.empty((0, 1), np.float32), np.empty(0, np.int64)
    return np.concatenate(features), np.concatenate(targets), np.concatenate(owners)


def pca_steps(x: np.ndarray, max_dim: int, seed: int) -> list[tuple]:
    components = min(int(max_dim), x.shape[1], max(1, len(x) - 2))
    steps = [("scale", StandardScaler())]
    if components < x.shape[1]:
        steps.append(("pca", PCA(n_components=components, random_state=seed, svd_solver="randomized")))
    return steps


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_mean: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred).reshape(len(y_pred), -1)
    baseline = np.broadcast_to(train_mean.reshape(1, -1), y_true.shape)
    mae_components = np.mean(np.abs(y_true - y_pred), axis=0)
    baseline_components = np.mean(np.abs(y_true - baseline), axis=0)
    mae = float(mae_components.mean())
    baseline_mae = float(baseline_components.mean())
    return {
        "count": len(y_true),
        "mae": mae,
        "mae_components": mae_components.tolist(),
        "baseline_mae": baseline_mae,
        "skill": float(1.0 - mae / max(baseline_mae, 1e-8)),
        "r2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")) if len(y_true) > 1 else float("nan"),
    }


def classification_metrics(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray | None) -> dict:
    y_true = y_true.reshape(-1).astype(np.int64)
    prediction = prediction.reshape(-1).astype(np.int64)
    classes = np.unique(y_true)
    result = {
        "count": len(y_true),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "classes": classes.tolist(),
    }
    if probability is not None and len(classes) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, probability[:, 1]))
        except ValueError:
            result["roc_auc"] = float("nan")
    else:
        result["roc_auc"] = float("nan")
    chance = 1.0 / max(len(classes), 1)
    result["skill"] = float((result["balanced_accuracy"] - chance) / max(1.0 - chance, 1e-8))
    return result


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kind: str,
    probe_type: str,
    args: argparse.Namespace,
):
    steps = pca_steps(x_train, args.max_pca_dim, args.seed)
    if kind == "regression":
        if probe_type == "linear":
            estimator = Ridge(alpha=float(args.ridge_alpha))
        else:
            estimator = MLPRegressor(
                hidden_layer_sizes=tuple(args.mlp_hidden),
                activation="relu",
                alpha=1e-3,
                max_iter=int(args.mlp_max_iter),
                early_stopping=True,
                validation_fraction=0.15,
                random_state=args.seed,
            )
    else:
        if probe_type == "linear":
            estimator = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=args.seed)
        else:
            estimator = MLPClassifier(
                hidden_layer_sizes=tuple(args.mlp_hidden),
                activation="relu",
                alpha=1e-3,
                max_iter=int(args.mlp_max_iter),
                early_stopping=True,
                validation_fraction=0.15,
                random_state=args.seed,
            )
    pipeline = Pipeline([*steps, ("probe", estimator)])
    target = y_train if kind == "regression" and y_train.shape[1] > 1 else y_train.reshape(-1)
    pipeline.fit(x_train, target)
    return pipeline


def evaluate_feature_task(
    cases: list[CachedCase],
    split: dict,
    feature: FeatureSpec,
    task: TaskSpec,
    args: argparse.Namespace,
) -> list[dict]:
    x_train, y_train, _ = build_dataset(cases, split["train"], feature, task)
    if len(x_train) < 12:
        return []
    if task.kind == "classification" and len(np.unique(y_train)) < 2:
        return []
    outputs = []
    probe_types = []
    if not args.skip_linear:
        probe_types.append("linear")
    if not args.skip_mlp:
        probe_types.append("mlp")
    for probe_type in probe_types:
        if probe_type == "mlp" and len(x_train) < 30:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model = fit_probe(x_train, y_train, task.kind, probe_type, args)
        except (ValueError, np.linalg.LinAlgError) as exc:
            outputs.append(
                {
                    "feature": feature.name,
                    "domain": task.domain,
                    "task": task.name,
                    "kind": task.kind,
                    "probe": probe_type,
                    "error": str(exc),
                }
            )
            continue
        train_mean = np.mean(y_train, axis=0)
        for split_name in ("train", "validation", "zero_shot_avatarrex"):
            x, y, _ = build_dataset(cases, split[split_name], feature, task)
            if len(x) == 0:
                continue
            if task.kind == "regression":
                prediction = np.asarray(model.predict(x)).reshape(len(x), -1)
                metrics = regression_metrics(y, prediction, train_mean)
            else:
                prediction = model.predict(x)
                probability = model.predict_proba(x) if hasattr(model, "predict_proba") else None
                metrics = classification_metrics(y, prediction, probability)
            outputs.append(
                {
                    "feature": feature.name,
                    "domain": task.domain,
                    "task": task.name,
                    "kind": task.kind,
                    "probe": probe_type,
                    "split": split_name,
                    **metrics,
                }
            )
    return outputs


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_lookup(rows: list[dict], domain: str, probe: str, split_name: str) -> dict[tuple[str, str], float]:
    return {
        (row["feature"], row["task"]): float(row.get("skill", np.nan))
        for row in rows
        if row.get("domain") == domain and row.get("probe") == probe and row.get("split") == split_name
    }


def plot_heatmap(
    path: Path,
    rows: list[dict],
    domain: str,
    probe: str,
    split_name: str,
    feature_names: list[str],
    task_names: list[str],
) -> None:
    lookup = score_lookup(rows, domain, probe, split_name)
    matrix = np.asarray([[lookup.get((feature, task), np.nan) for task in task_names] for feature in feature_names])
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(task_names)), max(6, 0.26 * len(feature_names))))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(task_names)), [name.replace("_", "\n") for name in task_names], fontsize=7)
    ax.set_yticks(range(len(feature_names)), feature_names, fontsize=6)
    ax.set_title(f"{domain} {probe} probe: {split_name} skill")
    fig.colorbar(image, ax=ax, label="Skill over constant baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def layer_index(name: str) -> int | None:
    if "_l" not in name:
        return None
    try:
        return int(name.rsplit("_l", 1)[1])
    except ValueError:
        return None


def plot_layer_curves(path: Path, rows: list[dict], domain: str, probe: str, split_name: str) -> None:
    selected_tasks = (
        ("frame", ("camera_relative_rotation", "human_world_root", "human_torso_heading")),
        ("boundary", ("boundary_rotation", "boundary_translation_direction", "explicit_failure")),
        ("patch", ("scene_depth_pred", "scene_world_coordinate_pred", "scene_surface_normal_pred")),
    )
    tasks = dict(selected_tasks).get(domain, ())
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for prefix, label in (("encoder", "Encoder"), ("decoder_image", "Decoder image"), ("decoder_state", "Decoder state"), ("decoder_patch", "Decoder patch"), ("encoder_patch", "Encoder patch")):
            points = []
            for row in rows:
                if row.get("domain") != domain or row.get("probe") != probe or row.get("split") != split_name or row.get("task") != task:
                    continue
                if not row["feature"].startswith(prefix):
                    continue
                idx = layer_index(row["feature"])
                if idx is not None:
                    points.append((idx, float(row.get("skill", np.nan))))
            if points:
                points.sort()
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Skill")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def best_row(rows: list[dict], task: str, split_name: str, probe: str = "linear") -> dict | None:
    valid = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("split") == split_name
        and row.get("probe") == probe
        and np.isfinite(float(row.get("skill", np.nan)))
    ]
    return max(valid, key=lambda row: float(row["skill"])) if valid else None


def architecture_judgement(rows: list[dict]) -> list[str]:
    conclusions = []
    for task, label in (
        ("boundary_rotation", "跨镜头旋转"),
        ("boundary_translation_direction", "跨镜头平移方向"),
        ("explicit_failure", "显式对齐失败判断"),
        ("human_torso_heading", "人体 torso heading"),
        ("scene_world_coordinate_pred", "模型自身 world pointmap"),
    ):
        validation = best_row(rows, task, "validation")
        zero = best_row(rows, task, "zero_shot_avatarrex")
        if validation is None:
            continue
        text = f"{label}在线性 probe 上最强 token 是 {validation['feature']}，未见场景 skill={validation['skill']:.3f}"
        if zero is not None:
            text += f"；AvatarReX 零样本最强为 {zero['feature']}，skill={zero['skill']:.3f}"
        conclusions.append(text + "。")
    failure_geo = next(
        (
            row
            for row in rows
            if row.get("feature") == "image_state_human"
            and row.get("task") == "explicit_failure"
            and row.get("split") == "zero_shot_avatarrex"
            and row.get("probe") == "mlp"
        ),
        None,
    )
    if failure_geo is not None and float(failure_geo.get("skill", -1.0)) > 0.2:
        conclusions.append("token 组合对失败可靠性具有零样本可读性，可优先用于 fallback/等待决策，而非直接回归完整 SE(3)。")
    boundary_zero = best_row(rows, "boundary_rotation", "zero_shot_avatarrex", "mlp")
    if boundary_zero is None or float(boundary_zero.get("skill", -1.0)) <= 0.0:
        conclusions.append("当前 token 对 boundary rotation 未表现出稳定零样本可读性，不应直接训练自由 SE(3) head。")
    conclusions.append("局部 scene patch 任务使用 Human3R 自身 pointmap 作为伪目标，只能说明 head-readout 可解码性，不能替代真实 3D correspondence Recall。")
    return conclusions


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V10 Latent Token Information Probe",
        "",
        f"- Cache cases: `{report['case_count']}`",
        f"- Train cases: `{report['split_sizes']['train']}`",
        f"- Unseen-scene validation: `{report['split_sizes']['validation']}`",
        f"- AvatarReX zero-shot: `{report['split_sizes']['zero_shot_avatarrex']}`",
        "- Human3R encoder、decoder、heads 全部冻结，仅训练 Linear/Small MLP probe。",
        "",
        "## 自动结论",
        "",
    ]
    lines.extend(f"- {item}" for item in report["architecture_judgement"])
    lines.extend(
        [
            "",
            "## 数据边界",
            "",
            "当前 180 个 AABB case 有可靠 camera 与 human GT，但没有统一验证过的静态背景 GT depth/mesh。",
            "因此全局 rotation、translation direction、human、reliability probe 是物理监督；patch depth/world/normal 是 Human3R 输出的 head-readout 伪监督。",
            "真实物理点 Recall@1/5、10/20/50cm 正确率必须等找到可靠 scene geometry 后再报告，当前结果不会冒充该指标。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cache_index)
    split = split_cases(cases)
    frame_features = frame_feature_specs(cases)
    boundary_features = boundary_feature_specs(cases)
    patch_features = [] if args.skip_patch_probe else patch_feature_specs(cases)
    if args.feature_names:
        wanted_features = set(args.feature_names)
        frame_features = [feature for feature in frame_features if feature.name in wanted_features]
        boundary_features = [feature for feature in boundary_features if feature.name in wanted_features]
        patch_features = [feature for feature in patch_features if feature.name in wanted_features]
    frame_tasks = FRAME_TASKS
    boundary_tasks = BOUNDARY_TASKS
    patch_tasks = PATCH_TASKS
    if args.task_names:
        wanted_tasks = set(args.task_names)
        frame_tasks = tuple(task for task in frame_tasks if task.name in wanted_tasks)
        boundary_tasks = tuple(task for task in boundary_tasks if task.name in wanted_tasks)
        patch_tasks = tuple(task for task in patch_tasks if task.name in wanted_tasks)
    rows = []
    jobs = [
        (frame_features, frame_tasks),
        (boundary_features, boundary_tasks),
        (patch_features, patch_tasks),
    ]
    total = sum(len(features) * len(tasks) for features, tasks in jobs)
    done = 0
    for features, tasks in jobs:
        for feature in features:
            for task in tasks:
                done += 1
                print(f">> [{done}/{total}] {feature.name} -> {task.name}", flush=True)
                rows.extend(evaluate_feature_task(cases, split, feature, task, args))
    if not rows:
        raise RuntimeError("No probes could be trained")
    write_rows(args.output_dir / "probe_metrics.csv", rows)
    (args.output_dir / "probe_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    domains = (
        ("frame", frame_features, frame_tasks),
        ("boundary", boundary_features, boundary_tasks),
        ("patch", patch_features, patch_tasks),
    )
    for domain, features, tasks in domains:
        if not features:
            continue
        for probe in ("linear", "mlp"):
            if probe == "mlp" and args.skip_mlp:
                continue
            for split_name in ("validation", "zero_shot_avatarrex"):
                plot_heatmap(
                    args.output_dir / f"{domain}_{probe}_{split_name}_heatmap.png",
                    rows,
                    domain,
                    probe,
                    split_name,
                    [feature.name for feature in features],
                    [task.name for task in tasks],
                )
                plot_layer_curves(
                    args.output_dir / f"{domain}_{probe}_{split_name}_layer_curves.png",
                    rows,
                    domain,
                    probe,
                    split_name,
                )
    report = {
        "case_count": len(cases),
        "split_sizes": {key: len(value) for key, value in split.items() if key != "debug"},
        "split_debug": split["debug"],
        "architecture_judgement": architecture_judgement(rows),
        "physical_patch_correspondence": {
            "available": False,
            "reason": "No uniformly verified static-scene GT depth/mesh in the current 180-case cache.",
        },
    }
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "probe_summary.md", report)
    print(f">> wrote probe results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
