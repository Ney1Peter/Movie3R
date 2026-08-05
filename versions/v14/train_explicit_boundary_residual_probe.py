#!/usr/bin/env python3
"""Train/evaluate small direct-SE(3) residual heads on frozen causal features.

This is a deliberately narrow Phase-4 experiment, not a replacement B0
retraining.  The Human3R/cross96 checkpoint stays frozen.  A head sees only
the cached first-post-cut latent/geometry features and predicts a right-side
local residual ``Delta``.  Runtime composition is ``B = B0 @ Delta``.

Development rules are fixed before reading confirmation data:

* train only on cross96 train96 pairs;
* select among one linear and two small MLP heads on pair-disjoint VSP dev;
* open confirmation only if a candidate improves mean by >=5%, improves the
  catastrophic tail by >=20%, preserves P95, and does not harm any source;
* otherwise retain exact B0 and record a No-Go.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "output/v14/fine_alignment_research/explicit_boundary_residual_features"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/explicit_boundary_residual_probe"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
SEED = 20260803
TRANSLATION_CAP_M = 3.0
ROTATION_CAP_RAD = float(np.pi)
ROTATION_METRIC_WEIGHT = float(.02 * 180.0 / np.pi)  # m-equivalent per radian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
    parser.add_argument("--train-report", type=Path, default=DEFAULT_ROOT / "train96/report.json")
    parser.add_argument("--dev-report", type=Path, default=DEFAULT_ROOT / "vsp_dev/report.json")
    parser.add_argument("--confirm-report", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def so3_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64) + skew(vector)
    hat = skew(vector / angle)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * hat + (1.0 - np.cos(angle)) * (hat @ hat)


def rotation_error_deg(camera: np.ndarray, target: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(camera[:3, :3].T @ target[:3, :3]) - 1.0) * .5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def camera_metrics(camera: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    translation = float(np.linalg.norm(camera[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(camera, target)
    return {"translation_m": translation, "rotation_deg": rotation, "composite": translation + .02 * rotation, "catastrophic": bool(translation > 1.0 or rotation > 30.0)}


def delta_matrix(vector: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = so3_exp(vector[3:])
    result[:3, 3] = vector[:3]
    return result


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("failures"):
        raise RuntimeError(f"{path} contains forward failures")
    if not payload.get("cases"):
        raise RuntimeError(f"{path} has no cases")
    return payload


def arrays(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features = np.asarray([row["feature"] for row in report["cases"]], dtype=np.float32)
    targets = np.asarray([row["target_right_se3_training_only"] for row in report["cases"]], dtype=np.float32)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise RuntimeError("Features/targets contain non-finite values")
    return features, targets, report["cases"]


class ResidualHead(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        if hidden == 0:
            self.net = nn.Linear(dim, 6)
        else:
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 6))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        raw = self.net(feature)
        # A bounded proposal is a transactional trust region, not label
        # clipping: labels outside the cap still contribute saturated loss.
        return torch.cat((TRANSLATION_CAP_M * torch.tanh(raw[:, :3]), ROTATION_CAP_RAD * torch.tanh(raw[:, 3:])), dim=-1)


@dataclass(frozen=True)
class Candidate:
    identifier: str
    hidden: int
    learning_rate: float
    weight_decay: float
    epochs: int


# Predeclared small capacity sweep; no per-source architecture exists.
CANDIDATES = (
    Candidate("linear_l2", 0, 2e-3, 2e-3, 800),
    Candidate("mlp64", 64, 1e-3, 2e-3, 1000),
    Candidate("mlp128", 128, 8e-4, 3e-3, 1000),
)


def fit(candidate: Candidate, x_train: np.ndarray, y_train: np.ndarray) -> tuple[ResidualHead, dict[str, Any]]:
    seed_everything(SEED)
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-5] = 1.0
    x = torch.from_numpy((x_train - mean) / std)
    y = torch.from_numpy(y_train)
    model = ResidualHead(x.shape[1], candidate.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay)
    generator = torch.Generator().manual_seed(SEED)
    final_loss = float("nan")
    for _ in range(candidate.epochs):
        permutation = torch.randperm(len(x), generator=generator)
        for indices in permutation.split(48):
            prediction = model(x[indices])
            translation_loss = F.smooth_l1_loss(prediction[:, :3], y[indices, :3], beta=.5)
            rotation_loss = F.smooth_l1_loss(prediction[:, 3:], y[indices, 3:], beta=.35)
            loss = translation_loss + ROTATION_METRIC_WEIGHT * rotation_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            final_loss = float(loss.detach())
    return model.eval(), {"feature_mean": mean.squeeze(0).tolist(), "feature_std": std.squeeze(0).tolist(), "final_train_batch_loss": final_loss}


def predict(model: ResidualHead, normalization: dict[str, Any], feature: np.ndarray) -> np.ndarray:
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(((feature - mean) / std)[None])).cpu().numpy()[0].astype(np.float64)


def summarize(rows: list[dict[str, Any]], predicted: np.ndarray) -> dict[str, Any]:
    def aggregate(indices: np.ndarray) -> dict[str, Any]:
        b0, residual = [], []
        for index in indices:
            row = rows[int(index)]
            b0_camera = np.asarray(row["b0_camera"], dtype=np.float64)
            target = np.asarray(row["target_camera_evaluation_only"], dtype=np.float64)
            b0.append(camera_metrics(b0_camera, target))
            residual.append(camera_metrics(b0_camera @ delta_matrix(predicted[int(index)]), target))
        def metric(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "translation_m": float(np.mean([item["translation_m"] for item in items])),
                "rotation_deg": float(np.mean([item["rotation_deg"] for item in items])),
                "composite": {"mean": float(np.mean([item["composite"] for item in items])), "p95": float(np.quantile([item["composite"] for item in items], .95))},
                "catastrophic_count": int(sum(item["catastrophic"] for item in items)),
            }
        base, head = metric(b0), metric(residual)
        return {
            "count": int(len(indices)), "b0": base, "head": head,
            "relative_mean_gain": float(1.0 - head["composite"]["mean"] / base["composite"]["mean"]),
            "catastrophic_relative_reduction": float(1.0 - head["catastrophic_count"] / max(base["catastrophic_count"], 1)),
        }
    all_indices = np.arange(len(rows))
    return {"overall": aggregate(all_indices), "by_source": {source: aggregate(np.asarray([index for index, row in enumerate(rows) if row["source"] == source])) for source in SOURCES}}


def qualifies(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    overall = summary["overall"]
    checks = {
        "mean_gain_at_least_5pct": overall["relative_mean_gain"] >= .05,
        "p95_noninferior": overall["head"]["composite"]["p95"] <= overall["b0"]["composite"]["p95"] + 1e-12,
        "catastrophic_reduction_at_least_20pct": overall["catastrophic_relative_reduction"] >= .20,
        "every_source_mean_noninferior": all(summary["by_source"][source]["relative_mean_gain"] >= -1e-12 for source in SOURCES),
        "at_least_two_source_gain_5pct": sum(summary["by_source"][source]["relative_mean_gain"] >= .05 for source in SOURCES) >= 2,
        "every_source_catastrophic_noninferior": all(summary["by_source"][source]["head"]["catastrophic_count"] <= summary["by_source"][source]["b0"]["catastrophic_count"] for source in SOURCES),
    }
    return all(checks.values()), checks


def state_json(model: ResidualHead) -> dict[str, Any]:
    return {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}


def load_state(candidate: Candidate, dimension: int, payload: dict[str, Any]) -> ResidualHead:
    model = ResidualHead(dimension, candidate.hidden)
    model.load_state_dict({key: torch.tensor(value, dtype=torch.float32) for key, value in payload.items()})
    return model.eval()


def dev(args: argparse.Namespace) -> None:
    train, development = load_report(args.train_report), load_report(args.dev_report)
    if train["feature_schema"] != development["feature_schema"] or train["feature_dimension"] != development["feature_dimension"]:
        raise RuntimeError("Train/dev frozen feature schemas differ")
    x_train, y_train, train_rows = arrays(train)
    x_dev, _, dev_rows = arrays(development)
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in CANDIDATES:
        model, normalization = fit(candidate, x_train, y_train)
        dev_prediction = np.vstack([predict(model, normalization, feature) for feature in x_dev])
        train_prediction = np.vstack([predict(model, normalization, feature) for feature in x_train])
        dev_summary = summarize(dev_rows, dev_prediction)
        qualified, checks = qualifies(dev_summary)
        records.append({
            "candidate": candidate.__dict__, "train_summary": summarize(train_rows, train_prediction),
            "development_summary": dev_summary, "qualified": qualified, "checks": checks,
            "normalization": normalization, "state_dict": state_json(model),
        })
    records.sort(key=lambda item: (item["development_summary"]["overall"]["head"]["composite"]["mean"], item["candidate"]["hidden"]))
    qualified = [item for item in records if item["qualified"]]
    report = {
        "experiment": "explicit_boundary_residual_head_pair_disjoint_development",
        "method": "frozen cross96 causal latent/geometry features -> bounded right-composed SE(3) residual",
        "train_input": str(args.train_report), "development_input": str(args.dev_report),
        "feature_schema": train["feature_schema"], "feature_dimension": train["feature_dimension"],
        "training_only_target": "inverse(B0) @ B_gt in raw post-shot gauge",
        "caps": {"translation_m": TRANSLATION_CAP_M, "rotation_deg": float(np.degrees(ROTATION_CAP_RAD))},
        "selection_rules": "fixed in script docstring and qualifies(); confirmation untouched when no candidate qualifies",
        "qualified_count": len(qualified), "candidates_ranked": records,
    }
    (args.output_dir / "DEV_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not qualified:
        print("NO_GO_EXPLICIT_BOUNDARY_RESIDUAL_LATENT_PROBE")
        return
    winner = qualified[0]
    policy = {
        "freeze_id": "EXPLICIT_BOUNDARY_RESIDUAL_HEAD_V1_20260803",
        "status": "frozen_after_pair_disjoint_dev_before_confirmation",
        "method": report["method"], "candidate": winner["candidate"], "normalization": winner["normalization"], "state_dict": winner["state_dict"],
        "caps": report["caps"], "development_summary": winner["development_summary"], "checks": winner["checks"],
        "fallback": "exact cross96 B0", "future_frames": 0, "shadow_state_commit": False,
        "confirmation_status": "not_run",
    }
    (args.output_dir / "FROZEN_POLICY_BEFORE_CONFIRM.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner["candidate"], "development": winner["development_summary"]}, indent=2))


def confirm(args: argparse.Namespace) -> None:
    if args.confirm_report is None or args.policy is None:
        raise ValueError("confirm requires --confirm-report and --policy")
    report, policy = load_report(args.confirm_report), json.loads(args.policy.read_text(encoding="utf-8"))
    candidate = Candidate(**policy["candidate"])
    features, _, rows = arrays(report)
    model = load_state(candidate, features.shape[1], policy["state_dict"])
    prediction = np.vstack([predict(model, policy["normalization"], feature) for feature in features])
    summary = summarize(rows, prediction)
    qualified, checks = qualifies(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CONFIRMATION.json").write_text(json.dumps({
        "experiment": "explicit boundary residual head confirmation", "input": str(args.confirm_report),
        "policy": str(args.policy), "summary": summary, "qualified": qualified, "checks": checks,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"qualified": qualified, "checks": checks, "summary": summary}, indent=2))


def main() -> None:
    args = parse_args()
    dev(args) if args.phase == "dev" else confirm(args)


if __name__ == "__main__":
    main()
