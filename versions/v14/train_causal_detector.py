#!/usr/bin/env python3
"""Train and audit small causal shot detectors on the frozen V10 image ledger.

The ledger contains only adjacent RGB-pair features.  This script adds a
strict leave-one-source-out comparison between the existing static logistic
baseline and two causal temporal heads (an MLP over the last ``history``
pairs and a small GRU).  No camera/SMPL/GT feature is admitted to the model;
the ``transition_angle_deg`` column is deliberately excluded.

The output is an audit artifact, not an automatic replacement of the frozen
detector.  A model is promotable only when its held-out source macro F1 and
false-positive rate are both no worse than the static baseline, and its
calibration error is not worse by more than 0.02.  The final all-source model
is exported only after that comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "output/archive/20260721/v10_detector_probe/image_feature_round1/detector_pair_features.csv"
IMAGE_FEATURES = (
    "ahash_hamming", "blur_l1", "dhash_hamming", "edge_l1", "flow_mean",
    "flow_median", "flow_p95", "gray_l1", "gray_l2", "gray_ncc_change",
    "hsv_hist_chisq", "orb_good_matches", "orb_good_ratio",
    "orb_homography_inlier_ratio", "orb_kp0", "orb_kp1", "orb_mean_dist",
    "rgb_hist_chisq", "rgb_l1", "rgb_l2",
)
SEED = 20260805


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "output/v14/detector_learning_audit")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def grouped(rows_in: list[dict[str, str]], history: int) -> list[dict]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows_in:
        groups.setdefault(row["pattern_id"], []).append(row)
    output = []
    for key, group in groups.items():
        group.sort(key=lambda r: int(r["pair_idx"]))
        source = group[0]["source"]
        x = np.asarray([[float(r[name]) for name in IMAGE_FEATURES] for r in group], dtype=np.float32)
        y = np.asarray([int(r["label"]) for r in group], dtype=np.int64)
        output.append({"id": key, "source": source, "x": x, "y": y})
    return output


def flat_examples(groups: list[dict], history: int) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    xx, yy, ids = [], [], []
    dim = len(IMAGE_FEATURES)
    for group in groups:
        for i, label in enumerate(group["y"]):
            start = max(0, i - history + 1)
            hist = group["x"][start:i + 1]
            padded = np.zeros((history, dim), dtype=np.float32)
            padded[-len(hist):] = hist
            xx.append(padded.reshape(-1)); yy.append(int(label)); ids.append((group["id"], i))
    return np.asarray(xx, dtype=np.float32), np.asarray(yy, dtype=np.int64), ids


class TemporalHead(nn.Module):
    def __init__(self, dim: int, history: int, kind: str) -> None:
        super().__init__(); self.kind = kind; self.history = history
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim * history, 48), nn.GELU(), nn.Linear(48, 1))
        elif kind == "gru":
            self.rnn = nn.GRU(dim, 32, batch_first=True); self.net = nn.Linear(32, 1)
        else: raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp": return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)
        return self.net(self.rnn(x)[0][:, -1]).squeeze(-1)


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float = .5) -> dict:
    pred = (prob >= threshold).astype(np.int64)
    return {
        "count": int(len(y)), "positive": int(y.sum()), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
        "fpr": float(((pred == 1) & (y == 0)).sum() / max(int((y == 0).sum()), 1)),
        "brier": float(brier_score_loss(y, prob)),
    }


def fit_torch(kind: str, x: np.ndarray, y: np.ndarray, history: int, epochs: int) -> tuple[TemporalHead, np.ndarray, np.ndarray]:
    seed(); mean, std = x.mean(0), x.std(0); std[std < 1e-6] = 1.
    xn = (x - mean) / std
    model = TemporalHead(len(IMAGE_FEATURES), history, kind)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    tx, ty = torch.from_numpy(xn.reshape(-1, history, len(IMAGE_FEATURES))), torch.from_numpy(y.astype(np.float32))
    for _ in range(epochs):
        logits = model(tx); loss = nn.functional.binary_cross_entropy_with_logits(logits, ty)
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval(), mean, std


def torch_predict(model: TemporalHead, x: np.ndarray, mean: np.ndarray, std: np.ndarray, history: int) -> np.ndarray:
    xn = (x - mean) / std
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(xn.reshape(-1, history, len(IMAGE_FEATURES))))).numpy()


def evaluate_fold(train: list[dict], test: list[dict], history: int, epochs: int) -> dict:
    xtr, ytr, _ = flat_examples(train, history); xte, yte, ids = flat_examples(test, history)
    static = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    static.fit(xtr, ytr); p_static = static.predict_proba(xte)[:, 1]
    result = {"static_logistic": metrics(yte, p_static)}
    for kind in ("mlp", "gru"):
        model, mean, std = fit_torch(kind, xtr, ytr, history, epochs)
        result[kind] = metrics(yte, torch_predict(model, xte, mean, std, history))
    result["source"] = test[0]["source"] if test else "unknown"
    result["test_groups"] = len(test); result["train_groups"] = len(train)
    return result


def main() -> None:
    a = args(); seed(); out = a.output.resolve()
    if out.exists() and not a.overwrite: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    grouped_rows = grouped(rows(a.csv.resolve()), a.history)
    source_names = sorted({g["source"] for g in grouped_rows})
    folds = []
    for held in source_names:
        train = [g for g in grouped_rows if g["source"] != held]
        test = [g for g in grouped_rows if g["source"] == held]
        folds.append(evaluate_fold(train, test, a.history, a.epochs))
    def macro(name: str, metric: str) -> float:
        return float(np.mean([fold[name][metric] for fold in folds]))
    summary = {name: {metric: macro(name, metric) for metric in ("f1", "fpr", "brier", "accuracy", "recall")} for name in ("static_logistic", "mlp", "gru")}
    best = "static_logistic"
    for name in ("mlp", "gru"):
        if summary[name]["f1"] >= summary[best]["f1"] and summary[name]["fpr"] <= summary[best]["fpr"] and summary[name]["brier"] <= summary[best]["brier"] + .02:
            best = name
    # Fit the selected temporal head once on the complete frozen ledger.  The
    # artifact is still only an image-statistics model; it is exported for an
    # optional runtime integration after the held-out audit has finished.
    if best in {"mlp", "gru"}:
        x_all, y_all, _ = flat_examples(grouped_rows, a.history)
        selected_model, selected_mean, selected_std = fit_torch(best, x_all, y_all, a.history, a.epochs)
        torch.save({
            "kind": best, "history": a.history, "feature_names": list(IMAGE_FEATURES),
            "mean": selected_mean, "std": selected_std,
            "state_dict": selected_model.state_dict(),
            "seed": SEED,
        }, out / "SELECTED_MODEL.pt")
    report = {
        "experiment": "causal_detector_leave_one_source_out",
        "input": str(a.csv.resolve()), "feature_names": list(IMAGE_FEATURES),
        "excluded_features": ["transition_angle_deg", "source", "pattern_id", "future_frames"],
        "history": a.history, "epochs": a.epochs, "seed": SEED,
        "folds": folds, "macro": summary,
        "selection": {"selected": best, "rule": "max macro F1 subject to FPR non-inferior and Brier delta <= 0.02"},
        "runtime": "causal adjacent RGB statistics only; no Human3R, camera, SMPL, GT or future frame",
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "REPORT.md").write_text("# Causal detector learning audit\n\n" + json.dumps({"macro": summary, "selection": report["selection"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"macro": summary, "selection": report["selection"]}, indent=2))


if __name__ == "__main__": main()
