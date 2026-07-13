#!/usr/bin/env python3
"""Probe simple shot-boundary detector signals on V10 pattern manifests.

This script intentionally stays outside the main model.  It compares cheap
pairwise signals that could decide whether frame t starts a new local segment:
pixel/color changes, edge changes, optical-flow magnitude, ORB matching, and a
GT camera-angle upper bound for reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ORDER = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
TRAINING_ROOT = Path("/data/wangzheng/iJCV-CODE/data/Training")
MVHUMAN_ROOT = Path("/data/wangzheng/iJCV-CODE/data/Training/mvhuman")


BASIC_FEATURES = [
    "rgb_l1",
    "rgb_l2",
    "gray_l1",
    "gray_l2",
    "gray_ncc_change",
    "blur_l1",
    "edge_l1",
    "rgb_hist_chisq",
    "hsv_hist_chisq",
    "ahash_hamming",
    "dhash_hamming",
    "flow_mean",
    "flow_median",
    "flow_p95",
]

MATCH_FEATURES = [
    "orb_good_matches",
    "orb_good_ratio",
    "orb_mean_dist",
    "orb_homography_inlier_ratio",
]

ORACLE_FEATURES = ["transition_angle_deg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_pattern_probe",
    )
    parser.add_argument(
        "--long12_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_long12_pattern_probe",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_detector_probe" / "image_feature_round1",
    )
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--orb_size", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def source_root(source: str) -> Path:
    return MVHUMAN_ROOT if source.startswith("mvhuman") else TRAINING_ROOT


def image_path(source: str, seq: str, frame: int) -> Path:
    return source_root(source) / str(seq) / "rgb" / f"{int(frame):08d}.png"


def load_image(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_gray(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def norm_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def gray_ncc_change(g0: np.ndarray, g1: np.ndarray) -> float:
    a = g0.astype(np.float32).reshape(-1)
    b = g1.astype(np.float32).reshape(-1)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    corr = float(np.dot(a, b) / denom)
    return 1.0 - corr


def hist_chisq(hist0: np.ndarray, hist1: np.ndarray) -> float:
    return float(0.5 * np.sum(((hist0 - hist1) ** 2) / (hist0 + hist1 + 1e-8)))


def rgb_hist(img: np.ndarray, bins: int = 16) -> np.ndarray:
    chans = []
    for channel in range(3):
        h = cv2.calcHist([img], [channel], None, [bins], [0, 256]).astype(np.float32).reshape(-1)
        h = h / max(float(h.sum()), 1e-8)
        chans.append(h)
    return np.concatenate(chans)


def hsv_hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256]).astype(np.float32)
    hist = hist.reshape(-1)
    return hist / max(float(hist.sum()), 1e-8)


def ahash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).reshape(-1)


def dhash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a != b)) / float(a.size)


def edge_diff(gray0: np.ndarray, gray1: np.ndarray) -> float:
    e0 = cv2.Canny(gray0, 80, 160).astype(np.float32) / 255.0
    e1 = cv2.Canny(gray1, 80, 160).astype(np.float32) / 255.0
    return float(np.mean(np.abs(e0 - e1)))


def flow_features(gray0: np.ndarray, gray1: np.ndarray) -> dict:
    g0 = gray0.astype(np.float32) / 255.0
    g1 = gray1.astype(np.float32) / 255.0
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=-1)
    return {
        "flow_mean": float(np.mean(mag)),
        "flow_median": float(np.median(mag)),
        "flow_p95": float(np.percentile(mag, 95)),
    }


def orb_features(path0: Path, path1: Path, size: int) -> dict:
    g0 = load_gray(path0, size)
    g1 = load_gray(path1, size)
    orb = cv2.ORB_create(nfeatures=800, fastThreshold=10)
    kp0, des0 = orb.detectAndCompute(g0, None)
    kp1, des1 = orb.detectAndCompute(g1, None)
    n0 = 0 if kp0 is None else len(kp0)
    n1 = 0 if kp1 is None else len(kp1)
    out = {
        "orb_kp0": float(n0),
        "orb_kp1": float(n1),
        "orb_good_matches": 0.0,
        "orb_good_ratio": 0.0,
        "orb_mean_dist": 256.0,
        "orb_homography_inlier_ratio": 0.0,
    }
    if des0 is None or des1 is None or n0 < 4 or n1 < 4:
        return out
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des0, des1, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    out["orb_good_matches"] = float(len(good))
    out["orb_good_ratio"] = float(len(good)) / max(float(min(n0, n1)), 1.0)
    if good:
        out["orb_mean_dist"] = float(np.mean([m.distance for m in good]))
    if len(good) >= 8:
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
        if mask is not None:
            out["orb_homography_inlier_ratio"] = float(mask.reshape(-1).mean())
    return out


def pair_features(path0: Path, path1: Path, image_size: int, orb_size: int) -> dict:
    img0 = load_image(path0, image_size)
    img1 = load_image(path1, image_size)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    f0 = norm_float(img0)
    f1 = norm_float(img1)
    g0 = gray0.astype(np.float32) / 255.0
    g1 = gray1.astype(np.float32) / 255.0
    blur0 = cv2.resize(gray0, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    blur1 = cv2.resize(gray1, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    feats = {
        "rgb_l1": float(np.mean(np.abs(f0 - f1))),
        "rgb_l2": float(np.sqrt(np.mean((f0 - f1) ** 2))),
        "gray_l1": float(np.mean(np.abs(g0 - g1))),
        "gray_l2": float(np.sqrt(np.mean((g0 - g1) ** 2))),
        "gray_ncc_change": gray_ncc_change(gray0, gray1),
        "blur_l1": float(np.mean(np.abs(blur0 - blur1))),
        "edge_l1": edge_diff(gray0, gray1),
        "rgb_hist_chisq": hist_chisq(rgb_hist(img0), rgb_hist(img1)),
        "hsv_hist_chisq": hist_chisq(hsv_hist(img0), hsv_hist(img1)),
        "ahash_hamming": hamming(ahash(gray0), ahash(gray1)),
        "dhash_hamming": hamming(dhash(gray0), dhash(gray1)),
    }
    feats.update(flow_features(gray0, gray1))
    feats.update(orb_features(path0, path1, orb_size))
    return feats


def load_pairs(args: argparse.Namespace) -> list[dict]:
    pairs = []
    roots = [("short4", args.pattern_root), ("long12", args.long12_root)]
    for manifest_set, root in roots:
        for source in SOURCE_ORDER:
            path = root / source / "train_all_patterns.jsonl"
            for record in read_jsonl(path):
                seqs = list(record["seqs"])
                frames = list(record["frames"])
                labels = list(record["shot_labels"])
                angles = list(record.get("transition_angles_deg", [0.0] * len(seqs)))
                pattern = str(record.get("clip_type", "unknown"))
                pattern_id = str(record.get("pattern_id", "unknown"))
                for idx in range(1, len(seqs)):
                    p0 = image_path(source, seqs[idx - 1], int(frames[idx - 1]))
                    p1 = image_path(source, seqs[idx], int(frames[idx]))
                    pairs.append(
                        {
                            "manifest_set": manifest_set,
                            "source": source,
                            "pattern": pattern,
                            "pattern_id": pattern_id,
                            "pair_idx": idx,
                            "seq_prev": seqs[idx - 1],
                            "seq_cur": seqs[idx],
                            "frame_prev": int(frames[idx - 1]),
                            "frame_cur": int(frames[idx]),
                            "label": int(labels[idx]),
                            "transition_angle_deg": float(angles[idx]),
                            "path_prev": str(p0),
                            "path_cur": str(p1),
                        }
                    )
    return pairs


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(np.mean(y_pred[y_true == 0])) if np.any(y_true == 0) else 0.0,
        "positive_rate": float(np.mean(y_pred)),
    }


def best_threshold(values: np.ndarray, labels: np.ndarray) -> tuple[float, int, float]:
    candidates = np.unique(values[np.isfinite(values)])
    if candidates.size == 0:
        return 0.0, 1, 0.0
    if candidates.size > 200:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 200))
    best = (-1.0, 0.0, 1)
    for sign in (1, -1):
        signed = sign * values
        for threshold in candidates:
            pred = (signed >= sign * threshold).astype(np.int64)
            score = f1_score(labels, pred, zero_division=0)
            if score > best[0]:
                best = (float(score), float(threshold), int(sign))
    return best[1], best[2], best[0]


def leave_source_threshold(rows: list[dict], feature: str) -> dict:
    y, p, thresholds, _ = predict_leave_source_threshold(rows, feature)
    out = metrics(y, p)
    out.update({"method": f"threshold:{feature}", "features": feature, "thresholds": thresholds})
    return out


def predict_leave_source_threshold(rows: list[dict], feature: str) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    y_all = []
    p_all = []
    thresholds = {}
    pred_rows = []
    for held_source in SOURCE_ORDER:
        train = [r for r in rows if r["source"] != held_source]
        test = [r for r in rows if r["source"] == held_source]
        train_values = np.asarray([float(r[feature]) for r in train], dtype=np.float64)
        train_labels = np.asarray([int(r["label"]) for r in train], dtype=np.int64)
        threshold, sign, train_f1 = best_threshold(train_values, train_labels)
        test_values = np.asarray([float(r[feature]) for r in test], dtype=np.float64)
        test_labels = np.asarray([int(r["label"]) for r in test], dtype=np.int64)
        pred = ((sign * test_values) >= (sign * threshold)).astype(np.int64)
        y_all.append(test_labels)
        p_all.append(pred)
        for row, pred_value in zip(test, pred):
            pred_rows.append({**row, "pred": int(pred_value), "held_source": held_source})
        thresholds[held_source] = {
            "threshold": threshold,
            "sign": sign,
            "train_f1": train_f1,
        }
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    return y, p, thresholds, pred_rows


def leave_source_model(rows: list[dict], feature_names: list[str], method: str) -> dict:
    y, p, prob, _ = predict_leave_source_model(rows, feature_names, method)
    out = metrics(y, p)
    out.update(
        {
            "method": method,
            "features": ",".join(feature_names),
            "mean_prob": float(np.mean(prob)),
        }
    )
    return out


def predict_leave_source_model(
    rows: list[dict],
    feature_names: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    y_all = []
    p_all = []
    prob_all = []
    pred_rows = []
    for held_source in SOURCE_ORDER:
        train = [r for r in rows if r["source"] != held_source]
        test = [r for r in rows if r["source"] == held_source]
        x_train = np.asarray([[float(r[f]) for f in feature_names] for r in train], dtype=np.float64)
        y_train = np.asarray([int(r["label"]) for r in train], dtype=np.int64)
        x_test = np.asarray([[float(r[f]) for f in feature_names] for r in test], dtype=np.float64)
        y_test = np.asarray([int(r["label"]) for r in test], dtype=np.int64)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        y_all.append(y_test)
        p_all.append(pred)
        prob_all.append(prob)
        for row, pred_value, prob_value in zip(test, pred, prob):
            pred_rows.append(
                {
                    **row,
                    "pred": int(pred_value),
                    "prob": float(prob_value),
                    "held_source": held_source,
                }
            )
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    prob = np.concatenate(prob_all)
    return y, p, prob, pred_rows


def group_metrics(pred_rows: list[dict], method: str, group_keys: list[str]) -> list[dict]:
    out = []
    for group_key in group_keys:
        groups = sorted({str(row[group_key]) for row in pred_rows})
        for group in groups:
            subset = [row for row in pred_rows if str(row[group_key]) == group]
            y = np.asarray([int(row["label"]) for row in subset], dtype=np.int64)
            p = np.asarray([int(row["pred"]) for row in subset], dtype=np.int64)
            row = metrics(y, p)
            row.update(
                {
                    "method": method,
                    "group_type": group_key,
                    "group": group,
                    "pairs": len(subset),
                    "positives": int(y.sum()),
                    "negatives": int(len(y) - y.sum()),
                }
            )
            out.append(row)
    return out


def selected_method_predictions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    specs = [
        ("threshold:rgb_l1", "threshold", ["rgb_l1"]),
        ("threshold:gray_l1", "threshold", ["gray_l1"]),
        ("threshold:hsv_hist_chisq", "threshold", ["hsv_hist_chisq"]),
        ("threshold:orb_good_ratio", "threshold", ["orb_good_ratio"]),
        ("threshold:transition_angle_deg", "threshold", ["transition_angle_deg"]),
        ("logreg:basic_image", "model", BASIC_FEATURES),
        ("logreg:orb_match", "model", MATCH_FEATURES),
        ("logreg:all_image", "model", BASIC_FEATURES + MATCH_FEATURES),
        ("logreg:all_plus_oracle", "model", BASIC_FEATURES + MATCH_FEATURES + ORACLE_FEATURES),
    ]
    pred_rows = []
    metric_rows = []
    for method, kind, features in specs:
        if kind == "threshold":
            _, _, _, method_rows = predict_leave_source_threshold(rows, features[0])
        else:
            _, _, _, method_rows = predict_leave_source_model(rows, features, method)
        compact_rows = []
        for row in method_rows:
            compact = {
                "method": method,
                "manifest_set": row["manifest_set"],
                "source": row["source"],
                "pattern": row["pattern"],
                "pattern_id": row["pattern_id"],
                "pair_idx": row["pair_idx"],
                "label": row["label"],
                "pred": row["pred"],
                "is_error": int(row["label"] != row["pred"]),
                "seq_prev": row["seq_prev"],
                "seq_cur": row["seq_cur"],
                "frame_prev": row["frame_prev"],
                "frame_cur": row["frame_cur"],
            }
            if "prob" in row:
                compact["prob"] = row["prob"]
            compact_rows.append(compact)
        pred_rows.extend(compact_rows)
        metric_rows.extend(group_metrics(compact_rows, method, ["source", "pattern", "manifest_set"]))
    return pred_rows, metric_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(rows: list[dict]) -> dict:
    out = {"total_pairs": len(rows), "positives": int(sum(r["label"] for r in rows))}
    out["negatives"] = out["total_pairs"] - out["positives"]
    by_source = {}
    by_pattern = {}
    for key, target in (("source", by_source), ("pattern", by_pattern)):
        for value in sorted({str(r[key]) for r in rows}):
            subset = [r for r in rows if str(r[key]) == value]
            target[value] = {
                "pairs": len(subset),
                "positives": int(sum(r["label"] for r in subset)),
                "negatives": int(len(subset) - sum(r["label"] for r in subset)),
            }
    out["by_source"] = by_source
    out["by_pattern"] = by_pattern
    return out


def write_markdown(path: Path, counts: dict, results: list[dict]) -> None:
    lines = [
        "# V10 Detector Feature Probe",
        "",
        "Evaluation uses leave-one-source-out validation. Higher F1 is better.",
        "",
        "## Dataset",
        "",
        f"- pairs: {counts['total_pairs']}",
        f"- positives: {counts['positives']}",
        f"- negatives: {counts['negatives']}",
        "",
        "## Top Methods",
        "",
        "| Method | F1 | Precision | Recall | Stable FPR | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda x: x["f1"], reverse=True)[:20]:
        lines.append(
            f"| {row['method']} | {row['f1']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['false_positive_rate']:.3f} | {row['accuracy']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `transition_angle_deg` is an oracle upper bound from GT camera metadata and is not deployable.",
        "- Pixel/color/edge/flow/match features are deployable from input frames.",
        "- Stable false positive rate matters: over-triggering would reset Human3R on normal continuous frames.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args)
    rows = []
    cache = {}
    total = len(pairs)
    for idx, pair in enumerate(pairs, start=1):
        path0 = Path(pair["path_prev"])
        path1 = Path(pair["path_cur"])
        key = (str(path0), str(path1))
        if key not in cache:
            cache[key] = pair_features(path0, path1, int(args.image_size), int(args.orb_size))
        row = dict(pair)
        row.update(cache[key])
        rows.append(row)
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"processed {idx}/{total}", flush=True)

    feature_fields = sorted(set().union(*(row.keys() for row in rows)))
    write_csv(args.output_dir / "detector_pair_features.csv", rows, feature_fields)

    results = []
    all_single = BASIC_FEATURES + MATCH_FEATURES + ORACLE_FEATURES
    for feature in all_single:
        results.append(leave_source_threshold(rows, feature))
    results.append(leave_source_model(rows, BASIC_FEATURES, "logreg:basic_image"))
    results.append(leave_source_model(rows, MATCH_FEATURES, "logreg:orb_match"))
    results.append(leave_source_model(rows, BASIC_FEATURES + MATCH_FEATURES, "logreg:all_image"))
    results.append(leave_source_model(rows, ORACLE_FEATURES, "logreg:oracle_gt_angle"))
    results.append(
        leave_source_model(
            rows,
            BASIC_FEATURES + MATCH_FEATURES + ORACLE_FEATURES,
            "logreg:all_plus_oracle",
        )
    )

    counts = summarize_counts(rows)
    serializable_results = []
    for row in results:
        clean = {k: v for k, v in row.items() if k != "thresholds"}
        clean["thresholds_json"] = json.dumps(row.get("thresholds", {}), sort_keys=True)
        serializable_results.append(clean)
    write_csv(
        args.output_dir / "detector_method_results.csv",
        sorted(serializable_results, key=lambda x: x["f1"], reverse=True),
    )
    pred_rows, group_rows = selected_method_predictions(rows)
    write_csv(args.output_dir / "detector_selected_predictions.csv", pred_rows)
    write_csv(
        args.output_dir / "detector_selected_group_metrics.csv",
        sorted(group_rows, key=lambda x: (x["method"], x["group_type"], x["group"])),
    )
    (args.output_dir / "dataset_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "detector_method_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "detector_feature_probe_summary.md", counts, results)
    print(json.dumps({"counts": counts, "top": sorted(serializable_results, key=lambda x: x["f1"], reverse=True)[:5]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
