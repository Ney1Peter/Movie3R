#!/usr/bin/env python3
"""Extract and rank formal-V9 correction tokens across decoder depth."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
MULTILAYER_PROBE = REPO_ROOT / "versions/v9/multilayer_information_probe"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(MULTILAYER_PROBE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_probe import (  # noqa: E402
    TEN_ROOT,
    camera_matrix_from_prediction,
    collect_records,
    gt_camera,
    make_dataset,
    mlp_fit_predict,
    pose_error,
    prepare_batch,
    record_key,
    relative_pose,
    ridge_fit_predict,
    rotation_to_6d,
    source_for_record,
    summarize_predictions,
    target_vector,
    vector_to_pose,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v9_decoder_correct_token_probe"
DECODER_LAYERS = (2, 5, 8, 11)
TOKEN_NAMES = ("semantic", "alignment", "momentum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-per-source", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--ridge", type=float, default=100.0)
    parser.add_argument("--mlp-steps", type=int, default=400)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    return parser.parse_args()


def pose_from_encoding(encoding: torch.Tensor) -> np.ndarray:
    from dust3r.utils.camera import pose_encoding_to_camera

    return (
        pose_encoding_to_camera(encoding.detach().float())[0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def relation_descriptor(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    pre = pre.reshape(pre.shape[0], -1)[0].float()
    post = post.reshape(post.shape[0], -1)[0].float()
    if pre.shape != post.shape:
        raise ValueError(f"Token shape changed across cut: {pre.shape} vs {post.shape}")
    pre_norm = torch.nn.functional.layer_norm(pre, pre.shape)
    post_norm = torch.nn.functional.layer_norm(post, post.shape)
    scale = float(pre.numel()) ** 0.5
    statistics = torch.stack(
        [
            pre.mean(),
            pre.std(unbiased=False),
            pre.norm() / scale,
            post.mean(),
            post.std(unbiased=False),
            post.norm() / scale,
            (post - pre).norm() / scale,
            torch.nn.functional.cosine_similarity(pre, post, dim=0),
        ]
    )
    return torch.cat(
        [
            pre_norm,
            post_norm,
            post_norm - pre_norm,
            pre_norm * post_norm,
            statistics,
        ]
    ).cpu()


@dataclass
class TokenCapture:
    values: dict[str, list[torch.Tensor]]
    layouts: list[tuple[int, int, int]]
    handles: list[Any]

    @classmethod
    def attach(cls, model) -> "TokenCapture":
        capture = cls(defaultdict(list), [], [])

        def prompt_pre_hook(_module, _args, kwargs):
            image_tokens = kwargs["image_tokens"]
            human_tokens = kwargs.get("human_tokens")
            human_count = (
                int(human_tokens.shape[1])
                if isinstance(human_tokens, torch.Tensor) and human_tokens.ndim == 3
                else 0
            )
            capture.layouts.append((int(image_tokens.shape[1]), human_count, 3))

        def prompt_hook(_module, _args, _kwargs, output):
            corr = output.corr_tokens.detach().float().cpu()
            if corr.shape[1] != 3:
                raise RuntimeError(f"Expected three formal-V9 correction tokens, got {corr.shape}")
            for token_index, token_name in enumerate(TOKEN_NAMES):
                capture.values[f"prompt_{token_name}"].append(corr[:, token_index])
            capture.values["prompt_mean"].append(corr.mean(dim=1))

        capture.handles.append(
            model.v8_pose_prompt.register_forward_pre_hook(prompt_pre_hook, with_kwargs=True)
        )
        capture.handles.append(
            model.v8_pose_prompt.register_forward_hook(prompt_hook, with_kwargs=True)
        )

        for layer_index in DECODER_LAYERS:
            def decoder_hook(_module, _args, output, layer_index=layer_index):
                tokens = output[0] if isinstance(output, (tuple, list)) else output
                if not capture.layouts:
                    raise RuntimeError("Decoder hook ran before V9 prompt layout capture")
                image_count, human_count, corr_count = capture.layouts[-1]
                expected = 1 + corr_count + image_count + human_count
                if tokens.shape[1] != expected:
                    raise RuntimeError(
                        f"Decoder layout mismatch at L{layer_index}: {tokens.shape[1]} vs {expected}"
                    )
                tokens = tokens.detach().float().cpu()
                prefix = f"decoder_l{layer_index:02d}"
                corr = tokens[:, 1:1 + corr_count]
                for token_index, token_name in enumerate(TOKEN_NAMES):
                    capture.values[f"{prefix}_{token_name}"].append(corr[:, token_index])
                capture.values[f"{prefix}_corr_mean"].append(corr.mean(dim=1))
                capture.values[f"{prefix}_pose"].append(tokens[:, 0])
                image_start = 1 + corr_count
                image_end = image_start + image_count
                capture.values[f"{prefix}_image_mean"].append(
                    tokens[:, image_start:image_end].mean(dim=1)
                )
                if human_count:
                    capture.values[f"{prefix}_human_mean"].append(
                        tokens[:, image_end:image_end + human_count].mean(dim=1)
                    )

            capture.handles.append(
                model.dec_blocks[layer_index].register_forward_hook(decoder_hook)
            )
        return capture

    def clear(self) -> None:
        self.values.clear()
        self.layouts.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def configure_formal_v9(model) -> None:
    model.eval()
    model.enable_v8_pose_prompt = True
    model.enable_v8_human_latent_corr = True
    for name, value in (
        ("v9_pre_decoder_change_gate_enabled", False),
        ("v9_raw_pose_step_gate_enabled", False),
        ("v9_clean_raw_pose_step_gate_enabled", False),
        ("v9_oracle_correction_gate_enabled", False),
    ):
        if hasattr(model, name):
            setattr(model, name, value)
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def valid_record_index(records: list[dict], seed: int):
    grouped = defaultdict(list)
    for record in records:
        grouped[source_for_record(record, record.get("source"))].append(record)
    datasets = {source: make_dataset(items, source, seed) for source, items in grouped.items()}
    indices = {
        (source, record_key(sample)): index
        for source, dataset in datasets.items()
        for index, sample in enumerate(dataset.samples)
    }
    entries = []
    for record in records:
        source = source_for_record(record, record.get("source"))
        key = (source, record_key(record))
        if key in indices:
            entries.append((record, source, indices[key]))
        else:
            print(f"skip incomplete record: {record.get('pattern_id', key)}", flush=True)
    return datasets, entries


def extract(args: argparse.Namespace, records: list[dict], model, capture: TokenCapture) -> list[dict]:
    datasets, entries = valid_record_index(records, args.seed)
    rows = []
    for global_index, (record, source, local_index) in enumerate(entries):
        gt_batch, model_batch = prepare_batch(datasets[source], local_index)
        capture.clear()
        started = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            predictions, _ = model.forward_recurrent_lighter(
                model_batch, args.device, ret_state=False, use_ttt3r=False
            )
        elapsed = time.perf_counter() - started
        expected_frames = len(model_batch)
        descriptors = {}
        for name, values in sorted(capture.values.items()):
            if len(values) != expected_frames:
                raise RuntimeError(f"{name} captured {len(values)} frames, expected {expected_frames}")
            descriptors[name] = relation_descriptor(values[-2], values[-1])

        gt_relative = relative_pose(gt_camera(gt_batch[-2]), gt_camera(gt_batch[-1]))
        raw_pre = predictions[-2].get("v8_raw_camera_pose")
        raw_post = predictions[-1].get("v8_raw_camera_pose")
        if raw_pre is None or raw_post is None:
            raise RuntimeError("Formal V9 did not expose v8_raw_camera_pose")
        raw_relative = relative_pose(pose_from_encoding(raw_pre), pose_from_encoding(raw_post))
        full_relative = relative_pose(
            camera_matrix_from_prediction(predictions[-2]),
            camera_matrix_from_prediction(predictions[-1]),
        )
        correction = gt_relative @ np.linalg.inv(raw_relative)
        row = {
            "pattern_id": str(record["pattern_id"]),
            "source": source,
            "split": str(record["split"]),
            "seqs": list(record["seqs"]),
            "frames": list(map(int, record["frames"])),
            "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
            "descriptors": descriptors,
            "target": torch.from_numpy(target_vector(correction)),
            "gt_relative": torch.from_numpy(gt_relative),
            "raw_relative": torch.from_numpy(raw_relative),
            "full_relative": torch.from_numpy(full_relative),
            "raw_error": pose_error(raw_relative, gt_relative),
            "full_error": pose_error(full_relative, gt_relative),
            "elapsed_s": elapsed,
        }
        rows.append(row)
        print(
            f"[{global_index + 1:03d}/{len(entries):03d}] {row['pattern_id']} "
            f"raw={row['raw_error']['composite']:.3f} full={row['full_error']['composite']:.3f} "
            f"{elapsed:.2f}s",
            flush=True,
        )
    return rows


def feature_groups(rows: list[dict]) -> dict[str, tuple[str, ...]]:
    available = tuple(sorted(rows[0]["descriptors"]))
    groups = {name: (name,) for name in available}
    for depth in ("prompt",) + tuple(f"decoder_l{layer:02d}" for layer in DECODER_LAYERS):
        token_features = tuple(f"{depth}_{name}" for name in TOKEN_NAMES)
        if all(name in available for name in token_features):
            groups[f"{depth}_corr_concat"] = token_features
            groups[f"{depth}_sem_align"] = token_features[:2]
            groups[f"{depth}_align_momentum"] = token_features[1:]
            groups[f"{depth}_sem_momentum"] = (token_features[0], token_features[2])
        native = tuple(
            name for name in (
                f"{depth}_corr_mean",
                f"{depth}_pose",
                f"{depth}_image_mean",
                f"{depth}_human_mean",
            ) if name in available
        )
        if native:
            groups[f"{depth}_native_concat"] = native
    groups["corr_l05_l08"] = (
        "decoder_l05_corr_mean",
        "decoder_l08_corr_mean",
    )
    groups["corr_l08_l11"] = (
        "decoder_l08_corr_mean",
        "decoder_l11_corr_mean",
    )
    groups["corr_all_depths"] = tuple(
        f"decoder_l{layer:02d}_corr_mean" for layer in DECODER_LAYERS
    )
    return groups


def stack_features(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    values = []
    for row in rows:
        raw_pose = torch.from_numpy(target_vector(row["raw_relative"].numpy()))
        values.append(torch.cat([raw_pose] + [row["descriptors"][name] for name in names]).numpy())
    return np.stack(values).astype(np.float64)


def stack_targets(rows: list[dict]) -> np.ndarray:
    return np.stack([row["target"].numpy() for row in rows]).astype(np.float64)


def summarize_residual(rows: list[dict], prediction: np.ndarray) -> dict:
    final_predictions = []
    for row, residual in zip(rows, prediction):
        final = vector_to_pose(residual) @ row["raw_relative"].numpy()
        final_predictions.append(target_vector(final))
    return summarize_predictions(rows, np.stack(final_predictions), "absolute")


def summarize_baseline(rows: list[dict], key: str) -> dict:
    return summarize_predictions(
        rows,
        np.stack([target_vector(row[key].numpy()) for row in rows]),
        "absolute",
    )


def evaluate(args: argparse.Namespace, rows: list[dict]) -> dict:
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval10"]
    single_rows = [row for row in eval_rows if "lbn1_1192" in row["pattern_id"]]
    groups = feature_groups(rows)
    train_y = stack_targets(train_rows)
    eval_y = stack_targets(eval_rows)
    report: dict[str, Any] = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "train_cases": len(train_rows),
            "eval_cases": len(eval_rows),
            "single_case": single_rows[0]["pattern_id"] if single_rows else None,
            "decoder_layers": DECODER_LAYERS,
        },
        "v9_raw_head_input": summarize_baseline(eval_rows, "raw_relative"),
        "formal_v9_full": summarize_baseline(eval_rows, "full_relative"),
        "groups": {},
    }
    for group_name, names in groups.items():
        train_x = stack_features(train_rows, names)
        eval_x = stack_features(eval_rows, names)
        ridge_eval = ridge_fit_predict(train_x, train_y, eval_x, args.ridge)
        mlp_train, mlp_eval = mlp_fit_predict(
            train_x,
            train_y,
            eval_x,
            args.mlp_steps,
            args.seed + int(hashlib.sha1(group_name.encode()).hexdigest()[:6], 16),
            args.device,
        )
        _, ten_overfit = mlp_fit_predict(
            eval_x,
            eval_y,
            eval_x,
            args.mlp_steps,
            args.seed + 1000 + int(hashlib.sha1(group_name.encode()).hexdigest()[:6], 16),
            args.device,
        )
        result = {
            "features": ["raw_relative_pose", *names],
            "dimension": int(train_x.shape[1]),
            "ridge_heldout": summarize_residual(eval_rows, ridge_eval),
            "mlp_train_fit": summarize_residual(train_rows, mlp_train),
            "mlp_heldout": summarize_residual(eval_rows, mlp_eval),
            "ten_case_overfit": summarize_residual(eval_rows, ten_overfit),
        }
        if single_rows:
            single_x = stack_features(single_rows, names)
            single_y = stack_targets(single_rows)
            _, single_overfit = mlp_fit_predict(
                single_x, single_y, single_x, max(500, args.mlp_steps), args.seed + 2000, args.device
            )
            result["single_case_overfit"] = summarize_residual(single_rows, single_overfit)
        report["groups"][group_name] = result
        print(
            f"fit {group_name:34s} ridge={result['ridge_heldout']['composite']['mean']:.4f} "
            f"mlp={result['mlp_heldout']['composite']['mean']:.4f}",
            flush=True,
        )
    report["ranking"] = sorted(
        (
            {
                "group": name,
                "ridge": result["ridge_heldout"]["composite"]["mean"],
                "mlp": result["mlp_heldout"]["composite"]["mean"],
                "ten_overfit": result["ten_case_overfit"]["composite"]["mean"],
            }
            for name, result in report["groups"].items()
        ),
        key=lambda item: item["ridge"],
    )
    return report


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# V9 Decoder / Correct-Token Frozen Probe",
        "",
        f"- Train cases: {report['protocol']['train_cases']}",
        f"- Frozen cuts: {report['protocol']['eval_cases']}",
        f"- V9 raw-head-input composite: {report['v9_raw_head_input']['composite']['mean']:.4f}",
        f"- Formal V9 full composite: {report['formal_v9_full']['composite']['mean']:.4f}",
        "",
        "| Feature group | Ridge held-out | MLP held-out | 10-cut overfit |",
        "|---|---:|---:|---:|",
    ]
    for row in report["ranking"]:
        lines.append(
            f"| {row['group']} | {row['ridge']:.4f} | {row['mlp']:.4f} | "
            f"{row['ten_overfit']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "token_cache.pt"
    train_records, eval_records = collect_records(args.train_per_source, args.seed)
    records = train_records + eval_records
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "train_records": train_records,
                "eval_records": eval_records,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.fit_only and (args.overwrite_cache or not cache_path.is_file()):
        from dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(str(args.checkpoint)).to(args.device).float()
        configure_formal_v9(model)
        capture = TokenCapture.attach(model)
        try:
            rows = extract(args, records, model, capture)
        finally:
            capture.close()
        torch.save(rows, cache_path)
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        rows = torch.load(cache_path, map_location="cpu", weights_only=False)

    if args.extract_only:
        return
    report = evaluate(args, rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output_dir / "report.md")
    print(f"wrote {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
