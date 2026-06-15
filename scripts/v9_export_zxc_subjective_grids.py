#!/usr/bin/env python3
"""Export two zxc held-out subjective comparison clips as 2x2 grids.

The script prepares demo-style four-frame folders, runs original Human3R plus
the two completed V9 checkpoints, and writes input/render grids for quick
subjective inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data/Training")


CLIPS = [
    {
        "id": "zxc_22070932_to_22053912_start1663",
        "group": "zxc",
        "seqA": "zxc/22070932",
        "seqB": "zxc/22053912",
        "start_frame": 1663,
        "view_angle_deg": 174.449979,
        "test_manifest_index": 716,
    },
    {
        "id": "zxc_22053925_to_22053917_start1545",
        "group": "zxc",
        "seqA": "zxc/22053925",
        "seqB": "zxc/22053917",
        "start_frame": 1545,
        "view_angle_deg": 173.805015,
        "test_manifest_index": 3,
    },
]


MODELS = [
    {
        "id": "original_human3r",
        "label": "Original Human3R",
        "path": REPO_ROOT / "src/human3r_896L.pth",
    },
    {
        "id": "pose_lora",
        "label": "V9 Pose LoRA",
        "path": REPO_ROOT / "output/v9_mixed_60h/v9_mixed_60h_pose_lora_bs10/checkpoint-final.pth",
    },
    {
        "id": "pose_human_lora",
        "label": "V9 Pose+Human LoRA",
        "path": REPO_ROOT / "output/v9_mixed_60h/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output/v9_subjective_zxc_two_clips",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument(
        "--mesh_render",
        action="store_true",
        help="Use demo.py's original mesh renderer. By default a lightweight SMPL point overlay is used.",
    )
    return parser.parse_args()


def rgb_path(seq: str, frame: int) -> Path:
    return DATA_ROOT / seq / "rgb" / f"{frame:08d}.png"


def clip_image_paths(clip: dict) -> list[Path]:
    start = int(clip["start_frame"])
    return [
        rgb_path(clip["seqA"], start),
        rgb_path(clip["seqA"], start + 1),
        rgb_path(clip["seqB"], start + 2),
        rgb_path(clip["seqB"], start + 3),
    ]


def copy_clip_images(clip: dict, clip_dir: Path) -> list[Path]:
    seq_dir = clip_dir / "seq_input"
    seq_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, src in enumerate(clip_image_paths(clip)):
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = seq_dir / f"{idx:06d}.png"
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def fit_to_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(cell_w / img.width, cell_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), (32, 32, 32))
    canvas.paste(img, ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
    return canvas


def make_grid(image_paths: list[Path], out_path: Path, title: str | None = None) -> None:
    if len(image_paths) != 4:
        raise ValueError(f"Expected 4 images, got {len(image_paths)}")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    title_h = 34 if title else 0
    grid = Image.new("RGB", (cell_w * 2, cell_h * 2 + title_h), (24, 24, 24))
    if title:
        draw = ImageDraw.Draw(grid)
        draw.text((12, 9), title, fill=(245, 245, 245))
    for idx, img in enumerate(images):
        tile = fit_to_cell(img, cell_w, cell_h)
        x = (idx % 2) * cell_w
        y = title_h + (idx // 2) * cell_h
        grid.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def make_comparison_grid(model_grid_paths: list[Path], out_path: Path, title: str) -> None:
    images = [Image.open(path).convert("RGB") for path in model_grid_paths if path.is_file()]
    if not images:
        return
    width = max(img.width for img in images)
    heights = [img.height for img in images]
    title_h = 38
    out = Image.new("RGB", (width, sum(heights) + title_h), (24, 24, 24))
    draw = ImageDraw.Draw(out)
    draw.text((12, 10), title, fill=(245, 245, 245))
    y = title_h
    for img in images:
        tile = fit_to_cell(img, width, img.height)
        out.paste(tile, (0, y))
        y += img.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def run_model(model: dict, seq_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    if not model["path"].is_file():
        raise FileNotFoundError(model["path"])
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if args.skip_inference and (out_dir / "camera").is_dir():
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_human3r_save_output.py"),
        "--model_path",
        str(model["path"]),
        "--seq_path",
        str(seq_dir),
        "--output_dir",
        str(out_dir),
        "--device",
        args.device,
        "--size",
        str(args.size),
        "--max_frames",
        "4",
        "--subsample",
        "1",
        "--reset_interval",
        "100",
        "--overwrite",
    ]
    if args.mesh_render:
        cmd.append("--render")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "output/tmp/mpl"))
    python_paths = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True, timeout=900)


def model_frame_paths(out_dir: Path, subdir: str) -> list[Path]:
    return [out_dir / subdir / f"{idx:06d}.png" for idx in range(4)]


def draw_projected_vertices(image: np.ndarray, verts: np.ndarray, intrinsics: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    if verts.size == 0:
        return image
    z = verts[:, 2]
    valid = z > 1.0e-4
    verts = verts[valid]
    if verts.size == 0:
        return image
    proj = (verts / verts[:, 2:3]) @ intrinsics.T
    xy = np.rint(proj[:, :2]).astype(np.int32)
    h, w = image.shape[:2]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy = xy[inside]
    if xy.size == 0:
        return image
    # Draw a sparse but visible vertex overlay without hiding the input image.
    xy = xy[:: max(1, len(xy) // 2500)]
    out = image.copy()
    for x, y in xy:
        x0, x1 = max(0, x - 1), min(w, x + 2)
        y0, y1 = max(0, y - 1), min(h, y + 2)
        out[y0:y1, x0:x1] = color
    return out


def make_smpl_point_overlays(out_dir: Path) -> list[Path]:
    from src.dust3r.utils import SMPL_Layer
    import torch

    color_paths = model_frame_paths(out_dir, "color")
    smpl_paths = [out_dir / "smpl" / f"{idx:06d}.npz" for idx in range(4)]
    camera_paths = [out_dir / "camera" / f"{idx:06d}.npz" for idx in range(4)]
    if not all(path.is_file() for path in color_paths + smpl_paths + camera_paths):
        return []

    first = np.load(smpl_paths[0], allow_pickle=True)
    num_betas = int(first["shape"].shape[-1]) if first["shape"].size else 10
    smpl_layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=num_betas,
        kid=False,
        person_center="head",
    )

    overlay_dir = out_dir / "smpl_points"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    palette = [
        (255, 220, 0),
        (30, 220, 255),
        (255, 80, 160),
        (80, 255, 120),
    ]
    for idx, (color_path, smpl_path, camera_path) in enumerate(zip(color_paths, smpl_paths, camera_paths)):
        image = imageio.imread(color_path)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        image = image[..., :3].copy()
        smpl = np.load(smpl_path, allow_pickle=True)
        camera = np.load(camera_path)
        intrinsics = camera["intrinsics"].astype(np.float32)
        shape = smpl["shape"].astype(np.float32)
        rotvec = smpl["rotvec"].astype(np.float32)
        transl = smpl["transl"].astype(np.float32)
        expression = smpl["expression"]
        if expression is not None:
            expression = expression.astype(np.float32)
        if shape.size and rotvec.size and transl.size:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    torch.from_numpy(rotvec),
                    torch.from_numpy(shape),
                    torch.from_numpy(transl),
                    None,
                    None,
                    K=torch.from_numpy(intrinsics).expand(shape.shape[0], -1, -1),
                    expression=None if expression is None else torch.from_numpy(expression),
                )
            verts_all = smpl_out["smpl_v3d"].detach().cpu().numpy()
            for human_idx, verts in enumerate(verts_all):
                image = draw_projected_vertices(image, verts, intrinsics, palette[human_idx % len(palette)])
        out_path = overlay_dir / f"{idx:06d}.png"
        imageio.imwrite(out_path, image)
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "output/tmp/mpl").mkdir(parents=True, exist_ok=True)

    manifest = {"clips": CLIPS, "models": [{k: str(v) for k, v in m.items()} for m in MODELS]}
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for clip in CLIPS:
        clip_dir = args.output_dir / clip["id"]
        copied = copy_clip_images(clip, clip_dir)
        make_grid(
            copied,
            clip_dir / "input_2x2.png",
            f"{clip['id']} | angle {clip['view_angle_deg']:.1f} deg | input frames",
        )
        model_render_grids = []
        model_color_grids = []
        for model in MODELS:
            out_dir = clip_dir / model["id"]
            run_model(model, clip_dir / "seq_input", out_dir, args)
            color_paths = model_frame_paths(out_dir, "color")
            if all(path.is_file() for path in color_paths):
                out_path = out_dir / "color_2x2.png"
                make_grid(color_paths, out_path, f"{clip['id']} | {model['label']} | color")
                model_color_grids.append(out_path)
            smpl_point_paths = make_smpl_point_overlays(out_dir)
            if len(smpl_point_paths) == 4:
                out_path = out_dir / "smpl_points_2x2.png"
                make_grid(smpl_point_paths, out_path, f"{clip['id']} | {model['label']} | SMPL point overlay")
                model_render_grids.append(out_path)
            render_paths = model_frame_paths(out_dir, "color_smpl")
            if all(path.is_file() for path in render_paths):
                out_path = out_dir / "render_2x2.png"
                make_grid(render_paths, out_path, f"{clip['id']} | {model['label']} | render")
                model_render_grids.append(out_path)
        make_comparison_grid(
            model_color_grids,
            clip_dir / "comparison_color_3models.png",
            f"{clip['id']} | color grids: Original / Pose LoRA / Pose+Human LoRA",
        )
        make_comparison_grid(
            model_render_grids,
            clip_dir / "comparison_smpl_overlay_3models.png",
            f"{clip['id']} | SMPL overlays: Original / Pose LoRA / Pose+Human LoRA",
        )
        print(json.dumps({"clip": clip["id"], "output_dir": str(clip_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
