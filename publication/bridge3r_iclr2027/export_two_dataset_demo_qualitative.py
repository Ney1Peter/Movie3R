#!/usr/bin/env python3
"""Export honest demo-style qualitative previews for two fixed paper cases.

The exporter deliberately separates *prediction geometry* from *display
geometry*.  Prediction payloads retain each method's native world coordinates
and are compatible with the ``demo.py --save`` directory layout.  The MP4
previews use a method-local, fixed virtual camera and an optional vertical-axis
flip only to make every body upright on screen.  No GT, evaluator alignment,
or geometry from another method is used in either representation.

External methods do not predict a scene point cloud.  Their depth/confidence
payload files are therefore explicit zero-valued placeholders and the manifest
marks the scene channel unavailable; Bridge3R backgrounds are never copied to
external baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


MOVIE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = MOVIE_ROOT.parent
for item in (MOVIE_ROOT, MOVIE_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from publication.bridge3r_iclr2027.runtime_contract import apply_locked_transaction  # noqa: E402


ARRAY_KEYS = (
    "cameras_c2w",
    "vertices_world",
    "joints_world",
    "persistent_ids",
    "native_ids",
    "valid",
)
METHOD_ORDER = ("strict", "bridge3r", "trace", "prompthmr_spec", "prompthmr_nospec")
METHOD_LABELS = {
    "strict": "Strict Human3R",
    "bridge3r": "Bridge3R",
    "trace": "TRACE (official)",
    "prompthmr_spec": "PromptHMR (official SPEC)",
    "prompthmr_nospec": "PromptHMR (no-SPEC adapter)",
}
ID_COLOURS_RGB = (
    (65, 105, 225),
    (238, 99, 82),
    (46, 160, 108),
    (155, 89, 182),
    (238, 174, 49),
    (65, 182, 196),
    (220, 80, 135),
    (120, 120, 120),
)


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    case_id: str
    angle_label: str
    angle_deg: float
    people: int
    cache: Path
    runtime: Path
    trace: Path
    prompthmr_spec: Path
    prompthmr_nospec: Path
    rgb_dir: Path | None
    rgb_outer_zip: Path | None
    metrics: Mapping[str, tuple[float, float, float]]


def absolute(relative: str) -> Path:
    return WORKSPACE_ROOT / relative


CASES = {
    "harmony4d": CaseSpec(
        dataset="Harmony4D",
        case_id="h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076",
        angle_label="extreme",
        angle_deg=150.7155280214479,
        people=2,
        cache=absolute(
            "Movie3R/output/v15_harmony4d/predictions/test_03_grappling2/"
            "h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076.npz"
        ),
        runtime=absolute(
            "Movie3R/output/v15_harmony4d/predictions/test_03_grappling2/"
            "h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076.runtime.json"
        ),
        trace=absolute(
            "data/Harmony4D_work_v17_full_test/external_predictions/"
            "trace_harmony4d_v2/test/converted/line005.npz"
        ),
        prompthmr_spec=absolute(
            "data/Harmony4D_work_v17_full_test/external_predictions/"
            "prompthmr_harmony4d/test/spec/harmony4d_test_spec/line005/prediction.npz"
        ),
        prompthmr_nospec=absolute(
            "data/Harmony4D_work_v17_full_test/external_predictions/"
            "prompthmr_harmony4d/test/nospec/harmony4d_test_nospec/line005/prediction.npz"
        ),
        rgb_dir=absolute(
            "data/Harmony4D_work_v17_full_test/external_predictions/"
            "trace_harmony4d_v2/test/runtime_inputs/"
            "h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076/images"
        ),
        rgb_outer_zip=None,
        metrics={
            "strict": (762.06, 447.70, 0.5688),
            "bridge3r": (546.06, 206.95, 0.7523),
            "trace": (float("nan"), 758.18, 0.0362),
            "prompthmr_spec": (372.21, 365.15, 0.0707),
            "prompthmr_nospec": (388.25, 330.86, 0.0560),
        },
    ),
    "egohumans": CaseSpec(
        dataset="EgoHumans",
        case_id="ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301",
        angle_label="extreme",
        angle_deg=176.74916252624004,
        people=3,
        cache=absolute(
            "Movie3R/output/v19_egohumans/test/predictions/"
            "legoassemble__003_legoassemble-002/"
            "ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301.npz"
        ),
        runtime=absolute(
            "Movie3R/output/v19_egohumans/test/predictions/"
            "legoassemble__003_legoassemble-002/"
            "ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301.runtime.json"
        ),
        trace=absolute(
            "data/EgoHuman_work_v19/external_predictions/"
            "trace_egohumans_v2/test/converted/line037.npz"
        ),
        prompthmr_spec=absolute(
            "data/EgoHuman_work_v19/external_predictions/"
            "prompthmr_egohumans/spec/egohumans_test_spec/line037/prediction.npz"
        ),
        prompthmr_nospec=absolute(
            "data/EgoHuman_work_v19/external_predictions/"
            "prompthmr_egohumans/nospec/egohumans_test_nospec/line037/prediction.npz"
        ),
        rgb_dir=None,
        rgb_outer_zip=absolute("data/EgoHuman.zip"),
        metrics={
            "strict": (393.1, 208.1, 0.514),
            "bridge3r": (337.0, 182.1, 0.951),
            "trace": (2359.0, 895.5, 0.143),
            "prompthmr_spec": (1056.1, 971.8, 0.422),
            "prompthmr_nospec": (1066.2, 992.5, 0.412),
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(CASES),
        default=list(CASES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MOVIE_ROOT / "output/bridge3r_two_dataset_demo_v2",
    )
    parser.add_argument("--pre", type=int, default=5)
    parser.add_argument("--post", type=int, default=25)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=450)
    parser.add_argument("--smoke", action="store_true", help="Render only cut-1 and cut frames.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    """Convert non-finite metric placeholders to strict JSON null values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_from_npz(path: Path, prefix: str | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if prefix is None:
            candidates = {name.split("__", 1)[0] for name in source.files if "__" in name}
            if len(candidates) != 1:
                raise ValueError(f"Expected one method prefix in {path}, found {sorted(candidates)}")
            prefix = next(iter(candidates))
        missing = [key for key in ARRAY_KEYS if f"{prefix}__{key}" not in source]
        if missing:
            raise KeyError(f"{path}: prefix {prefix} misses {missing}")
        return {key: np.asarray(source[f"{prefix}__{key}"]).copy() for key in ARRAY_KEYS}


def bridge3r_arrays(spec: CaseSpec, runtime: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = arrays_from_npz(spec.cache, "m3_b0_only")
    boundary = int(runtime["record"]["boundary_index"])
    pairs = [tuple(map(int, row)) for row in runtime["geometry"]["association"]["pairs"]]
    return apply_locked_transaction(source, boundary=boundary, pairs=pairs, cut_detected=True)


def load_methods(spec: CaseSpec, runtime: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    bridge, bridge_debug = bridge3r_arrays(spec, runtime)
    methods = {
        "strict": arrays_from_npz(spec.cache, "m0_strict_human3r"),
        "bridge3r": bridge,
        "trace": arrays_from_npz(spec.trace),
        "prompthmr_spec": arrays_from_npz(spec.prompthmr_spec),
        "prompthmr_nospec": arrays_from_npz(spec.prompthmr_nospec),
    }
    lengths = {name: len(value["valid"]) for name, value in methods.items()}
    expected = int(runtime["record"]["clip_length"])
    if set(lengths.values()) != {expected}:
        raise ValueError(f"Frame-count mismatch for {spec.case_id}: {lengths}, expected={expected}")
    return methods, bridge_debug


def selected_indices(runtime: dict[str, Any], pre: int, post: int, smoke: bool) -> list[int]:
    boundary = int(runtime["record"]["boundary_index"])
    if pre <= 0 or post <= 0 or pre > boundary or boundary + post > int(runtime["record"]["clip_length"]):
        raise ValueError((boundary, pre, post, runtime["record"]["clip_length"]))
    indices = list(range(boundary - pre, boundary + post))
    return [boundary - 1, boundary] if smoke else indices


def prepare_harmony_rgb(spec: CaseSpec, indices: list[int]) -> list[Path]:
    assert spec.rgb_dir is not None
    paths = sorted(spec.rgb_dir.glob("*.jpg"))
    if len(paths) != 150:
        raise ValueError(f"Expected 150 Harmony4D RGB frames, found {len(paths)} in {spec.rgb_dir}")
    return [paths[index] for index in indices]


def prepare_egohumans_rgb(
    spec: CaseSpec,
    runtime: dict[str, Any],
    indices: list[int],
    destination: Path,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [destination / f"{local:06d}.jpg" for local in range(len(indices))]
    if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return outputs
    record = runtime["record"]
    image_members = record.get("image_members")
    if not image_members:
        # The internal runtime intentionally omits the long member list.  The
        # TRACE runtime manifest is prediction-only and records the same fixed
        # case and RGB order used by every external baseline.
        manifest = absolute(
            "data/EgoHuman_work_v19/external_predictions/trace_egohumans_v2/"
            "manifests/egohumans_test.runtime.jsonl"
        )
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        matches = [row for row in rows if row["case_id"] == spec.case_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one TRACE manifest row for {spec.case_id}, found {len(matches)}")
        image_members = matches[0]["image_members"]
    wanted = {str(image_members[index]): outputs[local] for local, index in enumerate(indices)}
    assert spec.rgb_outer_zip is not None
    archive_entry = str(record["archive_entry"])
    found: set[str] = set()
    with zipfile.ZipFile(spec.rgb_outer_zip) as outer:
        with outer.open(archive_entry) as nested:
            with tarfile.open(fileobj=nested, mode="r|gz") as archive:
                for member in archive:
                    # The original EgoHumans tar stores an eight-component
                    # machine-specific prefix.  The audited protocol stages it
                    # with ``tar --strip-components=8``; reproduce exactly that
                    # logical member name without extracting the full archive.
                    parts = tuple(part for part in member.name.lstrip("./").split("/") if part)
                    name = "/".join(parts[8:])
                    if name not in wanted:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise OSError(f"Could not read {name} from {archive_entry}")
                    payload = source.read()
                    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise OSError(f"Could not decode {name}")
                    if not cv2.imwrite(str(wanted[name]), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError(f"Could not write {wanted[name]}")
                    found.add(name)
                    if len(found) == len(wanted):
                        break
    missing = sorted(set(wanted) - found)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected EgoHumans RGB members: {missing[:3]}")
    return outputs


def resize_letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def add_header(image: np.ndarray, title: str, subtitle: str, cut: bool) -> np.ndarray:
    header = np.full((72, image.shape[1], 3), (247, 247, 247), dtype=np.uint8)
    colour = (42, 78, 190) if not cut else (45, 72, 210)
    cv2.putText(header, title, (16, 29), cv2.FONT_HERSHEY_DUPLEX, 0.68, (28, 28, 28), 1, cv2.LINE_AA)
    cv2.putText(header, subtitle, (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    if cut:
        cv2.rectangle(header, (header.shape[1] - 112, 10), (header.shape[1] - 14, 60), colour, -1)
        cv2.putText(header, "CUT", (header.shape[1] - 91, 43), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def load_faces() -> np.ndarray:
    path = MOVIE_ROOT / "src/models/smpl/SMPL_NEUTRAL.pkl"
    with path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    return np.asarray(model["f"], dtype=np.int32)


def pelvis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(joints)[..., [1, 2], :].mean(axis=-2)


def display_flip(arrays: Mapping[str, np.ndarray], indices: list[int]) -> np.ndarray:
    joints = arrays["joints_world"][indices]
    valid = arrays["valid"][indices].astype(bool)
    head_offset = joints[..., 15, 1] - pelvis(joints)[..., 1]
    finite = head_offset[valid & np.isfinite(head_offset)]
    flip = -1.0 if len(finite) and float(np.median(finite)) < 0.0 else 1.0
    return np.diag([1.0, flip, 1.0]).astype(np.float32)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    up = np.asarray([0.0, 1.0, 0.0])
    backward = eye - target
    backward /= max(float(np.linalg.norm(backward)), 1e-9)
    right = np.cross(up, backward)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    camera_up = np.cross(backward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def view_plan(arrays: Mapping[str, np.ndarray], indices: list[int], aspect: float) -> dict[str, Any]:
    valid = arrays["valid"][indices].astype(bool)
    vertices = arrays["vertices_world"][indices][valid]
    roots = pelvis(arrays["joints_world"][indices])[valid]
    transform = display_flip(arrays, indices)
    vertices = vertices @ transform.T
    roots = roots @ transform.T
    if not len(vertices):
        raise ValueError("Cannot render a method with no valid mesh in the selected window")
    sampled = vertices.reshape(-1, 3)[::80]
    target = np.nanmedian(roots, axis=0)
    robust_lo, robust_hi = np.nanpercentile(sampled, [0.25, 99.75], axis=0)
    span = max(float(np.max(robust_hi - robust_lo)), 1.5)
    eye = target + span * np.asarray([1.35, 0.88, 1.55])
    camera_pose = look_at(eye, target)
    world_to_camera = np.linalg.inv(camera_pose)
    homogeneous = np.concatenate([sampled, np.ones((len(sampled), 1))], axis=1)
    camera_points = homogeneous @ world_to_camera.T
    x_lo, x_hi = np.nanpercentile(camera_points[:, 0], [0.25, 99.75])
    y_lo, y_hi = np.nanpercentile(camera_points[:, 1], [0.25, 99.75])
    xmag = max(float(x_hi - x_lo) * 0.60, 0.8)
    ymag = max(float(y_hi - y_lo) * 0.62, 0.8)
    if xmag / ymag < aspect:
        xmag = ymag * aspect
    else:
        ymag = xmag / aspect
    ground_y = float(np.nanpercentile(vertices[..., 1], 0.5))
    return {
        "transform": transform,
        "camera_pose": camera_pose,
        "xmag": xmag,
        "ymag": ymag,
        "ground_y": ground_y,
        "target": target,
        "span": span,
    }


def colour_for_id(identity: int) -> tuple[float, float, float, float]:
    rgb = ID_COLOURS_RGB[int(identity) % len(ID_COLOURS_RGB)]
    return tuple(value / 255.0 for value in rgb) + (1.0,)


def add_raymond_lights(scene: pyrender.Scene, camera_pose: np.ndarray) -> None:
    for yaw in (-0.8, 0.0, 0.8):
        pose = camera_pose.copy()
        rotation = np.asarray(
            [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
        )
        pose[:3, :3] = pose[:3, :3] @ rotation
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.1), pose=pose)


def render_world_frame(
    renderer: pyrender.OffscreenRenderer,
    arrays: Mapping[str, np.ndarray],
    global_index: int,
    history_indices: list[int],
    faces: np.ndarray,
    plan: Mapping[str, Any],
) -> np.ndarray:
    scene = pyrender.Scene(bg_color=np.asarray([0.965, 0.97, 0.98, 1.0]), ambient_light=np.ones(3) * 0.45)
    transform = np.asarray(plan["transform"])
    valid = arrays["valid"][global_index].astype(bool)
    identities = arrays["persistent_ids"][global_index]
    for slot in np.flatnonzero(valid):
        identity = int(identities[slot])
        vertices = np.asarray(arrays["vertices_world"][global_index, slot], dtype=np.float64) @ transform.T
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=colour_for_id(identity),
            metallicFactor=0.0,
            roughnessFactor=0.62,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))

    all_roots = pelvis(arrays["joints_world"])
    sphere_radius = max(0.014 * float(plan["span"]), 0.018)
    for history_index in history_indices:
        history_valid = arrays["valid"][history_index].astype(bool)
        history_ids = arrays["persistent_ids"][history_index]
        for slot in np.flatnonzero(history_valid):
            root = np.asarray(all_roots[history_index, slot], dtype=np.float64) @ transform.T
            identity = int(history_ids[slot])
            marker = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
            marker.apply_translation(root)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=colour_for_id(identity), metallicFactor=0.0, roughnessFactor=0.8
            )
            scene.add(pyrender.Mesh.from_trimesh(marker, material=material, smooth=True))

    floor_extent = max(float(plan["xmag"]), float(plan["ymag"])) * 2.2
    floor = trimesh.creation.box(extents=(floor_extent, max(0.006 * floor_extent, 0.008), floor_extent))
    floor.apply_translation((float(plan["target"][0]), float(plan["ground_y"]) - 0.015, float(plan["target"][2])))
    floor_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.83, 0.85, 0.88, 1.0), metallicFactor=0.0, roughnessFactor=1.0
    )
    scene.add(pyrender.Mesh.from_trimesh(floor, material=floor_material, smooth=False))
    camera = pyrender.OrthographicCamera(xmag=float(plan["xmag"]), ymag=float(plan["ymag"]), znear=0.01, zfar=1000.0)
    scene.add(camera, pose=np.asarray(plan["camera_pose"]))
    add_raymond_lights(scene, np.asarray(plan["camera_pose"]))
    colour, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return cv2.cvtColor(colour[:, :, :3], cv2.COLOR_RGB2BGR)


def make_shared_rgb(
    rgb_paths: list[Path],
    destination: Path,
    width: int,
    height: int,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, source in enumerate(rgb_paths):
        output = destination / f"{index:06d}.png"
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read RGB {source}")
        image = resize_letterbox(image, width, height)
        if not cv2.imwrite(str(output), image):
            raise OSError(f"Could not write {output}")
        outputs.append(output)
    return outputs


def ensure_link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def write_demo_payload(
    root: Path,
    arrays: Mapping[str, np.ndarray],
    indices: list[int],
    shared_rgb: list[Path],
    shared_depth: Path,
    shared_conf: Path,
    faces: np.ndarray,
    method: str,
) -> None:
    for name in ("color", "depth", "conf", "camera", "smpl"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for local, global_index in enumerate(indices):
        ensure_link(shared_rgb[local], root / "color" / f"{local:06d}.png")
        ensure_link(shared_depth, root / "depth" / f"{local:06d}.npy")
        ensure_link(shared_conf, root / "conf" / f"{local:06d}.npy")
        camera = np.asarray(arrays["cameras_c2w"][global_index], dtype=np.float32)
        np.savez(
            root / "camera" / f"{local:06d}.npz",
            pose=camera,
            intrinsics=np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 225.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )
        valid = arrays["valid"][global_index].astype(bool)
        vertices = np.asarray(arrays["vertices_world"][global_index, valid], dtype=np.float32)
        identities = np.asarray(arrays["persistent_ids"][global_index, valid], dtype=np.int64)
        count = len(vertices)
        np.savez(
            root / "smpl" / f"{local:06d}.npz",
            scores=np.zeros((1, 1), dtype=np.float32),
            msk=np.zeros((1, 1), dtype=np.float32),
            shape=np.zeros((count, 10), dtype=np.float32),
            rotvec=np.zeros((count, 53, 3), dtype=np.float32),
            transl=np.zeros((count, 3), dtype=np.float32),
            expression=np.zeros((count, 10), dtype=np.float32),
            smpl_id=identities,
            verts_world=vertices,
            faces=faces,
        )
    metadata = {
        "schema_version": "Bridge3R-demo-payload-mesh-only-v1",
        "method": method,
        "method_label": METHOD_LABELS[method],
        "format": "demo.py --save compatible mesh-only payload",
        "scene_channel_available": False,
        "depth_and_confidence": "explicit zero placeholders; no scene is claimed",
        "intrinsics": "visualization-only placeholder; not used by the exported MP4 renderer",
        "geometry_coordinates": "unaltered prediction world coordinates",
        "trace_camera_note": (
            "TRACE cameras_c2w is a diagnostic body-root proxy, not an official physical camera trajectory"
            if method == "trace" else None
        ),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def encode_video(frame_dir: Path, destination: Path, fps: int) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)


def render_dataset(
    spec: CaseSpec,
    output: Path,
    pre: int,
    post: int,
    fps: int,
    width: int,
    height: int,
    smoke: bool,
) -> dict[str, Any]:
    runtime = read_json(spec.runtime)
    if runtime["record"]["case_id"] != spec.case_id:
        raise ValueError(f"Runtime case mismatch: {runtime['record']['case_id']} vs {spec.case_id}")
    indices = selected_indices(runtime, pre, post, smoke)
    boundary = int(runtime["record"]["boundary_index"])
    methods, bridge_debug = load_methods(spec, runtime)
    dataset_root = output / spec.dataset.lower()
    source_rgb_root = dataset_root / "_selected_source_rgb"
    if spec.dataset == "Harmony4D":
        rgb_paths = prepare_harmony_rgb(spec, indices)
    else:
        rgb_paths = prepare_egohumans_rgb(spec, runtime, indices, source_rgb_root)
    shared_root = dataset_root / "_shared"
    shared_rgb = make_shared_rgb(rgb_paths, shared_root / "color", width, height)
    np.save(shared_root / "empty_depth.npy", np.zeros((height, width), dtype=np.float32))
    np.save(shared_root / "empty_conf.npy", np.zeros((height, width), dtype=np.float32))
    faces = load_faces()
    payload_roots = {}
    for method in METHOD_ORDER:
        payload_root = dataset_root / "payloads" / method
        write_demo_payload(
            payload_root, methods[method], indices, shared_rgb,
            shared_root / "empty_depth.npy", shared_root / "empty_conf.npy", faces, method,
        )
        payload_roots[method] = str(payload_root.resolve())

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    plans = {method: view_plan(methods[method], indices, width / height) for method in METHOD_ORDER}
    combined_dir = dataset_root / "combined_frames"
    combined_dir.mkdir(parents=True, exist_ok=True)
    method_dirs = {}
    for method in METHOD_ORDER:
        frame_dir = dataset_root / method / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        method_dirs[method] = frame_dir

    for local, global_index in enumerate(indices):
        source = cv2.imread(str(shared_rgb[local]), cv2.IMREAD_COLOR)
        is_post = global_index >= boundary
        source_panel = add_header(
            source,
            f"Input RGB | {spec.dataset}",
            f"frame {global_index:03d} | {'post-cut' if is_post else 'pre-cut'} | view span {spec.angle_deg:.1f} deg",
            is_post and global_index == boundary,
        )
        panels = [source_panel]
        history = [index for index in indices[:local + 1] if index <= global_index]
        for method in METHOD_ORDER:
            render = render_world_frame(renderer, methods[method], global_index, history, faces, plans[method])
            w, wa, idf1 = spec.metrics[method]
            panel = add_header(
                render,
                METHOD_LABELS[method],
                f"W {w:.1f} mm | WA {wa:.1f} mm | IDF1 {idf1:.3f}",
                is_post and global_index == boundary,
            )
            panels.append(panel)
            individual = np.concatenate([source_panel, panel], axis=1)
            cv2.imwrite(str(method_dirs[method] / f"{local:06d}.png"), individual)
        rows = [np.concatenate(panels[0:3], axis=1), np.concatenate(panels[3:6], axis=1)]
        combined = np.concatenate(rows, axis=0)
        cv2.imwrite(str(combined_dir / f"{local:06d}.png"), combined)
    renderer.delete()

    videos = {}
    if not smoke:
        combined_video = dataset_root / f"{spec.dataset.lower()}_five_method_comparison.mp4"
        encode_video(combined_dir, combined_video, fps)
        videos["combined"] = str(combined_video.resolve())
        for method in METHOD_ORDER:
            destination = dataset_root / method / f"{method}.mp4"
            encode_video(method_dirs[method], destination, fps)
            videos[method] = str(destination.resolve())
    keyframes = {
        "pre_cut": str((combined_dir / f"{indices.index(boundary - 1):06d}.png").resolve()),
        "post_cut": str((combined_dir / f"{indices.index(boundary):06d}.png").resolve()),
    }
    view_audit = {
        method: {
            "vertical_axis_flipped": bool(float(plans[method]["transform"][1, 1]) < 0.0),
            "fixed_virtual_camera": np.asarray(plans[method]["camera_pose"]).tolist(),
            "orthographic_xmag": float(plans[method]["xmag"]),
            "orthographic_ymag": float(plans[method]["ymag"]),
        }
        for method in METHOD_ORDER
    }
    return {
        "dataset": spec.dataset,
        "case_id": spec.case_id,
        "selection": {
            "requested_role": "one strong Bridge3R qualitative case",
            "angle_stratum": spec.angle_label,
            "camera_rotation_span_deg": spec.angle_deg,
            "people": spec.people,
            "selection_uses_reported_case_metrics": True,
            "representative_or_random_claimed": False,
        },
        "frame_window": {
            "indices": indices,
            "pre_count": sum(index < boundary for index in indices),
            "post_count": sum(index >= boundary for index in indices),
            "cut_index_in_original_clip": boundary,
            "cut_index_in_export": indices.index(boundary),
        },
        "inputs": {
            "runtime": str(spec.runtime.resolve()),
            "internal_cache": str(spec.cache.resolve()),
            "trace": str(spec.trace.resolve()),
            "prompthmr_spec": str(spec.prompthmr_spec.resolve()),
            "prompthmr_nospec": str(spec.prompthmr_nospec.resolve()),
            "rgb_sha256": [sha256(path) for path in rgb_paths],
        },
        "methods": {
            method: {
                "label": METHOD_LABELS[method],
                "metrics": {
                    "w_mpjpe_mm": spec.metrics[method][0],
                    "wa_mpjpe_mm": spec.metrics[method][1],
                    "idf1": spec.metrics[method][2],
                },
                "payload": payload_roots[method],
            }
            for method in METHOD_ORDER
        },
        "bridge3r_materialization_debug": bridge_debug,
        "display_only_transform_audit": view_audit,
        "videos": videos,
        "keyframes": keyframes,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    allowed_root = (MOVIE_ROOT / "output").resolve()
    if output != allowed_root and allowed_root not in output.parents:
        raise ValueError(f"Output must stay below {allowed_root}: {output}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    retained_reports: list[dict[str, Any]] = []
    if manifest_path.is_file() and not args.overwrite:
        previous = read_json(manifest_path)
        replacing = {CASES[name].dataset for name in args.datasets}
        retained_reports = [
            row for row in previous.get("cases", []) if row.get("dataset") not in replacing
        ]
    reports = []
    for dataset in args.datasets:
        reports.append(
            render_dataset(
                CASES[dataset], output, int(args.pre), int(args.post), int(args.fps),
                int(args.render_width), int(args.render_height), bool(args.smoke),
            )
        )
    manifest = {
        "schema_version": "Bridge3R-two-dataset-demo-qualitative-v2",
        "format": "mesh-only demo.py payloads plus fixed-view MP4/PNG previews",
        "methods": list(METHOD_ORDER),
        "contract": {
            "same_rgb_and_frame_indices_for_all_methods": True,
            "gt_used_for_rendering_or_alignment": False,
            "method_geometry_modified": False,
            "display_only_vertical_flip_may_be_applied": True,
            "display_camera_fixed_over_each_method_window": True,
            "external_scene_pointcloud_claimed": False,
            "trace_physical_camera_trajectory_claimed": False,
            "prompthmr_nospec_is_official_configuration": False,
        },
        "cases": retained_reports + reports,
    }
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "manifest": str(manifest_path),
        "cases": len(manifest["cases"]),
    }, indent=2))


if __name__ == "__main__":
    main()
