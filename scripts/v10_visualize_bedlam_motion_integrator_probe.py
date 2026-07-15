#!/usr/bin/env python3
"""Build a lightweight browser visualization for V10 BEDLAM integrator variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from v10_bedlam_motion_integrator_probe import (
    TransformHead,
    compute_metrics,
    load_bedlam_trajectory,
    make_episodes,
    make_items,
    stream_apply_variant,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "output" / "v10_bedlam_motion_integrator_probe" / "gt_synthetic_streaming_globalgauge"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "v10_bedlam_motion_integrator_probe" / "gt_synthetic_streaming_globalgauge_view"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def ns_from_saved_args(saved: dict) -> SimpleNamespace:
    defaults = {
        "trajectory_source": "bedlam_gt",
        "manifest": str(REPO_ROOT / "config" / "manifests" / "bedlam_seq000021_good_6fps" / "metadata.json"),
        "train_episodes": 512,
        "val_episodes": 128,
        "steps": 1500,
        "batch_size": 128,
        "hidden_dim": 256,
        "lr": 2e-3,
        "seed": 20260713,
        "segment_boundaries": [0, 10, 20],
        "max_rot_deg": 160.0,
        "max_trans": 4.0,
        "perturb_rot_deg": 120.0,
        "perturb_trans": 2.5,
        "global_rot_deg": 180.0,
        "global_trans": 5.0,
        "residual_max_rot_deg": 25.0,
        "residual_max_trans": 0.8,
        "log_every": 250,
        "device": "cpu",
    }
    defaults.update(saved)
    defaults["manifest"] = Path(defaults["manifest"])
    return SimpleNamespace(**defaults)


def load_model(run_dir: Path, name: str, in_dim: int, args: SimpleNamespace, device: torch.device) -> TransformHead:
    ckpt = torch.load(run_dir / f"{name}.pth", map_location="cpu")
    residual = bool(ckpt.get("residual", False))
    max_rot = args.residual_max_rot_deg if residual else args.max_rot_deg
    max_trans = args.residual_max_trans if residual else args.max_trans
    model = TransformHead(in_dim, args.hidden_dim, max_rot, max_trans).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def tensor_to_list(x: torch.Tensor) -> list:
    return np.asarray(x.detach().cpu(), dtype=np.float32).tolist()


def build_visual_data(run_dir: Path, episode_index: int, device: torch.device) -> dict:
    metrics_json = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    saved_args = ns_from_saved_args(metrics_json["args"])
    traj = load_bedlam_trajectory(saved_args.manifest)
    episodes = make_episodes(traj, max(episode_index + 1, 1), saved_args, seed_offset=100000)
    episode = episodes[episode_index]
    items = make_items([episode])
    models = {
        "current_only_mlp": load_model(run_dir, "current_only_mlp", items[0].feature_current.numel(), saved_args, device),
        "history_current_integrator": load_model(
            run_dir, "history_current_integrator", items[0].feature_history.numel(), saved_args, device
        ),
        "explicit_se3_residual_integrator": load_model(
            run_dir,
            "explicit_se3_residual_integrator",
            items[0].feature_residual.numel(),
            saved_args,
            device,
        ),
    }
    variants = [
        "raw_perturbed",
        "fixed_explicit_se3",
        "current_only_mlp",
        "history_current_integrator",
        "explicit_se3_residual_integrator",
        "oracle_se3_upper",
    ]
    data = {
        "frames": tensor_to_list(torch.arange(episode.target_root_t.shape[0])),
        "boundaries": episode.boundaries,
        "segment_ends": episode.segment_ends,
        "target": {
            "root_t": tensor_to_list(episode.target_root_t),
            "cam_t": tensor_to_list(episode.target_cam_t),
        },
        "variants": {},
    }
    for variant in variants:
        pred_root_t, _, pred_cam_t, _ = stream_apply_variant(episode, variant, models, saved_args, device)
        data["variants"][variant] = {
            "root_t": tensor_to_list(pred_root_t),
            "cam_t": tensor_to_list(pred_cam_t),
            "metrics": compute_metrics(episode, stream_apply_variant(episode, variant, models, saved_args, device)),
        }
    return data


def write_html(output_dir: Path, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectory_data.json").write_text(json.dumps(data), encoding="utf-8")
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>V10 BEDLAM Streaming Integrator Probe</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f8; color: #1f2933; }
    header { padding: 16px 20px 10px; background: #ffffff; border-bottom: 1px solid #d7dde3; }
    h1 { margin: 0 0 8px; font-size: 20px; font-weight: 700; }
    .note { color: #52616f; font-size: 13px; line-height: 1.45; max-width: 1100px; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    button { border: 1px solid #aeb8c2; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; font-size: 13px; }
    button.active { background: #1f6feb; color: white; border-color: #1f6feb; }
    main { display: grid; grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr); gap: 14px; padding: 14px; }
    section { background: white; border: 1px solid #d7dde3; border-radius: 8px; padding: 12px; }
    h2 { margin: 0 0 8px; font-size: 15px; }
    canvas { width: 100%; height: 520px; display: block; background: #fbfcfd; border: 1px solid #e3e7eb; border-radius: 6px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e5e8eb; text-align: right; padding: 7px 8px; }
    th:first-child, td:first-child { text-align: left; }
    .legend { font-size: 13px; color: #52616f; margin-top: 8px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } canvas { height: 420px; } }
  </style>
</head>
<body>
  <header>
    <h1>V10 BEDLAM Streaming Integrator Probe</h1>
    <div class="note">
      GT synthetic visualization. The selected variant is evaluated in a causal streaming loop:
      boundary frame uses only previous global state and current local frame, then caches one transform for the segment.
      Solid lines are prediction, dashed lines are GT.
    </div>
    <div id="buttons" class="controls"></div>
  </header>
  <main>
    <section>
      <h2>Top View: X / Z</h2>
      <canvas id="top" width="900" height="620"></canvas>
      <div class="legend">People are colored lines. Camera is red. Dashed = GT, solid = prediction.</div>
    </section>
    <section>
      <h2>Side View: X / Y</h2>
      <canvas id="side" width="900" height="620"></canvas>
      <div class="legend">Vertical mismatch and tilt-related translation errors are easier to inspect here.</div>
    </section>
    <section style="grid-column: 1 / -1;">
      <h2>Metrics</h2>
      <div id="metrics"></div>
    </section>
  </main>
  <script>
    const DATA = __DATA__;
    const variants = Object.keys(DATA.variants);
    const colors = ['#2f80ed', '#27ae60', '#9b51e0', '#f2994a', '#00a6a6', '#eb5757'];
    let current = variants[0];

    function allPoints(variant, dims) {
      const pts = [];
      const target = DATA.target.root_t;
      const pred = DATA.variants[variant].root_t;
      const targetCam = DATA.target.cam_t;
      const predCam = DATA.variants[variant].cam_t;
      for (const arr of [target, pred]) {
        for (const frame of arr) for (const p of frame) pts.push([p[dims[0]], p[dims[1]]]);
      }
      for (const arr of [targetCam, predCam]) for (const p of arr) pts.push([p[dims[0]], p[dims[1]]]);
      return pts;
    }

    function scaler(points, canvas) {
      const xs = points.map(p => p[0]);
      const ys = points.map(p => p[1]);
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      const padX = Math.max((maxX - minX) * 0.12, 0.5);
      const padY = Math.max((maxY - minY) * 0.12, 0.5);
      minX -= padX; maxX += padX; minY -= padY; maxY += padY;
      return p => {
        const x = 40 + (p[0] - minX) / (maxX - minX) * (canvas.width - 80);
        const y = canvas.height - 40 - (p[1] - minY) / (maxY - minY) * (canvas.height - 80);
        return [x, y];
      };
    }

    function drawLine(ctx, pts, map, color, dashed=false, width=2) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed ? [7, 7] : []);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const [x, y] = map(p);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    function drawDots(ctx, pts, map, color) {
      ctx.save();
      ctx.fillStyle = color;
      pts.forEach((p, i) => {
        const [x, y] = map(p);
        ctx.beginPath();
        ctx.arc(x, y, DATA.boundaries.includes(i) ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    function drawCanvas(canvasId, dims) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#fbfcfd';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#edf1f5';
      for (let i = 0; i < 8; i++) {
        const x = 40 + i * (canvas.width - 80) / 7;
        const y = 40 + i * (canvas.height - 80) / 7;
        ctx.beginPath(); ctx.moveTo(x, 40); ctx.lineTo(x, canvas.height - 40); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(canvas.width - 40, y); ctx.stroke();
      }
      const map = scaler(allPoints(current, dims), canvas);
      const gt = DATA.target.root_t;
      const pred = DATA.variants[current].root_t;
      const people = gt[0].length;
      for (let person = 0; person < people; person++) {
        const gtPts = gt.map(frame => [frame[person][dims[0]], frame[person][dims[1]]]);
        const predPts = pred.map(frame => [frame[person][dims[0]], frame[person][dims[1]]]);
        drawLine(ctx, gtPts, map, colors[person % colors.length], true, 2);
        drawLine(ctx, predPts, map, colors[person % colors.length], false, 2.5);
        drawDots(ctx, predPts, map, colors[person % colors.length]);
      }
      const gtCam = DATA.target.cam_t.map(p => [p[dims[0]], p[dims[1]]]);
      const predCam = DATA.variants[current].cam_t.map(p => [p[dims[0]], p[dims[1]]]);
      drawLine(ctx, gtCam, map, '#111827', true, 2);
      drawLine(ctx, predCam, map, '#d62728', false, 3);
      drawDots(ctx, predCam, map, '#d62728');
    }

    function renderMetrics() {
      const m = DATA.variants[current].metrics;
      const names = {
        root_trans_m: 'Root Trans',
        root_rot_deg: 'Root Rot',
        cam_trans_m: 'Cam Trans',
        cam_rot_deg: 'Cam Rot',
        boundary_jump_m: 'Boundary Jump',
        velocity_m: 'Velocity',
        accel_m: 'Acceleration',
        non_boundary_motion_m: 'Non-boundary',
        inter_person_dist_m: 'Inter-person'
      };
      let html = '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
      for (const [k, label] of Object.entries(names)) {
        const val = m[k];
        html += `<tr><td>${label}</td><td>${Number(val).toFixed(k.includes('deg') ? 2 : 4)}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('metrics').innerHTML = html;
    }

    function renderButtons() {
      const root = document.getElementById('buttons');
      root.innerHTML = '';
      variants.forEach(v => {
        const b = document.createElement('button');
        b.textContent = v;
        b.className = v === current ? 'active' : '';
        b.onclick = () => { current = v; render(); };
        root.appendChild(b);
      });
    }

    function render() {
      renderButtons();
      drawCanvas('top', [0, 2]);
      drawCanvas('side', [0, 1]);
      renderMetrics();
    }
    render();
  </script>
</body>
</html>
"""
    html = html.replace("__DATA__", json.dumps(data))
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data = build_visual_data(args.run_dir, args.episode_index, device)
    write_html(args.output_dir, data)
    print(f"Wrote {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
