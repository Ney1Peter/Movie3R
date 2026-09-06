#!/usr/bin/env python3
"""Export evidence-backed Shot3R teaser assets; never modify source predictions.

中文：所有人体/相机来自已冻结的预测；颜色直接对应 persistent ID。
仅改变渲染视角、照明和统一显示坐标，不修复网格、不重排人物身份。
"""
from __future__ import annotations

import argparse
import csv
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import trimesh
import cairosvg

ROOT = Path(__file__).resolve().parent
MOVIE = ROOT.parents[1]
WORK = MOVIE.parent
sys.path[:0] = [str(MOVIE), str(MOVIE / "src")]
from publication.bridge3r_iclr2027.export_two_dataset_demo_qualitative import (
    CASES, bridge3r_arrays, arrays_from_npz, load_faces, display_flip, look_at,
)

PALETTE = [(232, 103, 85), (20, 166, 160), (133, 105, 208), (219, 160, 45),
           (68, 125, 201), (208, 102, 159), (105, 136, 67), (91, 106, 132),
           (176, 108, 70), (111, 196, 191)]
SHOT_COLORS = [(51, 113, 202), (223, 149, 37), (133, 90, 196)]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FACES = None


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_image(p, a):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p)


def get_case(name):
    if name == "egohumans":
        spec = CASES[name]
        runtime = read_json(spec.runtime)
        arrays, _ = bridge3r_arrays(spec, runtime)
        demo = MOVIE / "output/bridge3r_two_dataset_demo_v2/egohumans"
        record = runtime["record"]
        metrics_path = MOVIE / "output/v19_egohumans/test/summary/case_metrics.csv"
        with metrics_path.open() as f:
            metrics = next(row for row in csv.DictReader(f)
                           if row['case_id']==record['case_id'] and row['method']=='v19_egohumans_frozen')
        sources = [demo / "_model_replay_rgb" / f"{t:06d}.jpg" for t in range(75)]
        return dict(name=name, record=record, arrays=arrays, cache=spec.cache,
                    runtime=spec.runtime, rgb=sources, selected=[15, 32, 42, 49, 50, 61, 74],
                    boundaries=[50], angles=[spec.angle_deg], fps=20., payload=demo / "payloads/bridge3r",
                    people=3, idf1=float(metrics['IDF1']),
                    w_mpjpe_mm=float(metrics['W-MPJPE_mm']), wa_mpjpe_mm=float(metrics['WA-MPJPE_mm']),
                    metrics_source=str(metrics_path),metrics_method='v19_egohumans_frozen')
    run = MOVIE / "publication/bridge3r_iclr2027/multicut/runs" / name
    runtime = read_json(run.with_suffix(".runtime.json"))
    record = runtime["record"]
    arrays = arrays_from_npz(run.with_suffix(".npz"), "bridge3r")
    rgb = []
    for cam, numbers in zip(record["shot_cameras"], record["shot_frame_numbers"]):
        rgb += [WORK / "data/Bridge3R_multicut_harmony4d/staging" / f"train_{record['sequence']}" /
                record["sequence"] / record["capture"] / "exo" / cam / "images" / f"{n:05d}.jpg"
                for n in numbers]
    ev = read_json(run.with_suffix(".evaluation.json"))["methods"]["bridge3r"]
    metrics = ev["multi_thumbs_named_provisional"]
    return dict(name=name, record=record, arrays=arrays, cache=run.with_suffix(".npz"),
                runtime=run.with_suffix(".runtime.json"), rgb=rgb,
                selected=[12, 32, 49, 50, 75, 99, 100, 125, 149],
                boundaries=record["boundaries"],
                angles=[v["rotation_deg"] for v in record["camera_transitions_evaluator_only"]],
                fps=30., payload=None, people=2, idf1=ev["identity"]["idf1"],
                w_mpjpe_mm=metrics["w_mpjpe_mm"]["mean"],
                wa_mpjpe_mm=metrics["wa_mpjpe_mm"]["mean"])


def shot(case, t):
    return int(np.searchsorted(case["boundaries"], t, side="right"))


def source_record(case, t):
    r = case["record"]
    s = shot(case, t)
    if case["name"] == "egohumans":
        cam = r["pre_camera"] if s == 0 else r["post_camera"]
        n = (r["pre_frame_numbers"] + r["post_frame_numbers"])[t]
    else:
        cam = r["shot_cameras"][s]
        n = sum(r["shot_frame_numbers"], [])[t]
    return dict(clip_frame_zero_based=t, source_frame=n, camera=cam,
                shot_one_based=s + 1, seconds_from_clip_start=t / case["fps"])


def contact_sheet(paths, labels, destination, columns=6, title=""):
    w, h, labelh = 340, 215, 42
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * w, rows * (h + labelh) + 64), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 16)
    draw.text((18, 16), title, fill="#132b3b", font=ImageFont.truetype(FONT, 24))
    for i, (p, label) in enumerate(zip(paths, labels)):
        raw = Image.open(p).convert("RGBA")
        im = Image.new("RGBA", raw.size, "white")
        im.alpha_composite(raw)
        im = im.convert("RGB")
        im.thumbnail((w - 8, h - 8))
        x, y = (i % columns) * w, 64 + (i // columns) * (h + labelh)
        canvas.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))
        draw.text((x + 8, y + h), label, fill="#34495b", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def audit():
    paths, labels, metrics = [], [], []
    for name in ["egohumans", "case_01", "case_02", "case_03", "case_04"]:
        c = get_case(name)
        indices = [15,32,49,50,61,74] if name == "egohumans" else [12, 49, 50, 99, 100, 149]
        for t in indices:
            paths.append(c["rgb"][t])
            labels.append(f"{name}  f{t:03d} / {source_record(c,t)['camera']}")
        metrics.append({k: c[k] for k in ["name", "people", "angles", "idf1", "w_mpjpe_mm", "wa_mpjpe_mm"]})
    contact_sheet(paths, labels, ROOT / "selection/candidate_rgb_contact_sheet.jpg", title="Candidate audit | real RGB, ordered in time")
    write_json(ROOT / "selection/candidate_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def export(case):
    dest = ROOT / "assets" / case["name"]
    dest.mkdir(parents=True, exist_ok=True)
    selected = case["selected"]
    frame_records = []
    arrays = case["arrays"]
    for t in selected:
        rgb_out = dest / "rgb" / f"f{t:03d}.jpg"
        rgb_out.parent.mkdir(exist_ok=True)
        shutil.copy2(case["rgb"][t], rgb_out)
        rec = source_record(case, t)
        rec.update(source_rgb=str(case["rgb"][t]), copied_rgb=str(rgb_out.relative_to(ROOT)),
                   rgb_sha256=sha256(rgb_out),
                   valid_persistent_ids=arrays["persistent_ids"][t][arrays["valid"][t].astype(bool)].tolist())
        frame_records.append(rec)
        mesh_dir = dest / "meshes" / f"f{t:03d}"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        scene = trimesh.Scene()
        for p in np.flatnonzero(arrays["valid"][t]):
            pid = int(arrays["persistent_ids"][t, p])
            m = trimesh.Trimesh(vertices=arrays["vertices_world"][t, p], faces=FACES, process=False)
            m.visual.vertex_colors = np.array(PALETTE[pid % len(PALETTE)] + (255,))
            m.export(mesh_dir / f"id{pid:02d}_slot{p:02d}.ply")
            scene.add_geometry(m, node_name=f"persistent_id_{pid}_slot_{p}")
        scene.export(mesh_dir / "all_people_native_world.glb")
    np.savez_compressed(dest / "selected_predictions_native_world.npz", frame_indices=np.asarray(selected),
                        faces=FACES, **{k: v[selected] for k, v in arrays.items()})
    # Preserve the complete frozen sequence for interactive inspection/reuse.
    np.savez_compressed(dest / "full_predictions_native_world.npz", faces=FACES, **arrays)
    if case["payload"]:
        p = case["payload"]
        meta = read_json(p / "metadata.json")
        for local, t in enumerate(meta["frame_indices"]):
            if t not in [45, 49, 50, 61, 74]:
                continue
            for kind, ext in [("color", "png"), ("camera", "npz"), ("smpl", "npz"), ("depth", "npy"), ("conf", "npy")]:
                q = dest / "scene_payload" / kind / f"f{t:03d}.{ext}"
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p / kind / f"{local:06d}.{ext}", q)
        shutil.copy2(p / "metadata.json", dest / "scene_payload/source_metadata.json")
    write_json(dest / "provenance.json", dict(
        case_id=case["record"]["case_id"], source_cache=str(case["cache"]),
        source_cache_sha256=sha256(case["cache"]), source_runtime=str(case["runtime"]),
        source_runtime_sha256=sha256(case["runtime"]),
        protocol=case["record"]["protocol"], fps=case["fps"], boundaries=case["boundaries"],
        angle_degrees_evaluator_annotation_only=case["angles"],
        idf1_full_evaluation_clip=case["idf1"], displayed_frames=frame_records,
        metrics_source=case.get('metrics_source',str(case['cache'].with_suffix('.evaluation.json'))),
        color_mapping={str(i): "#%02x%02x%02x" % tuple(v) for i, v in enumerate(PALETTE)},
        geometry_modified=False, ids_relabelled=False, ground_truth_geometry_used=False,
        full_sequence_prediction_prefix="locked transaction(m3_b0_only)" if case["name"] == "egohumans" else "bridge3r",
        display_only="One fixed y-axis sign convention and one fixed virtual camera per sequence. Neutral ground is a display aid, not a reconstructed plane.",
        rgb_note="Exact copies of existing staged RGB; EgoHumans staged JPEGs were decoded/re-encoded by the previous exporter, not archive-byte-identical. Faces were already anonymized in source."))
    print("exported", case["name"], flush=True)


def add_mesh(scene, verts, color, alpha=1., faces=None, roughness=.62):
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=tuple(np.asarray(color) / 255.) + (alpha,),
        metallicFactor=.08, roughnessFactor=roughness,
        alphaMode="BLEND" if alpha < 1. else "OPAQUE", doubleSided=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=FACES if faces is None else faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))


def cylinder(scene, a, b, color, radius=.009):
    a, b = np.asarray(a), np.asarray(b)
    if np.linalg.norm(a - b) < 1e-6:
        return
    m = trimesh.creation.cylinder(radius=radius, segment=np.vstack([a,b]), sections=8)
    add_mesh(scene, m.vertices, color, faces=m.faces)


def frustum(scene, pose, color, size=.32, radius=.009, intrinsic=None, image_shape=None):
    if intrinsic is None:
        corners = np.array([[-.8,-.45,1],[.8,-.45,1],[.8,.45,1],[-.8,.45,1]]) * size
    else:
        h,w = image_shape
        pix = np.array([[0,0],[w,0],[w,h],[0,h]])
        corners = np.c_[(pix[:,0]-intrinsic[0,2])/intrinsic[0,0],
                        (pix[:,1]-intrinsic[1,2])/intrinsic[1,1],np.ones(4)] * size
    points = np.vstack([pose[:3,3],corners @ pose[:3,:3].T + pose[:3,3]])
    for i,j in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]:
        cylinder(scene, points[i], points[j], color, radius)
    return points


def plan(case, width, height, include_cameras=False, direction=(1.35,.88,1.55)):
    a = case["arrays"]
    ids = case["selected"]
    flip = display_flip(a, ids)
    v = a["vertices_world"][ids][a["valid"][ids].astype(bool)] @ flip.T
    pts = v.reshape(-1,3)[::35]
    if include_cameras:
        pts = np.concatenate([pts, a["cameras_c2w"][ids,:3,3] @ flip.T])
    low, high = np.min(pts,axis=0), np.max(pts,axis=0)
    target = (low+high)/2
    span = max(float(np.ptp(pts,axis=0).max()),2.)
    camera = look_at(target+span*np.array(direction), target)
    pc = (pts-camera[:3,3]) @ camera[:3,:3]
    pc_low,pc_high = pc.min(0),pc.max(0)
    center_shift = (pc_low+pc_high)/2
    camera[:3,3] += camera[:3,0]*center_shift[0]+camera[:3,1]*center_shift[1]
    xmag,ymag = (pc_high[0]-pc_low[0])*.58,(pc_high[1]-pc_low[1])*.58
    aspect = width/height
    xmag = max(xmag,ymag*aspect)
    ymag = xmag/aspect
    return dict(flip=flip,camera=camera,xmag=xmag,ymag=ymag,
                floor=float(v[...,1].min())-.015,center=np.median(pts,axis=0),span=span)


def new_scene(p, transparent=True):
    scene = pyrender.Scene(bg_color=[1,1,1,0 if transparent else 1], ambient_light=np.ones(3)*.32)
    scene.add(pyrender.OrthographicCamera(xmag=p["xmag"],ymag=p["ymag"],znear=.01,zfar=1000.),pose=p["camera"])
    for direction,intensity in [((1.,2.,3.),2.0),((-2.,1.,1.),.65),((0.,2.,-2.),.8)]:
        pose = look_at(p["center"] + np.array(direction)*p["span"],p["center"])
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=intensity),pose=pose)
    return scene


def add_people(scene, case, t, p, alpha=1.):
    a=case["arrays"]
    for j in np.flatnonzero(a["valid"][t]):
        pid=int(a["persistent_ids"][t,j])
        add_mesh(scene,a["vertices_world"][t,j]@p["flip"].T,PALETTE[pid%len(PALETTE)],alpha)


def add_ground(scene,p):
    floor=trimesh.creation.box(extents=[p["span"]*1.6,.012,p["span"]*1.2])
    floor.apply_translation([p["center"][0],p["floor"]-.01,p["center"][2]])
    add_mesh(scene,floor.vertices,(242,245,247),faces=floor.faces)
    for offset in np.arange(-p["span"],p["span"]+.1,.5):
        center=p["center"]
        if abs(offset)>p["span"]*.55:
            continue
        cylinder(scene,[center[0]-p["span"]*.7,p["floor"],center[2]+offset],
                 [center[0]+p["span"]*.7,p["floor"],center[2]+offset],(214,221,227),.002)


def render_frames(case, width=800, height=640):
    dest=ROOT/"assets"/case["name"]/"renders"
    dest.mkdir(parents=True,exist_ok=True)
    p=plan(case,width,height)
    renderer=pyrender.OffscreenRenderer(width,height,point_size=2.)
    outputs=[]
    for t in case["selected"]:
        scene=new_scene(p)
        add_people(scene,case,t,p)
        rgba,_=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SKIP_CULL_FACES)
        q=dest/f"f{t:03d}_transparent.png"
        save_image(q,rgba)
        outputs.append(q)
    renderer.delete()
    write_json(dest/"fixed_view.json",{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()})
    contact_sheet(outputs,[f"f{t:03d}  IDs: {case['arrays']['persistent_ids'][t][case['arrays']['valid'][t].astype(bool)].tolist()}" for t in case['selected']],
                  ROOT/"selection"/f"{case['name']}_mesh_contact_sheet.jpg",columns=3,
                  title=f"{case['name']} | all valid predictions; fixed world view")
    print("rendered",case['name'],flush=True)


def load_scene_points(case, selected=(49,50)):
    root=case["payload"]
    meta=read_json(root/"metadata.json")
    points,colors=[],[]
    for t in selected:
        loc=meta["frame_indices"].index(t)
        name=f"{loc:06d}"
        rgb=np.asarray(Image.open(root/"color"/f"{name}.png").convert("RGB"))
        with np.load(root/"camera"/f"{name}.npz") as d:
            pose,K=d['pose'],d['intrinsics']
        with np.load(root/"smpl"/f"{name}.npz") as d:
            mask=np.squeeze(d['msk'])
        z=np.squeeze(np.load(root/"depth"/f"{name}.npy"))
        conf=np.squeeze(np.load(root/"conf"/f"{name}.npy"))
        h,w=rgb.shape[:2]
        assert z.shape == (h,w)
        yy,xx=np.indices((h,w))
        cam=np.stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z],-1)
        xyz=cam@pose[:3,:3].T+pose[:3,3]
        keep=np.isfinite(xyz).all(-1)&(z>1e-5)&(conf>1.)&(mask<.1)
        points.append(xyz[keep]);colors.append(rgb[keep])
    return np.concatenate(points),np.concatenate(colors)


def render_world(case, width=1800, height=680, direction=(1.35,.88,1.55), suffix="world"):
    p=plan(case,width,height,include_cameras=True,direction=direction)
    a=case["arrays"]
    scene=new_scene(p)
    xyz,col=load_scene_points(case)
    trimesh.points.PointCloud(xyz,col).export(ROOT/"assets"/case["name"]/"scene_points_native_world.ply")
    pts=xyz@p["flip"].T
    human=a["vertices_world"][case["selected"]][a["valid"][case["selected"]].astype(bool)]@p['flip'].T
    hmin,hmax=human.reshape(-1,3).min(0),human.reshape(-1,3).max(0)
    # Cutaway illustration only: omit distant walls/ceiling and retain raw PLY above.
    keep=(pts[:,1] >= hmin[1]-.65)&(pts[:,1] <= hmax[1]+.12)
    keep &= np.linalg.norm(pts[:,[0,2]]-p['center'][[0,2]],axis=1)<4.6
    front_depth=(pts-p['center'])@p['camera'][:3,2]
    body_depth=(human.reshape(-1,3)-p['center'])@p['camera'][:3,2]
    # Architectural cutaway: remove foreground walls, not human predictions.
    # 中文：仅移除遮挡人物的前侧场景点，所有未裁切点另存为原始 PLY。
    keep &= (front_depth<=np.percentile(body_depth,35)) | (pts[:,1]<hmin[1]+.25)
    pts,col=pts[keep][::2],col[keep][::2]
    col=np.clip(col*.64+255*.36,0,255).astype(np.uint8)
    scene.add(pyrender.Mesh.from_points(pts,colors=col))
    add_people(scene,case,42,p,alpha=.19)
    add_people(scene,case,74,p,alpha=1.)
    camera_projected=[]
    for t in [15,32,49,50,61,74]:
        pose=a["cameras_c2w"][t].copy()
        pose[:3,:3]=p["flip"]@pose[:3,:3]
        pose[:3,3]=p["flip"]@pose[:3,3]
        local=max(0,min(29,t-45))
        with np.load(case["payload"]/"camera"/f"{local:06d}.npz") as d:
            K=d["intrinsics"]
        # Intrinsics for pre-payload frames are unavailable: show only their center.
        if t<45:
            dot=trimesh.creation.icosphere(subdivisions=2,radius=.035)
            dot.apply_translation(pose[:3,3]); add_mesh(scene,dot.vertices,SHOT_COLORS[shot(case,t)],faces=dot.faces)
        else:
            frustum(scene,pose,SHOT_COLORS[shot(case,t)],size=.36 if t in [49,50] else .24,
                    radius=.013 if t in [49,50] else .006,intrinsic=K,image_shape=(288,512))
        q=(pose[:3,3]-p["camera"][:3,3])@p["camera"][:3,:3]
        camera_projected.append(dict(frame=t,x=float((q[0]/p['xmag']+1)*width/2),y=float((1-q[1]/p['ymag'])*height/2)))
    for times in [[15,32,49],[50,61,74]]:
        for first,last in zip(times[:-1],times[1:]):
            cylinder(scene,a['cameras_c2w'][first,:3,3]@p['flip'].T,a['cameras_c2w'][last,:3,3]@p['flip'].T,SHOT_COLORS[shot(case,first)],.008)
    renderer=pyrender.OffscreenRenderer(width,height,point_size=2.3)
    rgba,_=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SKIP_CULL_FACES)
    renderer.delete()
    dest=ROOT/"assets"/case['name']/"renders"/f"{suffix}.png"
    save_image(dest,rgba)
    write_json(dest.with_suffix('.json'),dict(scene_frames=[49,50],mesh_frames=[42,74],
               historical_mesh_alpha=.19,current_mesh_alpha=1.,
               display_transform=p['flip'].tolist(),virtual_camera=p['camera'].tolist(),
               xmag=p['xmag'],ymag=p['ymag'],camera_projected=camera_projected,
               scene_display_filter="confidence > 1, human mask < 0.1; height cutaway; radius < 4.6; non-floor points in front of body-depth percentile 35 omitted for cutaway; stride 2; colors mixed with 36% white",
               native_points_file="scene_points_native_world.ply"))
    print('world',dest,flush=True)


def main():
    global FACES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','export','render','world'])
    parser.add_argument('--cases',nargs='+',default=['egohumans','case_01','case_02'])
    args=parser.parse_args()
    FACES=load_faces()
    if args.stage=='audit':
        audit();return
    for name in args.cases:
        c=get_case(name)
        if args.stage=='export':export(c)
        elif args.stage=='render':render_frames(c)
        elif args.stage=='world':
            for direction,suffix in [((1.35,.88,1.55),'world_a'),((-1.5,.85,1.2),'world_b'),((.3,.85,-1.5),'world_c')]:
                render_world(c,direction=direction,suffix=suffix)


if __name__=='__main__':
    main()
