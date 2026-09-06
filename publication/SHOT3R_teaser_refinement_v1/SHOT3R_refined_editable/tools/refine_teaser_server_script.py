#!/usr/bin/env python3
"""Refine the uploaded editable teaser with source RGB and native predictions."""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from lxml import etree as ET
import pyrender
import trimesh
import cairosvg

ROOT = Path(__file__).resolve().parent
MOVIE = ROOT.parents[1]
SOURCE = ROOT / 'source/SHOT3R_editable'
OUT = ROOT / 'SHOT3R_refined_editable'
PAYLOAD = MOVIE / 'output/bridge3r_two_dataset_demo_v2/egohumans_selected_0_177/payloads/bridge3r'
RGB_ROOT = MOVIE / 'publication/Shot3R_Teaser_AI_Handoff_20260906/02_原始视频帧/EgoHumans/大角度三视角_连续时间九帧'
S = 'http://www.w3.org/2000/svg'
X = 'http://www.w3.org/1999/xlink'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'p': P, 'a': A, 'r': R}
COLORS = {0: (160, 122, 199), 1: (236, 109, 91), 2: (44, 179, 197)}
TIMES = [0, 1, 3, 5]
SOURCE_FRAMES = [251, 261, 311, 331]
ALPHAS = [0.20, 0.32, 0.48, 1.0]
FLIP = np.diag([1., -1., -1.])  # Proper 180-degree rotation; shared by every object.
BOX = (145, 308, 1306, 431)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def look_at(eye, target):
    back = np.asarray(eye) - target
    back /= np.linalg.norm(back)
    right = np.cross([0., 1., 0.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, up, back])
    pose[:3, 3] = eye
    return pose


def load_geometry():
    frames, points, colors, confs = [], [], [], []
    for index in range(6):
        key = f'{index:06d}'
        with np.load(PAYLOAD / 'smpl' / (key + '.npz')) as data:
            frame = {k: data[k].copy() for k in ['verts_world', 'smpl_id', 'faces']}
            mask = np.squeeze(data['msk'])
        frames.append(frame)
        rgb = np.asarray(Image.open(PAYLOAD / 'color' / (key + '.png')).convert('RGB'))
        z = np.squeeze(np.load(PAYLOAD / 'depth' / (key + '.npy')))
        conf = np.squeeze(np.load(PAYLOAD / 'conf' / (key + '.npy')))
        with np.load(PAYLOAD / 'camera' / (key + '.npz')) as camera:
            pose, K = camera['pose'], camera['intrinsics']
        y, x = np.indices(z.shape)
        cam = np.stack([(x-K[0,2])*z/K[0,0], (y-K[1,2])*z/K[1,1], z], -1)
        xyz = cam @ pose[:3,:3].T + pose[:3,3]
        mask = cv2.dilate((mask >= .1).astype(np.uint8), np.ones((5,5), np.uint8)) > 0
        valid = np.isfinite(xyz).all(-1) & (z > 1e-5) & (conf > 1.0) & ~mask
        points.append(xyz[valid]); colors.append(rgb[valid]); confs.append(conf[valid])
    return frames, np.concatenate(points), np.concatenate(colors), np.concatenate(confs)


def camera_plan(frames, direction, width, height):
    body = np.concatenate([frames[t]['verts_world'].reshape(-1,3) for t in TIMES]) @ FLIP.T
    center = (body.min(0) + body.max(0)) / 2
    pose = look_at(center + np.asarray(direction)*8, center)
    projected = (body-pose[:3,3]) @ pose[:3,:3]
    low, high = projected[:,:2].min(0), projected[:,:2].max(0)
    offset = (low + high)/2
    pose[:3,3] += pose[:3,0]*offset[0] + pose[:3,1]*offset[1]
    ymag = float(max((high[1]-low[1])/.80/2, (high[0]-low[0])/.59/2/(width/height)))
    return dict(camera=pose, xmag=ymag*width/height, ymag=ymag, center=center,
                floor=float(body[:,1].min()), body=body)


def make_scene(plan):
    scene = pyrender.Scene(bg_color=[1.,1.,1.,0.], ambient_light=np.ones(3)*.38)
    scene.add(pyrender.OrthographicCamera(xmag=plan['xmag'], ymag=plan['ymag'], znear=.01,zfar=100), pose=plan['camera'])
    for offset, strength in [((3.,5.,4.),2.0), ((-3.,2.,2.),.7), ((0.,3.,-4.),.9)]:
        pose=look_at(plan['center']+np.array(offset)*3,plan['center'])
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=strength),pose=pose)
    return scene


def add_person(scene, verts, faces, identity):
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=tuple(np.array(COLORS[identity])/255)+(1.,),
        metallicFactor=.0,roughnessFactor=.75,doubleSided=True)
    mesh=trimesh.Trimesh(vertices=verts@FLIP.T,faces=faces,process=False)
    scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=True))


def scene_points(points, colors, confs, plan):
    pts=points@FLIP.T
    body=plan['body']; floor=plan['floor']
    keep=(pts[:,1] >= floor-.40)&(pts[:,1]<=body[:,1].max()+.48)
    keep &= np.linalg.norm(pts[:,[0,2]]-plan['center'][[0,2]],axis=1)<4.4
    # Open-front architectural cutaway: remove foreground walls to show the meshes.
    depth=(pts-plan['center'])@plan['camera'][:3,2]
    bdepth=(body-plan['center'])@plan['camera'][:3,2]
    keep &= (depth < np.percentile(bdepth,28)) | (pts[:,1]<floor+.32)
    ids=np.flatnonzero(keep)
    ids=ids[np.argsort(-confs[ids],kind='stable')]
    grid=np.floor(pts[ids]/.024).astype(np.int32)
    _,first=np.unique(grid,axis=0,return_index=True)
    ids=ids[first]
    # Sampling keeps actual predicted coordinates (no geometry averaging or filling).
    col=np.clip(colors[ids].astype(float)*.78+255*.22,0,255).astype(np.uint8)
    return pts[ids],col,ids


def fade_edges(image):
    a=np.array(image).copy(); h,w=a.shape[:2]
    y,x=np.indices((h,w),dtype=float)
    edge=np.minimum.reduce([x/(w*.065),(w-1-x)/(w*.065),y/(h*.07),(h-1-y)/(h*.07),np.ones((h,w))])
    a[:,:,3]=(a[:,:,3]*np.clip(edge,0,1)).astype(np.uint8)
    return Image.fromarray(a)


def render_layers(direction, width=2612, height=862, export=True, label='selected'):
    frames,points,colors,confs=load_geometry()
    plan=camera_plan(frames,direction,width,height)
    pts,col,ids=scene_points(points,colors,confs,plan)
    renderer=pyrender.OffscreenRenderer(width,height,point_size=2.4*width/2612)
    scene=make_scene(plan)
    scene.add(pyrender.Mesh.from_points(pts,colors=col))
    rgba,_=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.FLAT)
    background=fade_edges(Image.fromarray(rgba))
    combined=background.copy()
    layers=[('pointcloud_background',background)]
    for ti,t in enumerate(TIMES):
        for identity in sorted(COLORS):
            frame=frames[t]; match=np.flatnonzero(frame['smpl_id']==identity)
            if len(match)!=1:
                raise ValueError(f'Frame {t} has {len(match)} detections for ID {identity}')
            scene=make_scene(plan)
            add_person(scene,frame['verts_world'][match[0]],frame['faces'],identity)
            rgba,_=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SKIP_CULL_FACES)
            layer=Image.fromarray(rgba)
            if ti<3:
                layer=layer.filter(ImageFilter.GaussianBlur([1.2,.8,.35][ti]*width/2612))
                alpha=np.asarray(layer.getchannel('A')).astype(float)*ALPHAS[ti]
                layer.putalpha(Image.fromarray(alpha.astype(np.uint8)))
            layers.append((f'person_id{identity}_t{SOURCE_FRAMES[ti]}',layer))
            combined.alpha_composite(layer)
    renderer.delete()
    previews=ROOT/'work'; previews.mkdir(exist_ok=True)
    canvas=Image.new('RGBA',combined.size,'white'); canvas.alpha_composite(combined)
    canvas.convert('RGB').save(previews/f'{label}.jpg',quality=94)
    if export:
        assets=OUT/'assets';assets.mkdir(parents=True,exist_ok=True)
        for name,im in layers: im.save(assets/f'{name}.png')
        combined.save(assets/'reconstruction_combined.png')
        native=OUT/'geometry'; native.mkdir(exist_ok=True)
        trimesh.points.PointCloud(points,colors).export(native/'predicted_background_native.ply')
        trimesh.points.PointCloud(points[ids],colors[ids]).export(native/'displayed_background_native.ply')
        meshes=trimesh.Scene()
        for ti,t in enumerate(TIMES):
            frame=frames[t]
            for slot,identity in enumerate(frame['smpl_id']):
                mesh=trimesh.Trimesh(frame['verts_world'][slot],frame['faces'],process=False)
                mesh.visual.vertex_colors=np.array(COLORS[int(identity)]+(round(ALPHAS[ti]*255),))
                meshes.add_geometry(mesh,node_name=f'ID{identity}_frame{SOURCE_FRAMES[ti]}')
        meshes.export(native/'four_times_native_meshes.glb')
        np.savez_compressed(native/'four_times_native_meshes.npz',source_frames=SOURCE_FRAMES,
            faces=frames[0]['faces'],vertices=np.stack([frames[t]['verts_world'][[list(frames[t]['smpl_id']).index(i) for i in range(3)]] for t in TIMES]),
            persistent_ids=np.arange(3))
        dump(OUT/'render_settings.json',dict(source_payload=str(PAYLOAD),payload_indices=TIMES,
            source_frames=SOURCE_FRAMES,alpha=ALPHAS,identity_colors=COLORS,direction=direction,
            display_rotation=FLIP.tolist(),virtual_camera=plan['camera'].tolist(),xmag=plan['xmag'],ymag=plan['ymag'],
            raw_point_count=len(points),displayed_point_count=len(ids),voxel_size=.024,
            point_size_pixels=2.4,person_mask_dilation=5,confidence_min=1.0,color_white_mix=.22,
            cutaway='Foreground non-floor points and distant/ceiling points omitted for display; native PLY preserved.',
            geometry_repaired=False,synthetic_motion=False,source_inference_shots=['cam03','cam04'],
            three_shot_inference_claimed=False))
        dump(OUT/'assets/layer_manifest.json',[dict(name=name,path=name+'.png',box=list(BOX)) for name,_ in layers])
    print(json.dumps(dict(label=label,direction=direction,points=len(ids),view=str(previews/f'{label}.jpg'))),flush=True)
    return layers


def photos():
    assets=OUT/'assets'; assets.mkdir(parents=True,exist_ok=True)
    originals=OUT/'original_frames'; originals.mkdir(exist_ok=True)
    mapping=[]
    for shot,(cam,frames) in enumerate([('cam03',[251,261,271]),('cam06',[281,291,301]),('cam04',[311,321,331])],1):
        for j,f in enumerate(frames,1):
            source=list(RGB_ROOT.glob(f'*/*frame{f:05d}.jpg'))
            if len(source)!=1: raise ValueError((f,source))
            q=source[0]
            shutil.copy2(q,originals/f'shot{shot}_{cam}_frame{f:05d}.jpg')
            # Full originals are preserved. Layout assets keep source aspect and enough resolution for 2x output.
            im=Image.open(q).convert('RGB'); im.thumbnail((1600,1000),Image.Resampling.LANCZOS)
            im.save(assets/f'shot_{shot}_frame_{j}.png')
            mapping.append(dict(shot=shot,frame_in_shot=j,camera=cam,source_frame=f,path=str(q),middle_top=j==2))
    dump(OUT/'photo_sources.json',mapping)


def inline_png(path): return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def svg_text(parent,id,text,x,y,size=15,color='#42536a'):
    e=ET.SubElement(parent,'{'+S+'}text',id=id,x=str(x),y=str(y),fill=color)
    e.set('font-family','Arial, Liberation Sans, sans-serif');e.set('font-size',str(size));e.text=text
    return e


def compose_svg(layers):
    tree=ET.parse(str(SOURCE/'SHOT3R_layered.svg'));root=tree.getroot()
    for e in root.iter('{'+S+'}image'):
        name=e.get('id','')
        m=re.match(r'(?:top|timeline)-shot-(\d)-original-frame-(\d)',name)
        if m: e.set('{'+X+'}href',inline_png(OUT/'assets'/f'shot_{m[1]}_frame_{m[2]}.png'))
    group=next(e for e in root if e.get('id','').startswith('03-Reconstruction'))
    for e in list(group): group.remove(e)
    for name,_ in layers:
        e=ET.SubElement(group,'{'+S+'}image',id=name,x=str(BOX[0]),y=str(BOX[1]),width=str(BOX[2]),height=str(BOX[3]),preserveAspectRatio='none')
        e.set('{'+X+'}href',inline_png(OUT/'assets'/f'{name}.png'))
    for e in root.iter('{'+S+'}text'):
        if e.get('id')=='world-frame-summary':
            spans=list(e)
            if spans: spans[0].text='3 people · 4 times';spans[1].text='1 world frame.'
    # The orange schematic camera must not cover the actual latest mesh.
    for e in root.iter('{'+S+'}g'):
        if '99' in e.get('id','') and any(x.get('id')=='cam99-angle' for x in e.iter()):
            e.set('transform','translate(210,0)')
            break
    g=ET.SubElement(root,'{'+S+'}g',id='06-Rendering-evidence-and-time-legend')
    svg_text(g,'render-source-note','Actual prediction',24,519,15)
    svg_text(g,'render-source-views','0° → 177° replay',24,539,15)
    svg_text(g,'motion-note','Faint: earlier poses',1450,530,15)
    svg_text(g,'motion-latest-note','Solid: latest mesh',1450,550,15)
    for i,(identity,rgb) in enumerate(COLORS.items()):
        x=600+i*166
        ET.SubElement(g,'{'+S+'}circle',cx=str(x),cy='879',r='6',fill='#%02x%02x%02x'%rgb)
        svg_text(g,f'identity-{identity}',f'Person ID {identity}',x+14,884,15)
    svg_text(g,'camera-schematic-note','Camera symbols indicate source viewpoints; positions are schematic.',440,916,14)
    tree.write(str(OUT/'SHOT3R_refined_layered.svg'),encoding='utf-8',xml_declaration=True)
    cairosvg.svg2png(url=str(OUT/'SHOT3R_refined_layered.svg'),write_to=str(OUT/'SHOT3R_refined_preview.png'),output_width=2508,output_height=1412)
    cairosvg.svg2pdf(url=str(OUT/'SHOT3R_refined_layered.svg'),write_to=str(OUT/'SHOT3R_refined.pdf'))


def set_pic_box(pic,box):
    x,y,w,h=box; xf=pic.find('p:spPr/a:xfrm',NS)
    for e,values in [(xf.find('a:off',NS),dict(x=x,y=y)),(xf.find('a:ext',NS),dict(cx=w,cy=h))]:
        for k,v in values.items(): e.set(k,str(round(v*9144)))
    bf=pic.find('p:blipFill',NS)
    src=bf.find('a:srcRect',NS)
    if src is not None: bf.remove(src)


def ppt_text(template,id,name,text,x,y,w,h,size=15,color='42536A'):
    sp=copy.deepcopy(template); sp.find('p:nvSpPr/p:cNvPr',NS).attrib.update({'id':str(id),'name':name})
    xf=sp.find('p:spPr/a:xfrm',NS)
    xf.find('a:off',NS).attrib.update({'x':str(round(x*9144)),'y':str(round(y*9144))})
    xf.find('a:ext',NS).attrib.update({'cx':str(round(w*9144)),'cy':str(round(h*9144))})
    tx=sp.find('p:txBody',NS)
    for p in list(tx.findall('a:p',NS)): tx.remove(p)
    body=tx.find('a:bodyPr',NS);body.set('wrap','none')
    para=ET.SubElement(tx,'{'+A+'}p');ET.SubElement(para,'{'+A+'}pPr',algn='l')
    run=ET.SubElement(para,'{'+A+'}r');rp=ET.SubElement(run,'{'+A+'}rPr',lang='en-US',sz=str(round(size*72)))
    fill=ET.SubElement(rp,'{'+A+'}solidFill');ET.SubElement(fill,'{'+A+'}srgbClr',val=color)
    ET.SubElement(rp,'{'+A+'}latin',typeface='Arial');ET.SubElement(run,'{'+A+'}t').text=text
    return sp


def compose_ppt(layers):
    with zipfile.ZipFile(SOURCE/'SHOT3R_editable.pptx') as source:
        contents={name:source.read(name) for name in source.namelist()}
    slide=ET.fromstring(contents['ppt/slides/slide1.xml'])
    rels=ET.fromstring(contents['ppt/slides/_rels/slide1.xml.rels'])
    byrid={e.get('Id'):e for e in rels}
    pictures=slide.findall('.//p:pic',NS)
    for pic in pictures:
        desc=pic.find('p:nvPicPr/p:cNvPr',NS)
        m=re.match(r'(?:top|timeline)-shot-(\d)-original-frame-(\d)',desc.get('name',''))
        if m:
            rid=pic.find('p:blipFill/a:blip',NS).get('{'+R+'}embed')
            path='ppt/'+byrid[rid].get('Target').replace('../','')
            contents[path]=(OUT/'assets'/f'shot_{m[1]}_frame_{m[2]}.png').read_bytes()
    group=next(g for g in slide.findall('.//p:grpSp',NS) if g.find('p:nvGrpSpPr/p:cNvPr',NS).get('name','').startswith('03 Reconstruction'))
    template=copy.deepcopy(group.find('p:pic',NS))
    for e in list(group):
        if ET.QName(e).localname not in ('nvGrpSpPr','grpSpPr'): group.remove(e)
    nextid=max(int(e.get('id')) for e in slide.findall('.//p:cNvPr',NS))+1
    for i,(name,_) in enumerate(layers):
        pic=copy.deepcopy(template)
        pic.find('p:nvPicPr/p:cNvPr',NS).attrib.update({'id':str(nextid),'name':name,'descr':name+'.png'});nextid+=1
        rid=f'rIdRefined{i}';media=f'refined_{name}.png'
        pic.find('p:blipFill/a:blip',NS).set('{'+R+'}embed',rid)
        ET.SubElement(rels,'{'+REL+'}Relationship',Id=rid,Type=R+'/image',Target='../media/'+media)
        contents['ppt/media/'+media]=(OUT/'assets'/f'{name}.png').read_bytes()
        set_pic_box(pic,BOX);group.append(pic)
    for sp in slide.findall('.//p:sp',NS):
        if sp.find('p:nvSpPr/p:cNvPr',NS).get('name')=='world-frame-summary':
            texts=sp.findall('.//a:t',NS);texts[0].text='3 people · 4 times';texts[1].text='1 world frame.'
    for g in slide.findall('.//p:grpSp',NS):
        if g.find('p:nvGrpSpPr/p:cNvPr',NS).get('name')=='Camera — 99 degrees':
            off=g.find('p:grpSpPr/a:xfrm/a:off',NS);off.set('x',str(int(off.get('x'))+round(210*9144)))
    tmpl=next(s for s in slide.findall('.//p:sp',NS) if s.find('p:nvSpPr/p:cNvPr',NS).get('name')=='cam0-angle')
    tree=slide.find('p:cSld/p:spTree',NS)
    notes=[('render-source-note','Actual prediction',24,501,190,23,15,'42536A'),
        ('render-source-views','0° → 177° replay',24,521,190,23,15,'42536A'),
        ('motion-note','Faint: earlier poses',1450,512,215,23,15,'42536A'),
        ('motion-latest-note','Solid: latest mesh',1450,532,215,23,15,'42536A'),
        ('camera-note','Camera symbols indicate source viewpoints; positions are schematic.',440,897,1000,23,14,'42536A')]
    for identity,rgb in COLORS.items():
        notes.append((f'identity-{identity}',f'●  Person ID {identity}',594+identity*166,864,165,27,15,'%02X%02X%02X'%rgb))
    for name,text,x,y,w,h,size,color in notes:
        tree.append(ppt_text(tmpl,nextid,name,text,x,y,w,h,size,color));nextid+=1
    contents['ppt/slides/slide1.xml']=ET.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True)
    contents['ppt/slides/_rels/slide1.xml.rels']=ET.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
    # Drop unused legacy conceptual images; keep only relationships referenced by the slide.
    used={e.get('{'+R+'}embed') for e in slide.findall('.//a:blip',NS)}
    for rel in list(rels):
        if rel.get('Type','').endswith('/image') and rel.get('Id') not in used:
            contents.pop('ppt/'+rel.get('Target').replace('../',''),None);rels.remove(rel)
    contents['ppt/slides/_rels/slide1.xml.rels']=ET.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
    # Replace inherited slide-note prose with the actual source contract.
    if 'ppt/notesSlides/notesSlide1.xml' in contents:
        root=ET.fromstring(contents['ppt/notesSlides/notesSlide1.xml']);texts=root.findall('.//a:t',NS)
        if texts:
            texts[0].text='Real source RGB: cam03 251/261/271; cam06 281/291/301; cam04 311/321/331. Center: four real times 251/261/311/331 from the prior 0-to-177-degree BRIDGE3R inference. Historical alpha 0.20/0.32/0.48, final 1.0. Camera symbols schematic. See README and render_settings.json.'
            for t in texts[1:]:t.text=''
        contents['ppt/notesSlides/notesSlide1.xml']=ET.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    with zipfile.ZipFile(OUT/'SHOT3R_refined_editable.pptx','w',zipfile.ZIP_DEFLATED) as z:
        for path,data in contents.items():z.writestr(path,data)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['candidates','build','compose'])
    p.add_argument('--direction',nargs=3,type=float,default=[1.5,.72,1.5]);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='candidates':
        for i,d in enumerate([(1.5,.72,1.5),(-1.5,.72,1.5),(1.6,.65,-1.2),(-1.6,.65,-1.2),(0.,.7,2.)]):
            render_layers(d,width=1306,height=431,export=False,label=f'camera_{i}')
        return
    if args.stage=='build':
        layers=render_layers(args.direction);photos()
    else:
        layers=[(e['name'],None) for e in json.loads((OUT/'assets/layer_manifest.json').read_text())]
    compose_svg(layers);compose_ppt(layers)
    print(OUT,flush=True)


if __name__=='__main__':main()
