#!/usr/bin/env python3
"""Continuous RGB / native prediction video, for auditing and figure selection.

中文：连续 60 帧，按源片段 20 FPS 播放；不是推理速度演示。
所有有效预测均显示；真实漏检、姿态误差和身份变化不被插值或修复。
"""
import json
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import build_teaser as b


def main():
    b.FACES=b.load_faces()
    c=b.get_case('egohumans')
    c['selected']=list(range(15,75))
    out=b.ROOT/'assets/egohumans/continuous'
    for folder in ['rgb','renders','video_frames']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    p=b.plan(c,800,600)
    renderer=pyrender.OffscreenRenderer(800,600)
    font=ImageFont.truetype(b.FONT,25)
    small=ImageFont.truetype(b.FONT,18)
    records=[]
    for i,t in enumerate(c['selected']):
        raw=out/'rgb'/f'f{t:03d}.jpg'
        shutil.copy2(c['rgb'][t],raw)
        scene=b.new_scene(p)
        b.add_people(scene,c,t,p)
        rgba,_=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SKIP_CULL_FACES)
        b.save_image(out/'renders'/f'f{t:03d}_transparent.png',rgba)
        canvas=Image.new('RGB',(1600,720),'white')
        d=ImageDraw.Draw(canvas)
        d.text((24,18),'Shot3R | Continuous monocular input and reconstruction',fill='#172d3d',font=font)
        d.text((24,55),'Source-rate playback (20 FPS), not measured inference speed',fill='#657685',font=small)
        im=Image.open(raw);im.thumbnail((780,550))
        canvas.paste(im,(12+(780-im.width)//2,111+(550-im.height)//2))
        render=Image.fromarray(rgba)
        canvas.paste(render,(800,93),render)
        shot=b.shot(c,t)+1
        color='#3371ca' if shot==1 else '#df9525'
        d.rectangle((18,97,790,101),fill=color)
        d.text((26,682),f'Shot {shot} | f{t:03d} | {t/20:.2f} s from clip start',font=small,fill=color)
        d.text((820,682),'Fixed world view | Colors = native persistent IDs',font=small,fill='#657685')
        if t in [50,51,52,53,54]:
            d.rounded_rectangle((644,115,785,152),radius=6,fill='#df9525')
            d.text((658,123),'177 deg cut',font=small,fill='white')
        canvas.save(out/'video_frames'/f'{i:06d}.png')
        rec=b.source_record(c,t)
        rec['rgb_sha256']=b.sha256(raw)
        rec['valid_ids']=c['arrays']['persistent_ids'][t][c['arrays']['valid'][t].astype(bool)].tolist()
        records.append(rec)
        if i%15==0:print('continuous frame',t,flush=True)
    renderer.delete()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','20',
                    '-i',str(out/'video_frames/%06d.png'),'-c:v','libx264','-crf','19',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'Shot3R_continuous_real_predictions.mp4')],check=True)
    b.write_json(out/'manifest.json',dict(frames=records,source_fps=20,playback_fps=20,
        inference_speed_claim=False,interpolation=False,
        fixed_view={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()},
        labels_zh_en={
            'Continuous monocular input and reconstruction':'连续单目输入与重建结果',
            'Source-rate playback (20 FPS), not measured inference speed':'按源片段 20 FPS 播放，并非测得的推理速度',
            'Fixed world view | Colors = native persistent IDs':'固定世界视角；颜色对应原始预测的持续人物 ID'}))
    print(out/'Shot3R_continuous_real_predictions.mp4',flush=True)


if __name__=='__main__':main()
