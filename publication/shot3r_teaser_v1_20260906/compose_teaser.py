#!/usr/bin/env python3
"""Editable, evidence-based figure composition. 中文：真实素材，矢量排版。

The raster panels contain the actual predictions; typography, rules, legends,
time arrows and cut annotations are editable SVG objects. No generated imagery.
"""
from pathlib import Path
import base64
import html
import json
import io
import cairosvg
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
INK='#172d3d'
MUTED='#657685'
TEAL='#14a6a0'
BLUE='#3371ca'
AMBER='#df9525'
CORAL='#e86755'
PURPLE='#8569d0'
TRANSLATIONS={
    'Streaming multi-person 4D reconstruction':'流式多人体四维重建',
    'A shared 3D world':'统一的三维世界坐标系',
    'Same world, same viewing angle':'同一世界坐标系、同一渲染视角',
    'Across a 177° viewpoint change':'跨越约 177° 的视角变化',
    'Predicted identities':'预测人物身份',
    'Shot 1':'镜头 1','Shot 2':'镜头 2','Shot 3':'镜头 3',
    'Monocular input stream':'单目视频输入流',
    'Sampled frames; time increases left to right':'抽样帧；时间从左向右推进',
    'Online updates':'在线更新',
    'Shared world coordinates':'统一世界坐标系',
    'Persistent person IDs':'持续关联的人物 ID',
    'Shot cut':'镜头切换',
    'Before the cut':'切镜前','After the cut':'切镜后',
    'Predicted cameras':'预测相机',
    'Faint meshes: earlier poses':'浅色网格：较早时刻的姿态',
    'Three-shot candidate':'三镜头候选',
    'Fixed world view; all valid predictions shown':'固定世界视角；保留全部有效预测',
    'Diagnostic preview, not the recommended teaser':'诊断预览，不推荐用作主 teaser',
    'Full-clip IDF1: 0.559':'完整测试片段的 IDF1：0.559',
    'Full-clip IDF1: 0.951':'完整测试片段的 IDF1：0.951',
    'EgoHumans · three people':'EgoHumans · 三人',
    'Shot cut · 177°':'镜头切换 · 约 177°',
}


class SVG:
    def __init__(self,w,h,title):
        self.w,self.h=w,h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                    f'<title>{html.escape(title)}</title>',
                    '<desc>Shot3R 真实实验可视化。原始网格和 ID 未经修复或重新标注；图像为抽样时刻，并非同时采集的多视角输入。</desc>',
                    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9" fill="none" stroke="#657685" stroke-width="1.8"/></marker></defs>',
                    '<rect width="100%" height="100%" fill="white"/>']

    def rect(self,x,y,w,h,fill='none',stroke='none',sw=1,rx=0):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self,x,y,s,size=22,color=INK,weight=400,anchor='start',spacing=None):
        size=max(size,20)
        if s in TRANSLATIONS:
            self.parts.append(f'<!-- {html.escape(s)}：{TRANSLATIONS[s]} -->')
        space=f' letter-spacing="{spacing}"' if spacing else ''
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Liberation Sans, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{space}>{html.escape(s)}</text>')

    def line(self,x1,y1,x2,y2,color='#dce3e9',width=1,dash=None,arrow=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+
                          (f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def image(self,path,x,y,w,h):
        p=Path(path)
        mime='image/png' if p.suffix=='.png' else 'image/jpeg'
        # Figure embeds are sized for the 2x export. Full-resolution original
        # assets remain separate and unchanged. 中文：内嵌缩略图，原图独立保留。
        im=Image.open(p)
        im.thumbnail((round(w*2),round(h*2)),Image.Resampling.LANCZOS)
        buffer=io.BytesIO()
        if mime=='image/jpeg':
            im.convert('RGB').save(buffer,format='JPEG',quality=95)
        else:
            im.save(buffer,format='PNG')
        data=base64.b64encode(buffer.getvalue()).decode()
        self.parts.append(f'<!-- Asset: {p.relative_to(ROOT)} -->')
        self.parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" xlink:href="data:{mime};base64,{data}"/>')

    def dot(self,x,y,color,r=5):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def save(self,name):
        OUT.mkdir(exist_ok=True)
        content='\n'.join(self.parts+['</svg>'])
        p=OUT/f'{name}.svg'
        p.write_text(content)
        cairosvg.svg2pdf(bytestring=content.encode(),write_to=str(p.with_suffix('.pdf')))
        cairosvg.svg2png(bytestring=content.encode(),write_to=str(p.with_suffix('.png')),output_width=self.w*2,output_height=self.h*2)
        cairosvg.svg2png(bytestring=content.encode(),write_to=str(OUT/f'{name}_preview.png'),output_width=1800,output_height=round(1800*self.h/self.w))
        print(p,flush=True)


def heading(s,diagnostic=False):
    s.text(30,51,'Shot3R',43,INK,700)
    s.line(228,22,228,56,'#d9e1e6',2)
    s.text(250,48,'Streaming multi-person 4D reconstruction',30,INK,500)
    if not diagnostic:
        s.text(1770,33,'EgoHumans · three people',19,MUTED,anchor='end')
        s.text(1770,57,'Full-clip IDF1: 0.951',18,TEAL,600,anchor='end')
    s.line(30,79,1770,79)


def identity_legend(s,x,y):
    s.text(x,y,'Predicted identities',19,MUTED)
    for i,col in enumerate([CORAL,TEAL,PURPLE]):
        at=x+206+i*90
        s.dot(at,y-6,col,6)
        s.text(at+14,y,f'ID {i}',19,INK,600)


def filmstrip(s,indices,y,case='egohumans',w=1740,x=30,thumb_h=None,labels=True):
    gap=14
    cw=(w-gap*(len(indices)-1))/len(indices)
    ch=thumb_h or cw*9/16
    cuts=[50] if case=='egohumans' else [50,100]
    colors=[BLUE,AMBER,PURPLE]
    for j,t in enumerate(indices):
        xx=x+j*(cw+gap)
        shot=sum(t>=cut for cut in cuts)
        s.rect(xx,y-4,cw,ch+8,'#f6f8fa')
        s.image(ROOT/'assets'/case/'rgb'/f'f{t:03d}.jpg',xx,y,cw,ch)
        s.rect(xx,y-4,cw,3,colors[shot])
        if labels:
            fps=20 if case=='egohumans' else 30
            s.text(xx+cw/2,y+ch+26,f'{t/fps:.2f} s · f{t:03d}',17,MUTED,anchor='middle')
    return cw,ch


def main_teaser():
    s=SVG(1800,880,'Shot3R teaser / 主 teaser：统一场景、身份关联和单目输入时间带')
    heading(s)
    s.text(38,119,'A shared 3D world',25,INK,600)
    identity_legend(s,38,151)
    # Original world render is a cutaway; no panel-specific 3D realignment.
    s.image(ROOT/'assets/egohumans/renders/world_a.png',12,157,1170,435)
    s.text(37,576,'Faint meshes: earlier poses',17,MUTED)
    s.line(1197,110,1197,575,'#dce3e9')
    s.text(1491,120,'Across a 177° viewpoint change',24,INK,600,anchor='middle')
    s.text(1491,150,'Same world, same viewing angle',18,MUTED,anchor='middle')
    for x,t,label,col in [(1217,42,'Before the cut',BLUE),(1503,50,'After the cut',AMBER)]:
        s.rect(x,175,266,319,'#f5f8fa',rx=8)
        s.rect(x+12,187,242,3,col)
        s.text(x+133,218,label,20,col,600,anchor='middle')
        s.image(ROOT/'assets/egohumans/renders'/f'f{t:03d}_transparent.png',x+3,229,260,227)
        s.text(x+133,478,f'{t/20:.2f} s · f{t:03d}',17,MUTED,anchor='middle')
    s.line(1477,332,1494,332,MUTED,1.6,arrow=True)
    s.text(1492,539,'Persistent person IDs',21,TEAL,600,anchor='middle')
    s.text(1492,568,'Shared world coordinates',20,MUTED,anchor='middle')
    s.text(30,620,'Monocular input stream',23,INK,600)
    s.text(1770,620,'Sampled frames; time increases left to right',17,MUTED,anchor='end')
    indices=[15,32,42,50,61,74]
    cw,ch=filmstrip(s,indices,654)
    mid=900
    s.line(mid,640,mid,843,AMBER,1.5,'5 5')
    s.rect(mid-70,631,140,26,'white')
    s.text(mid,650,'Shot cut · 177°',17,AMBER,600,anchor='middle')
    s.line(30,847,867,847,BLUE,2)
    s.line(933,847,1770,847,AMBER,2,arrow=True)
    s.text(447,872,'Shot 1',18,BLUE,600,anchor='middle')
    s.text(1354,872,'Shot 2',18,AMBER,600,anchor='middle')
    s.save('Shot3R_teaser_v1_panorama')


def temporal_teaser():
    s=SVG(1800,636,'Shot3R temporal teaser / 时间展开版：各面板使用相同世界视角')
    heading(s)
    s.text(30,117,'Same world, same viewing angle',22,INK,600)
    identity_legend(s,1220,115)
    indices=[15,32,42,50,61,74]
    gap=14;cw=(1740-gap*5)/6
    for j,t in enumerate(indices):
        x=30+j*(cw+gap)
        col=BLUE if t<50 else AMBER
        s.rect(x,139,cw,237,'#f5f8fa',rx=6)
        s.image(ROOT/'assets/egohumans/renders'/f'f{t:03d}_transparent.png',x,146,cw,220)
        s.rect(x,373,cw,3,col)
    s.text(30,408,'Monocular input stream',21,INK,600)
    s.text(1770,408,'Sampled frames; time increases left to right',17,MUTED,anchor='end')
    filmstrip(s,indices,430)
    s.line(900,130,900,618,AMBER,1.5,'6 5')
    s.rect(816,93,168,33,'white',AMBER,1,rx=8)
    s.text(900,115,'Shot cut · 177°',18,AMBER,600,anchor='middle')
    s.save('Shot3R_teaser_v1_temporal')


def multicut_preview():
    s=SVG(1800,625,'Shot3R three-shot candidate / 三镜头诊断备选，非主 teaser')
    heading(s,diagnostic=True)
    s.text(1770,35,'Three-shot candidate',20,MUTED,600,anchor='end')
    s.text(1770,59,'Full-clip IDF1: 0.559',18,MUTED,anchor='end')
    s.text(30,116,'Fixed world view; all valid predictions shown',22,INK,600)
    s.text(1770,116,'Diagnostic preview, not the recommended teaser',18,MUTED,anchor='end')
    indices=[12,49,50,99,100,125]
    gap=14;cw=(1740-gap*5)/6
    for j,t in enumerate(indices):
        x=30+j*(cw+gap)
        s.rect(x,137,cw,240,'#f5f8fa',rx=6)
        s.image(ROOT/'assets/case_01/renders'/f'f{t:03d}_transparent.png',x,145,cw,225)
    s.text(30,409,'Monocular input stream',21,INK,600)
    filmstrip(s,indices,430,case='case_01')
    for j,(start,end,col,label) in enumerate([(30,604,BLUE,'Shot 1'),(617,1184,AMBER,'Shot 2'),(1197,1770,PURPLE,'Shot 3')]):
        s.line(start,604,end,604,col,2)
        s.text((start+end)/2,624,label,17,col,600,anchor='middle')
    for x,angle in [(610,179),(1190,178)]:
        s.line(x,132,x,597,AMBER,1.5,'5 5')
        s.rect(x-47,274,94,45,'white',AMBER,1,rx=6)
        s.text(x,291,'Shot cut',15,AMBER,600,anchor='middle')
        s.text(x,310,f'{angle}°',18,AMBER,600,anchor='middle')
    s.save('Shot3R_three_shot_candidate_NOT_recommended')


if __name__=='__main__':
    main_teaser()
    temporal_teaser()
    multicut_preview()
    (ROOT/'figure_labels_zh_en.json').write_text(json.dumps(TRANSLATIONS,ensure_ascii=False,indent=2)+'\n')
