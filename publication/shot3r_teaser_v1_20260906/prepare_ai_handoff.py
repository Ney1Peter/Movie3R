#!/usr/bin/env python3
"""Package existing visuals for an external drawing AI; no new drawing.
中文：只复制、整理和裁切已有论文参考图，不生成新 teaser。
"""
from pathlib import Path
import hashlib
import html
import json
import shutil
import zipfile
import markdown
from PIL import Image

SOURCE=Path(__file__).resolve().parent
TARGET=SOURCE.parent/'Shot3R_Teaser_AI_Handoff_20260906'
WORK=SOURCE.parents[2]
COPIES=[]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(source,relative):
    dest=TARGET/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
    assert digest(source)==digest(dest)
    COPIES.append(dict(source=str(source),file=str(relative),bytes=dest.stat().st_size,sha256=digest(dest)))


def main():
    TARGET.mkdir(exist_ok=True)
    source_ref=Path('/tmp/shot3r_human3r_fig1_reference.png')
    copy(source_ref,Path('01_风格参考/Human3R_论文首页.png'))
    im=Image.open(source_ref)
    # Crop the existing Figure 1 artwork; no repainting, retouching, or generation.
    bounds=(round(im.width*.169),round(im.height*.196),round(im.width*.831),round(im.height*.338))
    crop=TARGET/'01_风格参考/Human3R_Figure1_裁切.png'
    im.crop(bounds).save(crop)
    records=TARGET/'07_素材溯源与检查'
    records.mkdir(exist_ok=True)
    (records/'参考图来源.json').write_text(json.dumps(dict(
        paper=str(WORK/'paper/Human3R-ori.pdf'),page=1,source_render=str(source_ref),
        source_size=im.size,crop_pixels=bounds,operation='crop of existing paper figure; no generated visual'),indent=2,ensure_ascii=False)+'\n')
    names={'egohumans':'EgoHumans','case_01':'Harmony4D_case01','case_02':'Harmony4D_case02'}
    for original,label in names.items():
        for p in sorted((SOURCE/'assets'/original/'rgb').glob('*.jpg')):
            copy(p,Path('02_原始视频帧')/label/p.name)
        for p in sorted((SOURCE/'assets'/original/'renders').glob('*.png')):
            copy(p,Path('03_重建效果参考')/label/p.name)
        copy(SOURCE/'assets'/original/'provenance.json',Path('07_素材溯源与检查')/f'{label}_provenance.json')
        copy(SOURCE/'selection'/f'{original}_mesh_contact_sheet.jpg',Path('03_重建效果参考')/f'{label}_重建总览.jpg')
    copy(SOURCE/'selection/candidate_rgb_contact_sheet.jpg',Path('02_原始视频帧/候选RGB总览.jpg'))
    copy(SOURCE/'assets/egohumans/continuous/Shot3R_continuous_real_predictions.mp4',Path('02_原始视频帧/EgoHumans/连续RGB与重建参考.mp4'))
    for stem,label in [('Shot3R_teaser_v1_panorama','全景版'),('Shot3R_teaser_v1_temporal','时间展开版'),
                       ('Shot3R_three_shot_candidate_NOT_recommended','三镜头诊断草稿_非推荐成品')]:
        for p in sorted((SOURCE/'figures').glob(stem+'*')):
            copy(p,Path('04_已有图稿_仅供内容参考')/label/p.name)
    for p in sorted((SOURCE/'editable').glob('*.pptx')):
        copy(p,Path('05_可编辑文件')/p.name)
    for ext in ['svg','png','pdf']:
        copy(SOURCE/'editable'/f'Shot3R_concept_template.{ext}',Path('05_可编辑文件')/f'Shot3R_concept_template.{ext}')
    copy(SOURCE/'editable/生成提示词_三镜头三人_中英文.md',Path('06_提示词与图注/生成提示词_三镜头三人_中英文.md'))
    copy(SOURCE/'CAPTIONS_ZH_EN.md',Path('07_素材溯源与检查/旧真实结果图注_不是新概念图图注.md'))
    copy(SOURCE/'validation_report.json',Path('07_素材溯源与检查/原素材校验报告.json'))
    copy(SOURCE/'selection/candidate_metrics.json',Path('07_素材溯源与检查/旧真实候选指标_不用于概念图.json'))
    copy(SOURCE/'selection/exact_boundary_f049_f050.jpg',Path('07_素材溯源与检查/真实相邻边界帧_含漏检.jpg'))
    (records/'素材复制清单.json').write_text(json.dumps(COPIES,indent=2,ensure_ascii=False)+'\n')
    brief=TARGET/'00_交接说明_先读这个.md'
    body=markdown.markdown(brief.read_text(),extensions=['tables','fenced_code'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shot3R 绘图交接</title>'
    page+='<style>body{max-width:1050px;margin:40px auto;padding:0 25px;color:#25313b;font:17px/1.8 sans-serif}img{max-width:100%;height:auto;border:1px solid #ddd}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:9px;text-align:left}pre{white-space:pre-wrap;background:#f5f7f8;padding:20px}a{color:#245aad}h1,h2,h3{line-height:1.4}</style><body>'
    (TARGET/'00_交接说明_浏览器打开.html').write_text(page+body+'</body></html>')
    portable=[]
    for p in TARGET.rglob('*'):
        if p.is_file():
            assert not p.is_symlink()
            portable.append(p)
    archive=TARGET.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(portable):z.write(p,str(Path(TARGET.name)/p.relative_to(TARGET)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(folder=str(TARGET),zip=str(archive),files=len(portable),
            copied_assets=len(COPIES),zip_bytes=archive.stat().st_size,generated_new_teaser=False),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
