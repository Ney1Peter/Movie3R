#!/usr/bin/env python3
"""Package only this task's generated artifacts. 中文：打包本次交付，不包含权重。"""
from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery'


def package(name,paths):
    OUT.mkdir(exist_ok=True)
    target=OUT/name
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(set(paths)):
            if p.is_file():
                z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    return {'file':str(target.relative_to(ROOT)),'bytes':target.stat().st_size}


def main():
    notes=[ROOT/'README.md',ROOT/'CAPTIONS_ZH_EN.md',ROOT/'figure_labels_zh_en.json',ROOT/'validation_report.json']
    figures=list((ROOT/'figures').glob('*'))
    # Lightweight handoff: all layouts, selected RGB/transparent renders, notes.
    selected=notes+figures
    for name in ['egohumans','case_01','case_02']:
        for folder in ['rgb','renders']:
            selected+=list((ROOT/'assets'/name/folder).glob('*'))
        selected.append(ROOT/'assets'/name/'provenance.json')
    selected+=list((ROOT/'selection').glob('*'))
    selected.append(ROOT/'assets/egohumans/continuous/Shot3R_continuous_real_predictions.mp4')
    report=[package('Shot3R_teaser_visual_assets.zip',selected)]
    complete=notes+figures+list(ROOT.glob('*.py'))
    for folder in ['assets','selection']:
        complete+=list((ROOT/folder).rglob('*'))
    report.append(package('Shot3R_teaser_full_bundle.zip',complete))
    (OUT/'package_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
