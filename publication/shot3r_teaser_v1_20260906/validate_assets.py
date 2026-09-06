#!/usr/bin/env python3
"""Read-only verification of exported scientific content. 中文：核验素材溯源。"""
import json
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import build_teaser as b


def main():
    report={'source_prediction_checks':{},'figure_checks':{},'source_results_modified':False}
    for name in ['egohumans','case_01','case_02']:
        c=b.get_case(name)
        root=b.ROOT/'assets'/name
        provenance=b.read_json(root/'provenance.json')
        assert b.sha256(c['cache'])==provenance['source_cache_sha256']
        assert b.sha256(c['runtime'])==provenance['source_runtime_sha256']
        with np.load(root/'full_predictions_native_world.npz') as d:
            for key,val in c['arrays'].items():
                assert np.array_equal(d[key],val,equal_nan=True),(name,key)
        with np.load(root/'selected_predictions_native_world.npz') as d:
            for key,val in c['arrays'].items():
                assert np.array_equal(d[key],val[c['selected']],equal_nan=True),(name,key)
        for row in provenance['displayed_frames']:
            rgb=b.ROOT/row['copied_rgb']
            assert b.sha256(rgb)==row['rgb_sha256']==b.sha256(row['source_rgb'])
            t=row['clip_frame_zero_based']
            assert row['valid_persistent_ids']==c['arrays']['persistent_ids'][t][c['arrays']['valid'][t].astype(bool)].tolist()
            im=Image.open(root/'renders'/f'f{t:03d}_transparent.png')
            assert im.mode=='RGBA'
            alpha=np.asarray(im.getchannel('A'))
            assert (alpha==0).any() and (alpha>0).any()
        report['source_prediction_checks'][name]={'all_arrays_equal_to_frozen_source':True,
              'rgb_hashes_match':True,'identity_arrays_unchanged':True,
              'selected_frames':c['selected'],'full_clip_idf1':c['idf1']}
    for name in ['Shot3R_teaser_v1_panorama','Shot3R_teaser_v1_temporal','Shot3R_three_shot_candidate_NOT_recommended']:
        root=b.ROOT/'figures'/name
        tree=ET.parse(root.with_suffix('.svg'))
        text_elements=tree.findall('.//{http://www.w3.org/2000/svg}text')
        viewbox=list(map(float,tree.getroot().attrib['viewBox'].split()))
        for t in text_elements:
            assert float(t.attrib['y'])<=viewbox[3],(name,t.text)
        assert '177' in ''.join(t.text or '' for t in text_elements) or '179' in ''.join(t.text or '' for t in text_elements)
        info=subprocess.check_output(['pdfinfo',str(root.with_suffix('.pdf'))],text=True)
        assert 'Pages:           1' in info
        report['figure_checks'][name]={'svg_parses':True,'text_not_below_canvas':True,
                'pdf_is_one_page':True,'png_size':Image.open(root.with_suffix('.png')).size,
                'pdf_bytes':root.with_suffix('.pdf').stat().st_size}
    movie=b.ROOT/'assets/egohumans/continuous/Shot3R_continuous_real_predictions.mp4'
    video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=width,height,nb_frames,r_frame_rate,duration','-of','json',str(movie)],text=True))
    stream=video['streams'][0]
    assert int(stream['nb_frames'])==60 and stream['r_frame_rate']=='20/1'
    manifest=b.read_json(movie.parent/'manifest.json')
    assert [v['clip_frame_zero_based'] for v in manifest['frames']]==list(range(15,75))
    report['continuous_video']=stream
    b.write_json(b.ROOT/'validation_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    # Direct boundary audit, intentionally showing the missed ID in f49.
    assets=b.ROOT/'assets/egohumans'
    b.contact_sheet([assets/'rgb/f049.jpg',assets/'renders/f049_transparent.png',
                     assets/'rgb/f050.jpg',assets/'renders/f050_transparent.png'],
                    ['f049: last pre-cut RGB','f049: predicted IDs [0, 1]',
                     'f050: first post-cut RGB','f050: predicted IDs [2, 0, 1]'],
                    b.ROOT/'selection/exact_boundary_f049_f050.jpg',columns=4,
                    title='Exact cut audit | no relabeling, repair, or interpolation')


if __name__=='__main__':main()
