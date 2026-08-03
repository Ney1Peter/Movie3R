#!/usr/bin/env python3
"""Build current-P0 dance/box offset0 caches for frozen P5 confirmation."""
from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
REPO_ROOT=Path(__file__).resolve().parents[2]
for p in (REPO_ROOT,REPO_ROOT/'src',REPO_ROOT/'scripts'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from dust3r.model import ARCroco3DStereo
from dust3r.utils.smpl_layer import SMPL_Layer
from versions.v13 import gt_id_consensus as gt
from versions.v14.probe_p1_foot_scene_observability import DEFAULT_CHECKPOINT,cache_case,configure_model,jsonable,sha256

DEFAULT_MANIFEST=REPO_ROOT/'config/manifests/v14_p5_brtc_ray_residual_confirm_20260803.json'
DEFAULT_OUT=REPO_ROOT/'output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation_cache'
DATA=Path('/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted')
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);p.add_argument('--model-path',type=Path,default=DEFAULT_CHECKPOINT);p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);p.add_argument('--device',default='cuda:0');p.add_argument('--max-cases',type=int,default=0);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def safe(path):
 if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):raise ValueError(f'outside workspace: {path}')
def labels_after_runtime(row,record,gtargs):
 r=row['runtime'];frame,pre,post=(int(record[k]) for k in ('frame','pre_camera','post_camera'))
 assigned,_=gt.assign_gt_identities(gtargs,r['pre_people'],r['pre_camera_c2w'],pre,frame,512,512)
 prelabels={int(person['detection_index']):str(identity) for identity,person in assigned.items()}
 postlabels={int(x['detection_index']):str(x['identity']) for x in row['evaluator']['assignment_evaluator_only']['assignments']}
 row['evaluator']['pre_labels_by_detection']=prelabels;row['evaluator']['post_labels_by_detection']=postlabels
 return row
def main():
 a=parse()
 for x in (a.manifest,a.model_path,a.output_dir):safe(x)
 payload=json.loads(a.manifest.read_text());records=payload['confirm'][:a.max_cases or None];cases=a.output_dir/'cases';cases.mkdir(parents=True,exist_ok=True)
 dev=torch.device(a.device);model=ARCroco3DStereo.from_pretrained(str(a.model_path)).to(dev);flags=configure_model(model);layer=SMPL_Layer(type='smplx',gender='neutral',num_betas=10,kid=False,person_center='head').to(dev).eval();paths=[];fails=[]
 try:
  for n,record in enumerate(records,1):
   path=cases/f"{record['event_id']}.pt"
   if path.is_file() and not a.overwrite:row=torch.load(path,map_location='cpu',weights_only=False)
   else:
    try:
     seq=str(record['sequence']);gt.IDENTITIES=gt.SEQUENCE_IDENTITIES[seq];gta=SimpleNamespace(data_root=a.data_root,sequence=seq,output_dir=a.output_dir/'cache'/seq/'frame_cache',size=512);row=cache_case(model,layer,record,gta,dev,512);row=labels_after_runtime(row,record,gta)
    except Exception as e:row={'status':'failed','record':record,'error':repr(e),'traceback':traceback.format_exc()}
    torch.save(row,path)
   if row['status']=='ok':paths.append(path);print(f'[{n:02d}/{len(records):02d}] {record["event_id"]} people={len(row["runtime"]["raw_post_people"])}',flush=True)
   else:fails.append({'event_id':record['event_id'],'error':row['error']});print(f'[{n:02d}/{len(records):02d}] FAILED {record["event_id"]}: {row["error"]}',flush=True)
   if dev.type=='cuda':torch.cuda.empty_cache()
 finally:
  del layer,model
  if dev.type=='cuda':torch.cuda.empty_cache()
 index={'schema':'p5_confirmation_cache_v1','checkpoint':str(a.model_path),'checkpoint_sha256':sha256(a.model_path),'manifest':str(a.manifest),'manifest_sha256':sha256(a.manifest),'flags':flags,'case_paths':[str(p) for p in paths],'failures':fails,'runtime_before_gt':True}
 out=a.output_dir/'P5_CONFIRM_CACHE_INDEX.json';out.write_text(json.dumps(jsonable(index),indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
