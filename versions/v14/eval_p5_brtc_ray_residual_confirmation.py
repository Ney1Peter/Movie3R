#!/usr/bin/env python3
"""Apply the frozen P5 Ridge once to current-P0 dance/box confirmation caches."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np,torch
REPO_ROOT=Path(__file__).resolve().parents[2]
for p in (REPO_ROOT,REPO_ROOT/'src',REPO_ROOT/'scripts'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from versions.v14.probe_p5_brtc_ray_residual_calibration import FEATURES,arrhash,digest,errors,ready,shift
DEFAULT_CACHE=REPO_ROOT/'output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation_cache/P5_CONFIRM_CACHE_INDEX.json'
DEFAULT_MODEL=REPO_ROOT/'output/v14/fine_alignment_research/p5_brtc_ray_residual_calibration/P5_FROZEN_MODEL_BEFORE_CONFIRM.json'
DEFAULT_OUT=REPO_ROOT/'output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation'
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-index',type=Path,default=DEFAULT_CACHE);p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);return p.parse_args()
def safe(p):
 if not str(p.resolve()).startswith(str(REPO_ROOT.resolve())):raise ValueError(f'outside workspace: {p}')
def summarize(rows,m):
 x=[r['metrics_evaluator_only'][m] for r in rows if 'metrics_evaluator_only'in r];return {'count':len(x),**{k:float(np.mean([z[k] for z in x])) if x else float('nan') for k in ('root_m','joint_m','vertex_m')}}
def harm(rows,m,k):
 x=[r['metrics_evaluator_only'][m][k]-r['metrics_evaluator_only']['brtc'][k] for r in rows if 'metrics_evaluator_only'in r];return float(np.mean(np.asarray(x)>.05)) if x else float('nan')
def main():
 a=parse()
 for p in (a.cache_index,a.model,a.output_dir):safe(p)
 index=json.loads(a.cache_index.read_text());model=json.loads(a.model.read_text())
 if tuple(model['features'])!=FEATURES or float(model['alpha'])!=1.:raise RuntimeError('not the frozen P5 model')
 mu,scale,coef,inter=np.asarray(model['scaler_mean']),np.asarray(model['scaler_scale']),np.asarray(model['ridge_coef']),float(model['ridge_intercept']);cap=float(model['clip_m']);rows=[];hashes={};fallback={}
 for path in index['case_paths']:
  c=torch.load(path,map_location='cpu',weights_only=False);r=c['runtime'];event=str(r['record']['event_id']);h=str(r['b0_camera_sha256'])
  if r['runtime_contract']['gt_used'] or r['runtime_contract']['future_post_frames_used'] or arrhash(r['b0_camera_c2w'])!=h:raise RuntimeError(f'invalid runtime {event}')
  hashes[event]=h;center=np.asarray(r['b0_camera_c2w'])[:3,3];brtcby={int(z['post_index']):z for z in r['brtc']['people']};runtime=[]
  for i,j in r['association']['pairs']:
   i,j=int(i),int(j);post=r['b0_post_people'][j];base=r['brtc_post_people'][j];br=brtcby[j];ray=np.asarray(post['root'])-center;ray/=max(float(np.linalg.norm(ray)),1e-12);feat=np.asarray([float(br['evidence'][k]) for k in FEATURES]);accepted=bool(br['accepted']);pred=float(np.clip(((feat-mu)/scale@coef)+inter,-cap,cap)) if accepted else 0.;candidate=shift(base,ray,pred) if accepted else {k:np.asarray(base[k],dtype=np.float64) for k in ('root','joints','vertices')};reason='applied' if accepted else 'brtc_rejected_exact_fallback';fallback[reason]=fallback.get(reason,0)+1
   runtime.append({'event_id':event,'sequence':r['record']['sequence'],'pre_detection':int(r['pre_people'][i]['detection_index']),'post_detection':int(post['detection_index']),'accepted':accepted,'prediction_m':pred,'runtime_reason':reason,'brtc':{k:np.asarray(base[k],dtype=np.float64) for k in ('root','joints','vertices')},'p5':candidate})
  if arrhash(r['b0_camera_c2w'])!=h:raise RuntimeError(f'camera mutated {event}')
  ev=c['evaluator'];prelab,postlab=ev['pre_labels_by_detection'],ev['post_labels_by_detection']
  for z in runtime:
   target=ev['target_by_detection'].get(z['post_detection']);z['correct_evaluator_only']=bool(prelab.get(z['pre_detection']) is not None and prelab.get(z['pre_detection'])==postlab.get(z['post_detection']))
   if target is not None:z['metrics_evaluator_only']={'brtc':errors(z['brtc'],target),'p5_ridge':errors(z['p5'],target)}
   z.pop('brtc');z.pop('p5');rows.append(z)
 allrows=[z for z in rows if 'metrics_evaluator_only'in z];correct=[z for z in allrows if z['accepted'] and z['correct_evaluator_only']];sums={'all_geometry_pairs':{m:summarize(allrows,m) for m in ('brtc','p5_ridge')},'correct_brtc_accepted':{m:summarize(correct,m) for m in ('brtc','p5_ridge')}};co,ao=sums['correct_brtc_accepted'],sums['all_geometry_pairs'];gate={'correct_rows_at_least_24':len(correct)>=24,'correct_root_gain_at_least_5mm':co['p5_ridge']['root_m']<=co['brtc']['root_m']-.005,'correct_joint_gain_at_least_5mm':co['p5_ridge']['joint_m']<=co['brtc']['joint_m']-.005,'correct_vertex_gain_at_least_5mm':co['p5_ridge']['vertex_m']<=co['brtc']['vertex_m']-.005,'all_root_gain_at_least_5mm':ao['p5_ridge']['root_m']<=ao['brtc']['root_m']-.005,'all_root_harm_le_10pct':harm(allrows,'p5_ridge','root_m')<=.10,'camera_bit_exact':True,'rejected_exact_fallback':True}
 out={'experiment':'v14_p5_brtc_ray_residual_calibration_confirmation','status':'QUALIFIED_P5_BRTC_RAY_RESIDUAL_CALIBRATION' if all(gate.values()) else 'NO_GO_BRTC_RAY_RESIDUAL_CALIBRATION_CONFIRMATION','cache_index':str(a.cache_index),'cache_index_sha256':digest(a.cache_index),'frozen_model':str(a.model),'frozen_model_sha256':digest(a.model),'summaries':sums,'harm_vs_brtc_evaluator_only':{'all':{k:harm(allrows,'p5_ridge',k) for k in ('root_m','joint_m','vertex_m')},'correct':{k:harm(correct,'p5_ridge',k) for k in ('root_m','joint_m','vertex_m')}},'counts':{'rows_with_target':len(allrows),'correct_brtc_accepted':len(correct),'fallback':fallback},'runtime_invariants':{'camera_sha256_by_event':hashes,'camera_max_abs_change':0.,'future_post_frames_used':0,'all_actions_before_gt':True,'external_pretrained_models':[]},'gate':gate,'rows':rows};a.output_dir.mkdir(parents=True,exist_ok=True);p=a.output_dir/'P5_CONFIRMATION_REPORT.json';p.write_text(json.dumps(ready(out),indent=2)+'\n');print(p,flush=True)
if __name__=='__main__':main()
