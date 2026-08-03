#!/usr/bin/env python3
"""P5: timestamp-grouped Ridge calibration of a frozen BRTC ray residual."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from typing import Any
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT=Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT,REPO_ROOT/'src',REPO_ROOT/'scripts'):
    if str(_p) not in sys.path: sys.path.insert(0,str(_p))
DEFAULT_P1=REPO_ROOT/'output/v14/fine_alignment_research/p1_foot_scene_observability_v2'
DEFAULT_P2=REPO_ROOT/'output/v14/fine_alignment_research/p2_native_token_who'
DEFAULT_OUT=REPO_ROOT/'output/v14/fine_alignment_research/p5_brtc_ray_residual_calibration'
FEATURES=('raw_m','valid_count','median_gap_m','max_gap_m','median_sine','min_sine','mad_m')
ALPHA,CAP=1.0,.30

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--p1-cache-dir',type=Path,default=DEFAULT_P1);p.add_argument('--p2-cache-dir',type=Path,default=DEFAULT_P2);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);return p.parse_args()
def safe(path:Path):
 if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):raise ValueError(f'outside workspace: {path}')
def digest(path:Path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(np.asarray(a,dtype=np.float64)).tobytes()).hexdigest()
def ready(x):
 if isinstance(x,Path):return str(x)
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):ready(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [ready(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def errors(person,target):
 j,v=min(len(person['joints']),len(target['joints_world'])),min(len(person['vertices']),len(target['vertices_world']))
 return {'root_m':float(np.linalg.norm(person['root']-target['root_world'])),'joint_m':float(np.linalg.norm(person['joints'][:j]-target['joints_world'][:j],axis=1).mean()),'vertex_m':float(np.linalg.norm(person['vertices'][:v]-target['vertices_world'][:v],axis=1).mean())}
def shift(person,ray,value):return {k:np.asarray(person[k],dtype=np.float64)+value*ray for k in ('root','joints','vertices')}
def summary(rows,method):
 vals=[r['metrics_evaluator_only'][method] for r in rows if r.get('metrics_evaluator_only')]
 return {'count':len(vals),**{k:float(np.mean([x[k] for x in vals])) if vals else float('nan') for k in ('root_m','joint_m','vertex_m')}}
def harm(rows,method,key):
 d=[r['metrics_evaluator_only'][method][key]-r['metrics_evaluator_only']['brtc'][key] for r in rows if r.get('metrics_evaluator_only')]
 return float(np.mean(np.asarray(d)>.05)) if d else float('nan')

def run(a):
 for p in (a.p1_cache_dir,a.p2_cache_dir,a.output_dir):safe(p)
 pi,qi=a.p1_cache_dir/'P1_CACHE_INDEX.json',a.p2_cache_dir/'P2_CACHE_INDEX.json';pindex=json.loads(pi.read_text());qindex=json.loads(qi.read_text())
 if pindex.get('schema')!=2:raise RuntimeError('requires P1 schema 2')
 qpaths={Path(x).stem:Path(x) for x in qindex['case_paths']};rows=[];hashes={}
 # Entire runtime feature/action payload is built before either P1 or P2 evaluator branch is opened.
 for path in pindex['case_paths']:
  c=torch.load(path,map_location='cpu',weights_only=False);r=c['runtime'];event=str(r['record']['event_id']);h=str(r['b0_camera_sha256'])
  if r['runtime_contract']['gt_used'] or r['runtime_contract']['future_post_frames_used'] or arrhash(r['b0_camera_c2w'])!=h:raise RuntimeError(f'invalid runtime {event}')
  hashes[event]=h;center=np.asarray(r['b0_camera_c2w'])[:3,3];accepted={int(z['post_index']):z for z in r['brtc']['people']};runtime=[]
  for i,j in r['association']['pairs']:
   i,j=int(i),int(j);post=r['b0_post_people'][j];brtc=r['brtc_post_people'][j];e=accepted[j]['evidence'];ray=np.asarray(post['root'])-center;ray=ray/max(float(np.linalg.norm(ray)),1e-12)
   runtime.append({'event_id':event,'timestamp':int(r['record']['frame']),'pre_index':i,'post_index':j,'pre_detection':int(r['pre_people'][i]['detection_index']),'post_detection':int(post['detection_index']),'accepted':bool(accepted[j]['accepted']),'feature':[float(e[k]) for k in FEATURES],'ray':ray,'brtc':{k:np.asarray(brtc[k],dtype=np.float64) for k in ('root','joints','vertices')}})
  if arrhash(r['b0_camera_c2w'])!=h:raise RuntimeError(f'camera mutation {event}')
  q=torch.load(qpaths[event],map_location='cpu',weights_only=False)
  if q['runtime']['b0_camera_sha256']!=h:raise RuntimeError(f'P1/P2 camera mismatch {event}')
  prelab,postlab=q['evaluator']['pre_labels_by_detection'],q['evaluator']['post_labels_by_detection'];targets=c['evaluator']['target_by_detection']
  for z in runtime:
   target=targets.get(z['post_detection']);z['correct_evaluator_only']=bool(prelab.get(z['pre_detection']) is not None and prelab.get(z['pre_detection'])==postlab.get(z['post_detection']));z['target_evaluator_only']=target
   if z['accepted'] and z['correct_evaluator_only'] and target is not None:z['label_evaluator_only']=float(np.dot(np.asarray(target['root_world'])-z['brtc']['root'],z['ray']))
   rows.append(z)
 train=[i for i,z in enumerate(rows) if 'label_evaluator_only' in z];groups=sorted({rows[i]['timestamp'] for i in train})
 if len(groups)<2:raise RuntimeError('need timestamp groups')
 pred=np.zeros(len(rows));folds=[]
 for g in groups:
  tr=[i for i in train if rows[i]['timestamp']!=g];te=[i for i,z in enumerate(rows) if z['timestamp']==g and z['accepted']]
  model=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA));model.fit(np.asarray([rows[i]['feature'] for i in tr]),np.asarray([rows[i]['label_evaluator_only'] for i in tr]));pred[te]=np.clip(model.predict(np.asarray([rows[i]['feature'] for i in te])),-CAP,CAP);folds.append({'held_timestamp':g,'train_correct_rows':len(tr),'test_accepted_rows':len(te)})
 for i,z in enumerate(rows):
  if z['accepted']:z['prediction_m']=float(pred[i]);candidate=shift(z['brtc'],z['ray'],pred[i])
  else:z['prediction_m']=0.;candidate=z['brtc']
  target=z['target_evaluator_only'];
  if target is not None:z['metrics_evaluator_only']={'brtc':errors(z['brtc'],target),'p5_ridge':errors(candidate,target)}
  z.pop('brtc');z.pop('ray');z.pop('target_evaluator_only')
 allrows=[z for z in rows if z.get('metrics_evaluator_only')];correct=[z for z in allrows if z['accepted'] and z['correct_evaluator_only']]
 X=np.asarray([rows[i]['feature'] for i in train]);y=np.asarray([rows[i]['label_evaluator_only'] for i in train]);final=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA));final.fit(X,y);scaler,ridge=final.named_steps['standardscaler'],final.named_steps['ridge']
 sums={'all_geometry_pairs':{m:summary(allrows,m) for m in ('brtc','p5_ridge')},'correct_brtc_accepted':{m:summary(correct,m) for m in ('brtc','p5_ridge')}};co=sums['correct_brtc_accepted'];ao=sums['all_geometry_pairs']
 gate={'correct_training_rows_at_least_60':len(train)>=60,'correct_root_gain_at_least_5mm':co['p5_ridge']['root_m']<=co['brtc']['root_m']-.005,'correct_joint_gain_at_least_5mm':co['p5_ridge']['joint_m']<=co['brtc']['joint_m']-.005,'correct_vertex_gain_at_least_5mm':co['p5_ridge']['vertex_m']<=co['brtc']['vertex_m']-.005,'correct_harm_le_10pct':all(harm(correct,'p5_ridge',k)<=.10 for k in ('root_m','joint_m','vertex_m')),'all_root_gain_at_least_5mm':ao['p5_ridge']['root_m']<=ao['brtc']['root_m']-.005,'all_root_harm_le_10pct':harm(allrows,'p5_ridge','root_m')<=.10,'camera_bit_exact':True,'runtime_features_before_evaluator':True}
 model={'features':FEATURES,'alpha':ALPHA,'clip_m':CAP,'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'ridge_coef':ridge.coef_,'ridge_intercept':ridge.intercept_,'training_correct_rows':len(train),'training_timestamps':groups}
 report={'experiment':'v14_p5_brtc_ray_residual_calibration','status':'GO_TO_FROZEN_CONFIRMATION' if all(gate.values()) else 'NO_GO_BRTC_RAY_RESIDUAL_CALIBRATION','p1_cache_index':str(pi),'p1_cache_index_sha256':digest(pi),'p2_cache_index':str(qi),'p2_cache_index_sha256':digest(qi),'protocol':{'features':FEATURES,'ridge_alpha':ALPHA,'cap_m':CAP,'cv':'leave-one-timestamp-out'},'folds':folds,'summaries':sums,'harm_vs_brtc_evaluator_only':{'all_geometry_pairs':{k:harm(allrows,'p5_ridge',k) for k in ('root_m','joint_m','vertex_m')},'correct_brtc_accepted':{k:harm(correct,'p5_ridge',k) for k in ('root_m','joint_m','vertex_m')}},'runtime_invariants':{'camera_sha256_by_event':hashes,'camera_max_abs_change':0.,'all_actions_before_gt':True,'future_post_frames_used':0,'external_pretrained_models':[]},'gate':gate,'frozen_model_candidate':model,'rows':rows}
 a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'P5_BRTC_RAY_RESIDUAL_CALIBRATION_REPORT.json';out.write_text(json.dumps(ready(report),indent=2)+'\n');
 if all(gate.values()):(a.output_dir/'P5_FROZEN_MODEL_BEFORE_CONFIRM.json').write_text(json.dumps(ready(model),indent=2)+'\n')
 return out
if __name__=='__main__':print(run(args()),flush=True)
