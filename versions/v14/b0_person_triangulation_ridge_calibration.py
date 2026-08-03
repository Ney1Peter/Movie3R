"""Evaluator-free P5 residual calibration after frozen BRTC-LC."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Iterable
import numpy as np
from versions.v14.b0_person_triangulation import refine_matched_people,DEFAULT_CONFIG,PersonTriangulationConfig
FEATURES=('raw_m','valid_count','median_gap_m','max_gap_m','median_sine','min_sine','mad_m')
@dataclass(frozen=True)
class RidgeCalibration:
 mean:np.ndarray;scale:np.ndarray;coef:np.ndarray;intercept:float;cap_m:float
 @classmethod
 def load(cls,path:Path)->'RidgeCalibration':
  x=json.loads(Path(path).read_text())
  if tuple(x['features'])!=FEATURES or float(x['alpha'])!=1.:raise ValueError('unexpected P5 model')
  return cls(np.asarray(x['scaler_mean'],float),np.asarray(x['scaler_scale'],float),np.asarray(x['ridge_coef'],float),float(x['ridge_intercept']),float(x['clip_m']))
 def predict(self,evidence:dict[str,Any])->float:
  x=np.asarray([float(evidence[k]) for k in FEATURES]);v=float(((x-self.mean)/self.scale)@self.coef+self.intercept);return float(np.clip(v,-self.cap_m,self.cap_m))
def refine_matched_people_ridge_calibration(pre_camera:Any,post_camera:Any,pre_people:list[dict],post_people:list[dict],matches:Iterable[tuple[int,int]],calibration:RidgeCalibration,config:PersonTriangulationConfig=DEFAULT_CONFIG):
 corrected,debug=refine_matched_people(pre_camera,post_camera,pre_people,post_people,matches,config);center=np.asarray(post_camera,float)[:3,3]
 for row in debug['people']:
  j=int(row['post_index'])
  if not row['accepted']:
   row['p5']={'applied':False,'reason':'brtc_rejected_exact_fallback','residual_m':0.};continue
  ray=np.asarray(post_people[j]['root'],float)-center;ray/=max(float(np.linalg.norm(ray)),1e-12);residual=calibration.predict(row['evidence'])
  for key in ('root','joints','vertices'):
   if key in corrected[j]:corrected[j][key]=np.asarray(corrected[j][key],float)+residual*ray
  row['p5']={'applied':True,'reason':'applied','residual_m':residual,'ray_world':ray}
 debug.update({'camera_update':'none','p5_update':'bounded ridge residual along raw post root ray','p5_features':FEATURES,'p5_applied_count':sum(bool(x['p5']['applied']) for x in debug['people']),'unmatched_policy':'exact B0','rejected_policy':'exact B0'})
 return corrected,debug
