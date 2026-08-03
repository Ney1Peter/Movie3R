#!/usr/bin/env python3
"""Freeze dance/box offset0 first-post rows for P5 confirmation."""
from __future__ import annotations
import json
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[2]
OUT=REPO_ROOT/'config/manifests/v14_p5_brtc_ray_residual_confirm_20260803.json'
def main():
 records=[]
 for seq in ('dance','box'):
  report=REPO_ROOT/f'output/v14/b0_identity_matching_extended/{seq}/v14_b0_identity_matching.json'
  for row in json.loads(report.read_text())['cases']:
   c=row['case']
   if int(c['offset'])==0:records.append({'event_id':f'confirm_{c["key"]}','sequence':seq,'pre_camera':int(c['source_camera']),'post_camera':int(c['target_camera']),'frame':int(c['timestamp'])})
 payload={'schema_version':1,'created':'2026-08-03','purpose':'P5 frozen replay confirmation; current P0 forward, dance/box first-post offset0 only','source_reports':['output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json','output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json'],'confirm':records}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,indent=2)+'\n');print(OUT,len(records),flush=True)
if __name__=='__main__':main()
