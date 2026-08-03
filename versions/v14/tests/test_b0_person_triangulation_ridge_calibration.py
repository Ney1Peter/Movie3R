import numpy as np
from versions.v14.b0_person_triangulation_ridge_calibration import RidgeCalibration,refine_matched_people_ridge_calibration
def person(root):
 root=np.asarray(root,float);j=np.stack([root+[0,0,0],root+[.1,0,0],root+[0,.1,0],root+[0,0,.1],root+[.1,.1,0],root+[.1,0,.1],root+[0,.1,.1],root+[.1,.1,.1],root+[.2,0,0],root+[0,.2,0],root+[0,0,.2],root+[.2,.2,0],root+[.2,0,.2],root+[0,.2,.2],root+[.2,.2,.2],root+[.3,0,0],root+[0,.3,0],root+[.3,.3,.3]])
 return {'root':root,'joints':j,'vertices':j.copy()}
def test_camera_and_rejected_fallback_are_exact():
 c=np.eye(4);pre=[person([0,0,2])];post=[person([.1,0,2])];cal=RidgeCalibration(np.zeros(7),np.ones(7),np.zeros(7),.1,.3);out,debug=refine_matched_people_ridge_calibration(c,c,pre,post,[(0,0)],cal)
 assert debug['camera_update']=='none'
 assert np.array_equal(out[0]['root'],post[0]['root'])
 assert debug['people'][0]['p5']['applied'] is False
