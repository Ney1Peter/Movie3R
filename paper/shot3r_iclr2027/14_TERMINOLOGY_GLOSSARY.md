# BRIDGE3R Terminology Lock

| Preferred term | Definition | Avoid |
|---|---|---|
| same-scene viewpoint cut | abrupt switch between synchronized or temporally adjacent cameras observing the same physical scene and principal people | arbitrary editing, cross-scene cut |
| read-only learned gauge pathway | boundary evaluation that may use emitted history but whose recurrent state and shadow outputs are discarded | history owner, geometry parent |
| clean-reset recurrent state | independently evaluated reset state and the only state propagated after an event | state owner, runtime parent |
| coarse gauge | camera composition formed from read-only and reset predictions | independently regressed scene transform |
| prediction-only association | Hungarian matching from predicted pelvis, torso orientation, and root-centred joints | oracle identity matching |
| shared translation | one boundary translation applied to post-cut cameras, joints, and vertices | dense-scene correction |
| Coverage | accepted prediction--target instances divided by the complete protocol denominator | accuracy, recall without definition |
| pair accuracy | correct association among evaluator-valid endpoint pairs | full recovery |
| continuation recovery | correct associations divided by all GT cross-cut continuations | conditional precision |
| ATE-Sim3 / ATE-SE3 | camera trajectory error after similarity / rigid alignment | unqualified ATE |
| oracle timing | evaluator-boundary diagnostic input | deployable baseline |
| same-checkpoint sensitivity | inference mask applied to one trained checkpoint | retraining ablation |
