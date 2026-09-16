# Secondary registration metrics

IDF1 and Coverage use the full fixed case denominator. Local MPJPE/MPVPE verify that a shared rigid transform does not alter within-camera pose or shape. Boundary and seam quantities diagnose cross-shot geometry; fitting time covers registration only and is not end-to-end inference time.

| Dataset | Method | IDF1 | Coverage | Local MPJPE (mm) | Local MPVPE (mm) | Boundary t (m) | Boundary R (deg) | Post root (m) | Seam root (m) | Fit time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| egobody | Human3R per-shot reset | 0.834 (n=129) | 0.981 (n=129) | 70.9 (n=129) | 79.0 (n=129) | 3.374 (n=129) | 55.7 (n=129) | 1.364 (n=129) | 1.332 (n=129) | 0.000 (n=129) |
| egobody | + pelvis translation | 0.896 (n=129) | 0.981 (n=129) | 70.9 (n=129) | 79.0 (n=129) | 2.986 (n=129) | 55.7 (n=129) | 0.852 (n=129) | 0.761 (n=129) | 0.002 (n=129) |
| egobody | + human-joint SE(3) | 0.896 (n=129) | 0.981 (n=129) | 70.9 (n=129) | 79.0 (n=129) | 1.029 (n=129) | 17.9 (n=129) | 0.360 (n=129) | 0.241 (n=129) | 0.078 (n=129) |
| egobody | + scene registration + ICP | 0.834 (n=129) | 0.981 (n=129) | 70.9 (n=129) | 79.0 (n=129) | 2.238 (n=129) | 44.7 (n=129) | 1.324 (n=129) | 1.320 (n=129) | 0.524 (n=129) |
| egobody | Shot3R | 0.985 (n=129) | 0.981 (n=129) | 70.9 (n=129) | 79.0 (n=129) | 0.679 (n=129) | 8.0 (n=129) | 0.364 (n=129) | 0.309 (n=129) | -- |
| egohumans | Human3R per-shot reset | 0.455 (n=90) | 0.616 (n=90) | 108.5 (n=90) | 127.8 (n=90) | 8.901 (n=90) | 104.5 (n=90) | 3.854 (n=80) | 2.653 (n=60) | 0.000 (n=90) |
| egohumans | + pelvis translation | 0.506 (n=90) | 0.616 (n=90) | 108.5 (n=90) | 127.8 (n=90) | 9.905 (n=90) | 104.5 (n=90) | 3.777 (n=80) | 2.312 (n=60) | 0.007 (n=90) |
| egohumans | + human-joint SE(3) | 0.510 (n=90) | 0.616 (n=90) | 108.5 (n=90) | 127.8 (n=90) | 5.676 (n=90) | 48.8 (n=90) | 2.615 (n=80) | 1.072 (n=60) | 0.119 (n=90) |
| egohumans | + scene registration + ICP | 0.455 (n=90) | 0.616 (n=90) | 108.5 (n=90) | 127.8 (n=90) | 8.970 (n=90) | 102.4 (n=90) | 4.253 (n=80) | 3.150 (n=60) | 39.132 (n=90) |
| egohumans | Shot3R | 0.571 (n=90) | 0.616 (n=90) | 108.5 (n=90) | 127.8 (n=90) | 4.135 (n=90) | 23.1 (n=90) | 2.408 (n=80) | 1.227 (n=60) | -- |

A missing fitting-time entry for Shot3R is intentional: the traditional methods report post-processing time, whereas Shot3R is a streaming reconstruction model and its end-to-end runtime is not directly comparable to registration-only time.
