# Shot3R abstract, TL;DR, and keywords

Confirmed on 2026-09-18 for the current v072 manuscript.

## Abstract

Streaming multi-person 4D reconstruction from multi-shot monocular video requires camera poses, human motion, and scene geometry to remain consistent in a common world frame despite abrupt viewpoint changes. Existing recurrent methods typically rely on continuity between adjacent observations and can therefore incur substantial reconstruction errors at shot transitions. To address this problem, we present Shot3R, a streaming framework that decouples cross-shot relation estimation from within-shot recurrent reconstruction. At each shot transition, a learned history-conditioned alignment module integrates the current observation with historical context in latent space to infer the inter-shot spatial relation. Geometry-aware identity association guides further refinement of this relation, yielding a shared alignment of cameras and humans. Shot3R retains the spatial reference established by previous shots while isolating the new shot's recurrent dynamics from incompatible historical states, supporting both cross-shot geometric consistency and within-shot reconstruction stability. The framework enables online reconstruction without accessing future frames or performing sequence-level optimization. Experiments on three multi-person video datasets demonstrate improvements in world-frame human reconstruction, camera trajectory estimation, and cross-shot identity consistency, with greater robustness to large viewpoint changes.

## TL;DR

Shot3R enables streaming multi-person 4D reconstruction across abrupt shot transitions in a common world frame by decoupling cross-shot relation estimation from within-shot recurrent reconstruction.

## Keywords

streaming 4D reconstruction, multi-shot monocular video, multi-person reconstruction, human–scene reconstruction, cross-shot alignment, recurrent state propagation
