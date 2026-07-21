# V54 Synthetic Explicit Shot Bridge

## Design

- Train on explicit Human3R pointmaps and SMPL-X anchors, never raw tokens.
- Create known small/medium/large SE(3) or Sim(3) perturbations on continuous frames.
- Compare raw scale, learned Sim(3), and DA3-normalized geometry.
- Evaluate real cuts with four-fold leave-one-source-out.
- Final factorized version keeps V16 torso rotation and learns point correspondence for explicit translation solving.

## Main Result

- Fixed: 1.715 m, 24.20 deg.
- Learned factorized: 1.186 m, 16.04 deg.
- V53 reference: 0.397 m, 12.09 deg.
- Factorized learning improves AvatarReX and both MVHuman sources, but THuman translation degrades from 0.483 m to 2.210 m.

## Decision

The HumanMM-style synthetic perturbation idea is useful for generating supervision, but simple transformed continuous-frame pointclouds do not reproduce real camera-cut reconstruction mismatch. Full learned SE(3) is rejected. The factorized learned correspondence branch has real signal but is not source-safe and is currently inferior to V53. Keep V53 as the main method; revisit training only with substantially more realistic synchronized multi-camera overlap/correspondence data.
