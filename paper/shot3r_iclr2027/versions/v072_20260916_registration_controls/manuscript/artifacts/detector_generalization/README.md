# Detector generalization audit

This artifact compares retained RGB-only detector traces without changing a
threshold or rerunning reconstruction. EgoBody has 129/129
exact first triggers and no off-boundary positives. On the held-out weak-texture
MVHuman stress set, only 10/50 first triggers
are exact; 40 are early, with a median offset of
-48.0 frames. The raw trace still
responds at every annotated boundary, but also produces 399
off-boundary positives (53.56
per 1000 scored transitions). The distinction between first-trigger timing and
raw boundary recall is therefore essential.
