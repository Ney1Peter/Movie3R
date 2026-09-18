# v029 Inconsistency List

1. Abstract uses stronger causal and viewpoint wording than the reported uncertainty supports.
2. Method alternates between “state ownership”, “owns”, and precise propagation semantics; the final paper must prefer the latter.
3. Temporal correction token is described but not explicitly formulated.
4. `gamma` is not clearly identified as an internal learned fusion coefficient rather than a supervised gate.
5. The transformation scope for camera, human joints/vertices, scene/point map and recurrent state is not tabulated.
6. The mechanism table presents oracle and causal rows as if they were a single additive ablation chain.
7. The mechanism table omits IDF1 and Coverage even though association is one of its columns.
8. Detector success on EgoBody and early firing on weak-texture MVHuman are separated across main/supplement, obscuring domain sensitivity.
9. Association accuracy uses conditional denominators that are not visible in the main paper.
10. `pre-registered` appears without a public preregistration record.
11. AI Use Statement exists in a file but is not included by `main.tex`.
12. Runtime caption does not make the limited sample scope prominent enough.
