# Movie3R-Hybrid V14

V14 是正式的 one-shot latent coarse-to-fine streaming版本。仓库中
`versions/v12/experiments/v14_*` 是历史实验编号，不属于本目录的正式 V14 release。

当前状态（2026-08-01）：

```text
V14.1 event-only routing/preprocessing bugs fixed
corrected single-event upper bound completed
old 10-event pilot withdrawn; corrected rerun not started
V14.2 learned-B0 segment propagation validated on one single-human case
learned B0 before identity matching validated on 41 controlled MultiHuman cuts
frozen dance/box evaluation: 61/61 and 77/78 controlled cuts all-correct
causal B0 + automatic-ID + uniform multi-human loop closed on 24-frame 2/3-person probes
180-cut anchor audit: B0-only outperforms every tested human rotation/translation refinement
camera-only fair retraining converged; joint human supervision gives a small B0 training benefit
Human3R-internal root-depth post-processing rejected after 180-cut and 32-case near/far stress tests
guarded DA3 passes controlled three/dance/box but degrades the local EgoHumans diagnostic
BRTC-LC person root/layout refinement passes a fresh 42-cut/125-person confirmation set
current main route: frozen B0 camera Boundary, then camera-frozen two-view person triangulation
```

当前第一阶段文档：

- `docs/V14_FULL_METHOD_DESIGN_FOR_REVIEW_20260729.md`（完整方法、实现状态、流程图与 ICLR 评审表）
- `docs/V14_ICLR_FINALIZATION_PLAN_20260729.md`（投稿收敛、实验门槛与严格执行顺序）
- `docs/V14_PLAIN_LANGUAGE_PIPELINE_20260729.md`（多人单目 multi-shot 完整流程通俗说明）
- `docs/V14_1_ONE_SHOT_SHADOW_CORRECTION_TRAINING.md`
- `docs/V14_1_INITIAL_PILOT_RESULTS_20260727.md`
- `docs/V14_2_SINGLE_SEQUENCE_RESULTS_20260727.md`
- `docs/V14_2_MULTIHUMAN_SINGLE_CASE_20260727.md`
- `docs/V14_B0_IDENTITY_MATCHING_RESULTS_20260728.md`
- `docs/V14_B0_IDENTITY_MATCHING_FROZEN_EVAL_20260728.md`
- `docs/V14_CAUSAL_AUTOID_MULTIHUMAN_STATUS_20260729.md`
- `docs/V14_CAMERA_HUMAN_RELATIVE_GEOMETRY_AUDIT_20260730.md`（相机/人体相对几何、
  180-cut B0 refinement 消融与 camera-only 公平重训）
- `docs/V14_INTERNAL_ROOT_DEPTH_FEASIBILITY_20260730.md`（内部 pointmap、mask、
  persistent apparent-size scale 与冻结 B0 下的 root-depth 可行性结论）
- `docs/V14_B0_FINE_ALIGNMENT_RESEARCH_TASK_20260730.md`（完整实验账本、失败路线、
  冻结协议与 source-diversity audit）
- `docs/V14_B0_DA3_FINE_ALIGNMENT_FINAL_20260730.md`（最终精对齐原理、架构、公式、
  180-cut 结果、runtime 与落地接口；当前作为已审计的候选/消融保留）
- `docs/V14_B0_TWO_VIEW_TRIANGULATION_FINAL_20260731.md`（当前 BRTC-LC 人体
  root/layout 精对齐主线、冻结确认集、runtime 与失败回退）
- `docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md`（Multi-THuMBS
  指标、公开参考线、本地近似评测和可比性限制）

总体方法规范的仓库内归档：

```text
versions/v14/docs/Movie3R-V14.MD
```

当前诊断用单样本 checkpoint（位于易失的 `/dev/shm`）：

```text
/dev/shm/movie3r_v14_1/v14_1_cut_event_single_simplified_exact_runtime/checkpoint-best.pth
/dev/shm/movie3r_v14_1/v14_1_cut_event_single_v9_parity_exact_runtime/checkpoint-best.pth
/dev/shm/movie3r_v14_1/v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth
/dev/shm/movie3r_v14_1/v14_1_v9_event_only_boundary_geometry_camera_only_self20_fp32_e80/checkpoint-best.pth
```

冻结依赖：

```text
V9: full correct-token decoder refinement and corrected pose/human heads
V12: pre-decode hard reset and fixed shot Boundary
V13: mean_raw_t uniform multi-human consensus
```

V14.1 使用两帧 pre-cut context 和一张显式标记的 post-cut event frame，只在 event
frame 启用 pose/human correction。当前活动 checkpoint 保留 V9-parity 的
semantic/alignment/momentum 三个 correct tokens；two-token `no_momentum` 路径只是
简化诊断消融。V14.1 训练本身暂不接 identity、V13 residual、shot Boundary propagation
或 automatic cut detector；这些模块由后续 V14 runner 组合验证。

简化架构不是 V9 的严格等价版本。V9-parity 诊断配置保留 momentum、reliability、
learned gate 和 context head LoRA，用于逐项验证哪些简化可以安全删除。

当前主链路为：

```text
pre-cut native Human3R tracks
-> first-post-cut V14 shadow branch
-> fresh Human3R hard-reset branch
-> learned B0
-> discard shadow state/humans/pointmap
-> apply and freeze the B0 camera/scene Boundary
-> B0-assisted anonymous root+torso+centred-joints Hungarian
-> last-pre / first-post five-core-joint two-view ray triangulation
-> ray-gap / parallax / joint-MAD observable gate
-> group-median person shift + pre-layout-selected individual residual
-> rigidly translate accepted post people; rejected/unmatched people remain exact B0
-> keep camera, pointmap, pose and shape unchanged
```

受控身份匹配目前只覆盖 cut 前后检测集合相同的样本。下一步仍需解决
appearance/beta identity cue、dustbin 和人数变化，但 identity 不再无条件改写 Boundary。
严格 GT-ID 的 180-cut 诊断显示：旧人体 root translation、完整多人 refinement 和
rotation-only refinement 均弱于 `B0 only`。BRTC-LC 不再修改 camera Boundary，而是利用冻结
B0 后的跨镜头相机基线显式重算逐人 root 深度，并用多人布局共识抑制独立修正噪声。在新的
`three offset1` 确认集上，world root/joint/vertex 分别从 `0.3779/0.4117/0.3891 m`
降至 `0.2314/0.2745/0.2525 m`，pairwise-vector 从 `0.3297 m` 降至 `0.2588 m`；
相机改动严格为零。当前结论只覆盖 root/depth 与多人布局，不代表内部 pose/orientation 或
Multi-THuMBS 官方协议已经解决。
