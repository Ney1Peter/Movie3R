# V14 Cut-First Cross-Source Correction 实验交接

更新时间：2026-08-02

本文档面向新的研究会话，独立记录本轮实验的研究问题、权重来源、训练过程、
冻结评测、可视化观察、已成立结论和未解决问题。阅读本文后，不应再把 camera
pose 指标改善等同于完整人体和场景对齐成功。

---

## 1. 一句话结论

跨相机、跨数据来源的 first-post-cut 监督，确实使 V9/V14 隐式 correction 在
held-out camera alignment 上明显优于旧的单 AvatarReX V14.1 checkpoint；但是
当前 deployable 路径只保留 camera-derived `B0`，丢弃 shadow human/scene，因而
camera 指标的明显改善没有稳定转化为人体和场景的主观改善，MVHuman 上仍有大量
catastrophic failure。

当前权重应保留为：

```text
learned coarse B0 proposal
```

不能将它描述为已经完成的最终 Boundary，也不能描述为完整三维纠正模型。

---

## 2. 本轮实验回答什么问题

旧 V14.1 只在一个 AvatarReX camera-cut event 上微调。该单样本模型在部分其他
数据上有一定泛化，但不能区分以下两种可能：

1. V9/Human3R 已经提供了有效的跨视角先验，只缺少更广泛的 cut 监督。
2. 单样本结果只是特定相机对或特定数据分布上的偶然拟合。

因此本轮实验只测试：

> 保持 V9/V14.1 架构和 loss 不变，将训练监督扩大到 AvatarReX、THuman、
> MVHuman100、MVHuman200 的跨相机 cut，并且只纠正 cut 后第一帧，是否能提高
> held-out camera gauge recovery？

本轮没有新增 correct token、decoder、head、matcher、fusion 或外部网络。

---

## 3. 三个权重必须严格区分

### 3.1 正式 V9 基座

```text
/data/wangzheng/iJCV-CODE/Movie3R/checkpoints/
v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth
```

来源关系：

```text
Original Human3R
-> formal V9 mixed AvatarReX + THuman training
```

### 3.2 旧的单 AvatarReX V14.1 诊断权重

```text
/dev/shm/movie3r_v14_1/
v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/
checkpoint-best.pth
```

来源关系：

```text
formal V9
-> one AvatarReX lbn1_1192 cut event
-> event-only fine-tuning for 80 epochs
```

这是本文表格中的 `Old one-Avatar checkpoint`。它不是“只用一个 AvatarReX
样本从 Human3R 零开始训练”的模型，而是继承了 Human3R 和正式 V9 的先验。

该文件位于易失性的 `/dev/shm`，目前仍存在，但不应视为长期归档位置。

### 3.3 当前跨来源最终权重

实验名：

```text
v14_cut_first_cross_source_96ps_e6
```

权重：

```text
/data/wangzheng/iJCV-CODE/Movie3R/output/v14_cut_first_cross_source/
v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth
```

来源关系：

```text
formal V9
-> 96 cut events/source x 4 sources
-> 384 events, 6 epochs
```

关键事实：当前 `cross96` 没有从旧的单 AvatarReX V14.1 checkpoint 继续训练。
旧 V14.1 和 `cross96` 是从同一个正式 V9 基座独立训练的两个分支。这样可以把
性能差异主要归因于监督覆盖，而不是顺序微调历史。

---

## 4. 保留的模型结构

本轮完整保留 V9-parity correction 路径：

```text
input image
-> DINOv2/CUT3R-Human3R encoder tokens
-> V9 correct token
   - semantic component
   - alignment component
   - momentum/history component
-> correct token 与 image/pose/human tokens 一起进入 decoder attention
-> refined correction representation
-> camera correction head + human correction head
-> pose-head LoRA + human-head LoRA
-> explicit corrected camera/human outputs
```

训练时只对明确标记的 first-post-cut event frame 启用 correction token 和两个
head LoRA。两个 pre-cut context frames 走冻结的普通 Human3R 路径。

geometry-preservation loss 保持不变：

```text
self pointmap keep weight   = 20.0
shared pointmap loss weight = 0.1
human param keep weight     = 0.1
```

对应基础配置：

```text
config/train_v14_1_cut_event_single_v9_event_only_geometry.yaml
```

本轮只改变训练 cut 的来源和数量，没有改变上述结构与权重。

---

## 5. 实际 deployable 推理路径

一次 camera cut 同时运行两个角色不同的分支。

### 5.1 Shadow correction branch

```text
pre-cut recurrent state
-> 只读使用
-> cut 后第一帧启用 V9/V14 correction
-> 得到 corrected shadow camera C_shadow
```

shadow branch 可以在 decoder 中利用旧状态和 correct token 做 attention/refinement，
但它的 state、human 和 scene 都不写入正式流。

### 5.2 Raw-reset branch

```text
cut 后第一帧 decode 前 hard reset
-> correction token/head LoRA 全关闭
-> 独立运行原版 Human3R
-> 得到 clean raw camera C_raw_reset
```

该 raw-reset state 是 cut 后 shot 唯一继续提交和传播的 recurrent state。

### 5.3 显式 coarse Boundary

```text
B0 = C_shadow @ inverse(C_raw_reset)
```

然后执行：

```text
discard shadow state
discard shadow human
discard shadow scene
cache one B0
apply the same B0 to raw-reset camera, pointmap and all humans
apply the same B0 to later frames in this post-cut segment
```

这一步只改变 raw-reset shot 的世界坐标 gauge，不改变 raw branch 内部预测的人、
相机和场景之间的相对结构。

整个过程满足：

```text
causal
streaming
no future post-cut frames
no history rewrite
no global BA
shadow state never committed
one cached shared B0 per cut
```

---

## 6. 训练数据构造

每个训练 event 使用三张图：

```text
frames:      [t-1, t, t]
sequences:   [camera A, camera A, camera B]
shot_labels: [0, 0, 1]
```

这里 `camera B frame t` 是 first-post-cut frame。该协议主要监督同步跨相机视角
变化，不包含长时间 post-cut future context。

训练来源：

```text
AvatarReX
THuman
MVHuman100
MVHuman200
```

当前训练配置使用 `max_humans=1`，因此这是 first-post-cut coarse correction 的
单人训练实验，不是最终 automatic multi-human identity/consensus 训练。

manifest 构造脚本：

```text
versions/v14/cut_first_cross_source/build_manifests.py
```

固定随机种子：

```text
20260801
```

`cross24` 和 `cross96` 排除了 frozen10 和 frozen180 中出现的所有无向 camera
pair。每个来源的 camera-pair overlap 都是零。冻结评测 record 没有参与训练、
early stopping、checkpoint selection 或超参数调整。

---

## 7. 分阶段训练过程

三个阶段都从正式 V9 独立初始化，不是逐阶段继续训练：

| Stage | 每来源事件数 | 总事件数 | Epochs | Final loss |
|---|---:|---:|---:|---:|
| cross10 | 3 / 2 / 3 / 2 | 10 | 40 | 0.0467 |
| cross24 | 24 each | 96 | 12 | 0.2045 |
| cross96 | 96 each | 384 | 6 | 0.9020 |

`cross96` 分来源 final loss：

| Source | Final loss |
|---|---:|
| AvatarReX | 1.4232 |
| THuman | 0.3452 |
| MVHuman100 | 1.1987 |
| MVHuman200 | 0.6408 |

所有训练按预先固定的 epoch 结束：

```text
no early stopping
no NaN
no OOM
no source-routing error
only checkpoint-final used for cross24/cross96 conclusions
```

更大 stage 的 loss 更高并不直接表示训练退化，因为 camera/source diversity 更大，
每个 event 的重复次数也更少。

训练集 capacity：

| Stage | N | B0 composite | P90 | Catastrophic |
|---|---:|---:|---:|---:|
| cross10 | 10 | 0.1028 | 0.2018 | 0 / 10 |
| cross24 | 96 | 0.2989 | 0.4881 | 1 / 96 |
| cross96 | 384 | 0.6122 | 1.0406 | 29 / 384 |

这说明 correction path 有拟合能力，但 `cross96` 连训练集都没有饱和。剩余错误
不应全部归因于 held-out domain shift。

---

## 8. 评价定义

Camera 指标：

```text
translation error: meter
rotation error: degree
camera composite = translation + 0.02 * rotation
catastrophic = translation > 1.0 m OR rotation > 30 deg
```

三种输出：

```text
raw_reset:
    独立 hard-reset Human3R 输出

shadow_event:
    correction branch 的诊断输出，包含 corrected camera/human

b0_runtime:
    丢弃 shadow human/scene/state，只将 camera-derived B0
    应用于独立 raw-reset 输出；这是部署路径
```

GT camera 和 GT human 只用于评测，不参与 `B0` 推理。

---

## 9. Frozen10 结果

| Model | Camera T | Camera R | Composite | P90 | P95 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Raw hard reset | 3.1015 | 133.947 | 5.7805 | 7.2919 | 7.7626 | 10 / 10 |
| Old one-Avatar | 1.1980 | 64.863 | 2.4953 | 5.3101 | 5.4701 | 6 / 10 |
| cross10 | 1.1985 | 45.240 | 2.1033 | 3.0773 | 3.7982 | 8 / 10 |
| cross24 | 0.9714 | 51.670 | 2.0048 | 4.4742 | 4.6174 | 5 / 10 |
| cross96 | 1.0861 | 42.427 | **1.9346** | **3.8858** | 5.0526 | **4 / 10** |

`cross10` 虽然改善均值，但 catastrophic 数量上升，说明少量跨来源样本不足以
提高安全性。扩大到 `cross24/cross96` 后，均值、tail 和 catastrophic 才一起
超过 old one-Avatar checkpoint。

---

## 10. Frozen180 主要结果

| Model | Camera T | Camera R | Composite | P90 | P95 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Raw hard reset | 3.1006 | 120.838 | 5.5173 | 8.5929 | 9.2995 | 180 / 180 |
| Old one-Avatar | 1.0984 | 59.455 | 2.2875 | 4.9074 | 5.6680 | 107 / 180 |
| cross10 | 1.3093 | 59.280 | 2.4949 | 4.9273 | 5.7202 | 126 / 180 |
| cross24 | 1.0260 | 46.707 | 1.9602 | 4.2133 | 5.2791 | 97 / 180 |
| cross96 | **0.9073** | **41.302** | **1.7333** | **3.9670** | **4.7186** | **86 / 180** |

相对 old one-Avatar checkpoint，`cross96`：

```text
translation error  -17.4%
rotation error     -30.5%
composite error    -24.2%
P90 composite      -19.2%
P95 composite      -16.8%
catastrophic       107 -> 86
```

这证明跨来源监督带来了真实 held-out camera gain，但 `86/180` catastrophic
仍然远不能满足最终部署要求。

---

## 11. Cross96 分来源结果

| Source | N | Camera T | Camera R | Composite | P90 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| AvatarReX | 48 | 1.0462 | 35.481 | 1.7558 | 4.0350 | 19 / 48 |
| THuman | 48 | 0.2935 | 1.985 | 0.3332 | 0.4626 | 2 / 48 |
| MVHuman100 | 48 | 1.0751 | 63.249 | 2.3401 | 4.1397 | 36 / 48 |
| MVHuman200 | 36 | 1.3169 | 72.222 | 2.7613 | 5.0122 | 29 / 36 |

THuman 很强；AvatarReX 有明显改善；MVHuman100/200 仍主导失败 tail，特别是
wide-view rotation failure。因此不能根据 THuman 或少数最好样本宣称模型已经
学会通用 shot-invariant gauge recovery。

---

## 12. Camera 与 human/scene 结果存在核心分歧

Frozen180 的整体 human-head 结果：

| Output | Camera composite | Human head error |
|---|---:|---:|
| raw_reset | 5.5173 | 1.1046 m |
| shadow_event | 1.7333 | **0.4083 m** |
| b0_runtime | 1.7333 | **1.2288 m** |

解释：

1. `shadow_event` 同时纠正 camera 和 human，因此 human-head 指标明显改善。
2. `b0_runtime` 为保持 clean recurrent state 和 raw relative geometry，丢弃了
   shadow human/scene。
3. `b0_runtime` 只把 camera-derived rigid B0 统一乘到 raw-reset reconstruction。
4. 因此 `shadow_event` 的 human gain 不会自动进入部署结果。
5. Camera pose 变准不代表 raw depth、human root depth、scene shape 或跨 shot
   surface overlap 同时变准。

分来源 human-head 进一步说明问题：

| Source | raw_reset | shadow_event | b0_runtime |
|---|---:|---:|---:|
| AvatarReX | 0.7533 | 0.5218 | 1.1458 |
| THuman | 0.7202 | 0.2571 | 0.3264 |
| MVHuman100 | 1.8908 | 0.3284 | 2.0736 |
| MVHuman200 | 1.0374 | 0.5651 | 1.4160 |

所以当前实验最重要的未解矛盾是：

```text
shadow correction contains useful human information
but committing shadow state/geometry risks contaminating the clean raw stream
while camera-only B0 does not preserve the shadow human gain
```

---

## 13. 可视化记录

viewer payload：

```text
output/v14_cut_first_cross_source/visualization_cross96/
```

导出脚本：

```text
versions/v14/cut_first_cross_source/export_visualization_cases.py
```

导出严格使用：

```text
camera-derived B0
-> raw-reset camera
-> raw-reset pointmap
-> raw-reset humans
```

没有将 shadow human 当作最终人体。四个 viewer 的灰色 camera 是 raw-reset camera
参考，不是 GT；viewer UI 中继承的 `GT camera` 名称在这些窗口里不能按字面理解。

选取案例：

| Port | Source | Raw comp. | B0 comp. | 观察 |
|---|---|---:|---:|---|
| 8101 | AvatarReX | 3.091 | 0.319 | camera 指标强改善例 |
| 8102 | THuman | 9.358 | 0.062 | camera 与 human 都明显改善的强例 |
| 8103 | MVHuman100 | 5.572 | 0.338 | camera 指标强，但完整视觉改善有限 |
| 8104 | MVHuman200 | 4.479 | 6.112 | 明确失败/回归例 |

`8103` 的具体诊断：

```text
camera composite:
5.5721 -> 0.3379

human head:
raw_reset    1.9688 m
shadow_event 0.0937 m
b0_runtime   1.6717 m
```

因此用户主观看到“8103 没好多少”是合理的。它只应标记为 camera-pose success，
不应标记为 full-geometry visual success。后续可视化抽样不能只按 camera composite
挑最好样本。

截至 2026-08-02，四个 viewer 暂时运行在 `8101-8104`；端口是临时状态，后续
会话应先用 `ss -ltnp` 检查，不能假设它们一直存在。

---

## 14. Streaming / no-cut parity

`cross96` 与原版 `src/human3r_896L.pth` 在 no-event sequence 上比较：

```text
camera pose max_abs = 0.0
pointmap max_abs    = 0.0
confidence max_abs  = 0.0
SMPL fields max_abs = 0.0
all shapes match    = true
```

因此 event-only routing 没有改变普通无 cut 帧。该结果支持：

```text
normal frames use original Human3R behavior
only first post-cut frame invokes shadow correction
later frames use clean reset state + cached B0
```

B0 应用后的 camera 与 shadow camera 数值一致，4x4 matrix maximum disagreement
小于 `2.4e-7`。

---

## 15. 已经成立的结论

1. V9 correct-token、decoder refinement 和 two-head correction 可以改造成只在
   first-post-cut frame 触发的 non-committing shadow transaction。
2. 无 cut 帧可以与原版 Human3R 保持数值完全一致。
3. 只纠正 cut 后第一帧，已经足以产生可传播给整个 raw-reset shot 的显式 `B0`。
4. 跨相机、跨来源监督相对单 AvatarReX V14.1 确实改善 held-out camera mean、
   tail 和 catastrophic count。
5. 更大数据覆盖从 `cross24` 到 `cross96` 继续有效，单纯的单样本过拟合不是全部
   解释。
6. state continuity 和 world gauge continuity 可以在实现上分离：shadow 读取旧
   state 恢复 gauge，raw branch 保持 clean state。

---

## 16. 尚未成立的结论

1. `cross96` 不是安全的最终 Boundary：frozen180 仍有 `86/180` catastrophic。
2. Camera 指标改善没有稳定转化为 human/scene/full-geometry 改善。
3. MVHuman wide-view rotation 仍大量失败。
4. 当前训练和核心评测是 `max_humans=1`，不能据此声称 automatic multi-human
   route 已完成。
5. 当前没有解决 shot scale、monocular root depth 或 raw scene geometry error。
6. 当前没有证明应提交 shadow human/scene；这样做可能破坏 state purity 和 shot
   内 Human3R 的稳定相对结构。
7. 当前没有证明继续无结构地扩大数据或增加 correct-token 复杂度会消除失败 tail。
8. 选取少数 camera metric 最好案例不能替代完整视觉和 human metric 评测。

---

## 17. 当前研究决策

保留：

```text
first-post-cut shadow correction
camera-derived coarse B0
non-committing state/gauge decomposition
cross-source supervision result
```

不保留为最终结论：

```text
cross96 B0 is a deployable final Boundary
camera metric success equals full 3D success
MVHuman is solved
multi-human is solved
```

当前合理定位：

```text
cross96 B0 = learned coarse proposal that moves some cuts into a better basin
```

后续只有在存在保守 gate/fallback 或 bounded explicit refinement 时，才能考虑将其
接入完整系统。

---

## 18. 建议下一会话优先分析的问题

下面是需要分析的问题，不是已经冻结的新方案：

1. 为什么 `shadow_event` human-head 为 `0.408 m`，而 `b0_runtime` 为
   `1.229 m`？差值来自 raw human depth、camera-human relative geometry，还是
   pointmap/scale？
2. Human auxiliary correction 是否真的帮助 camera B0？需要使用已有 camera-only
   ablation 做同协议对比，不能凭结构直觉决定删除 human head。
3. 如何定义 full-geometry success，而不是继续只按 camera composite 选择？至少应
   联合 camera、human root/head、scene overlap 和主观可视化。
4. 能否建立 precision-first B0 acceptance gate，在困难 MVHuman case 上回退 raw
   或已有安全 baseline，从而降低新增 catastrophic failure？
5. 如果继续做 fine alignment，是否应严格写成 `B_final = DeltaB * B0`，并将
   `DeltaB` 限制在小 residual 范围，而不是重新估计完整 SE(3)？
6. 什么时候接回多人路线？合理顺序应先确认 coarse B0 对完整几何确实有帮助，再
   测试 B0-before-WHO 和冻结 Uniform Multi-Human Consensus。
7. 是否存在“camera 正确但 full geometry 错误”的可观测部署信号？如果没有，
   cross96 只能作为分析性 proposal，而不能无条件启用。

下一会话不应优先增加新 matcher、learned fusion、外部 depth model 或更复杂 memory。
首先需要解决当前最明确的证据冲突：

```text
camera coarse alignment improves
but deployable raw human/scene alignment often does not
```

---

## 19. 关键代码与结果索引

实验说明与既有汇总：

```text
versions/v14/cut_first_cross_source/README.md
versions/v14/cut_first_cross_source/RESULTS.md
```

训练配置：

```text
config/train_v14_1_cut_first_cross_source_10.yaml
config/train_v14_1_cut_first_cross_source_24ps.yaml
config/train_v14_1_cut_first_cross_source_96ps.yaml
config/train_v14_1_cut_event_single_v9_event_only_geometry.yaml
```

评测与 parity：

```text
versions/v14/cut_first_cross_source/evaluate_cut_events.py
versions/v14/cut_first_cross_source/evaluate_four_source_b0.py
versions/v14/cut_first_cross_source/audit_reset_only_parity.py
```

最终 frozen reports：

```text
output/v14_cut_first_cross_source/eval_cross96_frozen10/
output/v14_cut_first_cross_source/eval_cross96_180/
output/v14_cut_first_cross_source/eval_cross96_train96_capacity/
output/v14_cut_first_cross_source/reset_only_parity_cross96/
```

旧单 AvatarReX 对照 reports：

```text
output/v14_cut_first_cross_source/eval_current_single_frozen10/
output/v14_cut_first_cross_source/eval_current_single_180/
output/v14_cut_first_cross_source/reset_only_parity_current_single/
```

可视化：

```text
versions/v14/cut_first_cross_source/export_visualization_cases.py
output/v14_cut_first_cross_source/visualization_cross96/manifest.json
```

完整方法背景：

```text
versions/v14/docs/Movie3R-V14.MD
versions/v14/docs/V14_FULL_METHOD_DESIGN_FOR_REVIEW_20260729.md
versions/v14/docs/V14_ICLR_FINALIZATION_PLAN_20260729.md
```

---

## 20. 给下一会话的最短读取顺序

```text
1. 本文档
2. versions/v14/cut_first_cross_source/RESULTS.md
3. output/v14_cut_first_cross_source/eval_cross96_180/four_source_b0_evaluation.md
4. output/v14_cut_first_cross_source/visualization_cross96/manifest.json
5. versions/v14/docs/V14_ICLR_FINALIZATION_PLAN_20260729.md
```

分析下一步时，应同时使用 camera 数值、human 数值和可视化证据，不要只沿用其中
任何一类证据。
