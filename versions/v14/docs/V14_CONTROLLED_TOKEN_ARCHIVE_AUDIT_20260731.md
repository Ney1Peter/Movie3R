# V14 Controlled Token Archive / API 审计（2026-07-31）

## 1. 审计结论

本次只读检查了：

```text
scripts/v10_geometry_anchor_weight_probe.py
scripts/v10_geometry_token_residual_probe.py
scripts/v10_token_alignment_4source_probe.py
scripts/v9_learned_stream_alignment_overfit.py
src/dust3r/model.py
src/dust3r/datasets/avatarrex.py

output/archive/20260721/v10_geometry_anchor_weight_probe/
  medium_s50_human_token_nogate_20260709/
  medium_s50_human_token_nogate_heldout_offset50_20260709/
```

最终判断是：

```text
现有 archive 的 token 可以复用为 controlled single-human feature archive；
现有 archive 不能直接当作已经带正确 label、严格 held-out 的
per-person B0 root-residual 训练集。
```

原因有四个：

1. `token_features.npz` 和主 checkpoint 没有保存 `target_joints` / `target_poses`；
   label 必须从 `selected_records.json` 和原数据重新构造。
2. V10 原监督针对 strict Human3R **raw local-reset** 后的整个 B segment 共享
   `SE(3)` 对齐，不是冻结 B0 后每个人独立的 root residual。
3. `human_token_*` 已对人物维做 mean/std pooling；本 archive 恰好每帧只有一个预测人，
   因而可以暂时视为单人 token，但不能直接推广为多人逐人 token API。
4. `offset0` 与 `offset50` 只做到 manifest-row disjoint；actor、相机、时间和图像均有重合，
   不能称为严格 held-out。

因此可行的安全用途是：复用 frozen token 先做 controlled observability probe；重新生成
冻结 B0 后的明确 root label，并重做 actor/event-disjoint split。不能把 PTH 中的
`t_fixed`、`t_learned` 或 V10 `target_joints - pred_joints` 直接命名为当前 V14 的
person-local fine-alignment label。

## 2. 两个 archive 的基本结构

两个 archive 均有：

- 200 条 `selected_records`，每个 source 50 条；
- 200 个 `samples/<index>_<pattern_id>/token_features.npz`；
- 每个样本 4 帧 Human3R raw local-reset 输出；
- 每帧 `smpl/*.npz` 都恰好有 1 个预测人，共检查 `800 + 800` 帧，未发现 0 人或多人；
- 每个 token array 均为 `float32`，400 个 NPZ 中未发现 NaN/Inf；
- `anchor_weight_head_4source_probe.pth` 中 `samples` 与 `selected_records.json`
  数量和顺序完全一致。

训练 archive 参数：

```text
samples_per_source = 50
source_offset      = 0
boundary           = 2
token_feature_set  = human_only
eval_only          = false
```

所谓 held-out archive 参数：

```text
samples_per_source = 50
source_offset      = 50
boundary           = 2
token_feature_set  = human_only
eval_only          = true
eval_checkpoint    = offset0 checkpoint
```

`source_offset` 的代码语义是：每个 source 跳过前 50 条不在 bad registry 中的 usable
manifest record，再取后 50 条。它保证的是 usable-record 序号 `0..49` 与 `50..99`
不重复，不实施 actor、sequence、camera-pair 或 timestamp blocking。

## 3. 每个 token array 的精确 shape 和语义

### 3.1 AABB 四个时间步

对 `selected_records` 中的：

```text
(seqA, seqB, start_frame=t)
```

数组第一维固定为四个当前帧：

| row | 实际 view | 语义 |
|---:|---|---|
| 0 | `seqA @ t` | A segment 第 1 帧 |
| 1 | `seqA @ t+1` | A segment 第 2 帧；处理完该帧后执行 reset |
| 2 | `seqB @ t+2` | B segment 第 1 帧，即 boundary current frame |
| 3 | `seqB @ t+3` | B segment 第 2 帧 |

这不是 cut 两侧同 timestamp 的同步 AABB，而是连续人体 motion 的
`A(t,t+1) -> B(t+2,t+3)`。

`views[boundary-1]`，即 row 1，携带 `reset=1`。reset 在 row 1 完成 recurrent
update 后生效。因此：

- row 1 的 `*_after` 是 reset 后状态；
- row 2 的 `*_before` 与 row 1 的 `*_after` bit-exact；
- 实测 400 个样本中，所有相邻 `state_after[i] == state_before[i+1]`、
  `memory_after[i] == memory_before[i+1]` 均为 bit-exact；
- row 1 reset 到 frame-0 初始化状态，实测
  `state_after[1] == state_before[0]` 和
  `memory_after[1] == memory_before[0]` bit-exact。

这点很重要：这里的 legacy local reset 不是在 B segment 第一张图像上重新构造 fresh state，
而是回到 rollout 的 frame-0 initialization。

### 3.2 十个数组

400 个 NPZ 的 shape 完全一致：

| key | shape | 代码语义 |
|---|---:|---|
| `pose_token_in` | `[4, 768]` | 当前帧进入 recurrent decoder 的 pose query；单 token flatten |
| `pose_token_out` | `[4, 768]` | 当前帧供 pose head 使用的 decoder pose token；单 token flatten |
| `human_token_in` | `[4, 1536]` | 当前帧 fused CUT3R+Multi-HMR person prompts 在人物维的 `[mean(768), std(768)]` |
| `human_token_out` | `[4, 1536]` | 最终 recurrent decoder human tokens 在人物维的 `[mean(768), std(768)]` |
| `state_summary_before` | `[4, 1536]` | 当前帧 recurrent state 更新前，token 维 `[mean(768), std(768)]` |
| `state_summary_new` | `[4, 1536]` | 当前帧 decoder 提议的新 state，token 维 `[mean, std]` |
| `state_summary_after` | `[4, 1536]` | update mask 和 reset 后最终 state，token 维 `[mean, std]` |
| `pose_memory_summary_before` | `[4, 3072]` | pose memory 更新前，memory-slot 维 `[mean(1536), std(1536)]` |
| `pose_memory_summary_new` | `[4, 3072]` | 当前帧提议的新 pose memory，slot 维 `[mean, std]` |
| `pose_memory_summary_after` | `[4, 3072]` | update/reset 后最终 pose memory，slot 维 `[mean, std]` |

`_debug_pool_token` 的精确定义是：

```python
concat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
```

因此 `human_token_in/out` 不是保存的逐人 token。只是由于两个 archive 的每一帧都只有
一个 Human3R detection，人物维 std 恒为 0。实测 400 个样本的
`human_token_in[:,768:]` 与 `human_token_out[:,768:]` 全部严格为 0。

这意味着：

- 名义 `[4,1536]` 的 human array 当前只有前 768 维承载信息；
- 它可用于本次 single-human controlled probe；
- 对多人数据，mean/std 会丢失人物顺序、identity 和逐人对应关系，不能作为 per-person
  head 的正式 archive 格式。

### 3.3 V10 实际喂给 head 的 feature

本次两个 run 都选择 `TOKEN_FEATURE_SETS["human_only"] = ("human_token_out",)`。
对 `[4,1536]` 的 `human_token_out`，代码计算：

```text
hist = mean(row0, row1)
cur  = row2
feature = concat(hist, cur, cur-hist, abs(cur-hist))
```

所以：

```text
token_features shape = [200, 6144]
```

row 3 完全不进入这个 boundary pair feature。实测从所有 NPZ 重建出的 feature 与 PTH
里的 `token_features` bit-exact。

anchor-weight probe 又拼接 192 维 predicted-geometry feature：

```text
features shape       = [200, 6336]
token_features shape = [200, 6144]
geometry feature     = 192
```

PTH 中其他主要数组为：

```text
joint_ids       [19]
base_weights    [19]
learned_weights [200,19]
gate            [200]
R_fixed         [200,3,3]
t_fixed         [200,3]
R_learned       [200,3,3]
t_learned       [200,3]
```

这些是 shared segment rigid transform / anchor-weight probe 的中间量，不是逐人 GT root
label。

## 4. Archive 保存了什么，没有保存什么

### 4.1 已保存的 prediction

每个 sample 的 `original_human3r_local_reset` 保存：

```text
camera/{0..3}.npz: pose [4,4], intrinsics [3,3]
smpl/{0..3}.npz:
  rotvec [1,53,3]
  shape [1,10]
  transl [1,3]
  expression [1,10]
```

调用 `load_sequence(..., 4)` 可重建：

```text
pred_poses        [4,4,4]
pred_joints_cam   [4,127,3]
pred_joints_world [4,127,3]
```

### 4.2 未保存的 supervision

以下数组只在当时的 `TokenCachedSample` 内存对象里存在，没有进入 per-sample NPZ 或
`anchor_weight_head_4source_probe.pth`：

```text
target_poses  [4,4,4]
target_joints [4,127,3]
```

variants summary 只保存了每个样本的误差统计和 `gt_bridge_debug.bridge_R/bridge_t`，没有
保存完整 target root/joints。因此不能只打开 token NPZ 或主 PTH 就得到 label。

## 5. 如何从 selected_records 正确恢复 GT alignment target

### 5.1 selected_records 是否足够定位原数据

在当前数据目录和代码版本仍存在的前提下，足够。每条 record 至少包含：

```text
source, group, seqA, seqB, start_frame
```

加上 `run_args.json` 中的：

```text
data_root=/data/wangzheng/iJCV-CODE/data
resolution=(512,288)
resize_mode=human3r_demo
boundary=2
```

即可通过原来的 `load_aabb_views_for_record` 确定性重建四个 view。该函数还根据 source
选择 `Training` 或 `Training/mvhuman`、AvatarReX raw calibration root 和
`pair_scope`，不能用一个简化的通用路径替代。

本次还按 `load_da3_depth=False` 的原始要求检查了 400 条 record 对应的 1600 个 view；
当前 RGB、camera 和 SMPL 必需文件均仍存在，缺失数为 0。因此截至本次审计，label
重建不是被数据缺失阻塞，而是必须显式执行并保存 provenance。

### 5.2 恢复 V10 原 alignment target 的正确 API

应复用原代码路径：

```text
record
-> load_aabb_views_for_record(record, original_args)
-> SMPLModel.update_smpl_gt(views)
-> load_sequence(saved_local_dir, 4)
-> extract_gt_world(
       views,
       pred_data.poses,
       pred_data.joints_world,
       boundary=2,
       joint_ids=sorted(STABLE_JOINTS + FOOT_JOINTS))
-> target_poses, target_joints, bridge_debug
```

`extract_gt_world` 并不是简单读取一个 world translation。它先：

1. 用 `raw_camera_pose`（存在时）或 `camera_pose` 把 GT joints 从 camera 变到 GT world；
2. 用 A segment 的两帧、19 个 stable/foot joints，求一个无尺度 rigid transform
   `bridge_R/bridge_t`；
3. 把完整 GT camera 和人体 target 映射到 Human3R A-segment predicted gauge。

因此 V10 target 的精确定义是：

```text
GT world --(A-segment human-anchor rigid bridge)--> Human3R A gauge
```

不是未经对齐的 dataset world，也不是 post-camera local coordinate。

已有 variants summary 中的 `gt_bridge_debug` 可用于重查 bridge，但为了防止 sample/order
错配，建议用 record + saved prediction 重算，并断言重算 bridge 与 summary 一致。

### 5.3 “root”必须显式定义

当前 Human3R `SMPL_Layer(person_center="head")` 中：

- `smpl/*.npz` 的 `transl` 是 head-center translation；
- `load_sequence` 的 `pred_joints[..., 0, :]` 是 pelvis；
- 实测样本中 `transl == pred_joints_cam[..., 15, :]`，而不等于 joint 0 pelvis。

所以不能把 saved `transl` 无条件叫作 pelvis/root。训练文件中必须明确选择：

```text
head-root label:   target_joints[2, head_idx] - pred_head[2]
pelvis-root label: target_joints[2, 0]        - pred_joints[2,0]
```

V14 当前讨论的 world pelvis/root fine alignment 建议统一使用 joint 0 pelvis，并把
Human3R head translation 只作为一个额外 predicted feature，避免两种中心混用。

## 6. 为什么 V10 target 不能直接作为 V14 B0 person residual

V10 archive 的 prediction 是 strict original Human3R local-reset。其 B segment camera 接近
局部单位 gauge。原实验目标是为整段 B 同时更新 camera 和人体：

```text
raw post camera + raw post joints
-> one shared R,t for the whole B segment
```

`t_fixed/t_learned` 以及 `target_joints - pred_joints` 同时包含：

- shot camera/gauge 跳变；
- shared segment rotation/translation；
- Human3R person-local root/pose error。

而当前 V14 要学的是：

```text
已经冻结并应用 B0 camera Boundary
-> camera bit-exact 不变
-> 只给当前 person 一个 bounded root/orientation residual
```

两者的 label 空间不同。直接拿 raw V10 residual 训练，会让 person head 再次学习本应由 B0
处理的 shared camera transform，等价于把已经冻结的粗对齐偷偷塞回人体 correction。

正确的 V14 pelvis label 应在 B0 已应用后的同一 A gauge 中定义，例如：

```text
p_b0    = B0 已变换后的 predicted pelvis
p_gt    = target_joints[2, pelvis=0]
C_b0    = 冻结 B0 后的当前 camera center
r       = normalize(p_b0 - C_b0)
delta*  = dot(p_gt - p_b0, r)
```

若训练 3D residual，则是 bounded `p_gt - p_b0`；若训练 ray residual，则使用上述标量。
所有 candidate feature 必须在打开 GT 之前冻结。B0 transform/version、root convention、
frame index 和 bridge provenance 都必须随 label 保存。

## 7. offset0 / offset50 是否真正不重合

### 7.1 它们做到的隔离

| key | train | offset50 | overlap |
|---|---:|---:|---:|
| selected records | 200 | 200 | 0 |
| `pattern_id` | 200 | 200 | 0 |
| exact `(source,seqA,seqB,start)` | 200 | 200 | 0 |
| per-source usable local index | `0..49` | `50..99` | 0 |

`mvhuman200` 因 bad registry 跳过个别 manifest rows，原始 manifest index 是训练
`0..50`、offset50 `51..103`，也无相同行。

### 7.2 它们没有做到的隔离

| 语义实体 | train unique | offset50 unique | overlap | offset50 overlap 比例 |
|---|---:|---:|---:|---:|
| actor/group `(source,group)` | 15 | 15 | 15 | **100%** |
| camera sequence `(source,seq)` | 227 | 220 | 140 | **63.6%** |
| ordered camera pair | 197 | 194 | 14 | 7.2% |
| unordered camera pair | 195 | 190 | 20 | 10.5% |
| exact RGB observation `(source,seq,frame)` | 800 | 800 | 5 | 0.625% |
| GT actor-time `(source,group,frame)` | 761 | 777 | 72 | **9.27%** |

更直观地说：

- `31/200` 个 offset50 records 至少共享一个训练集中的完全相同 actor-time；
- 放宽到同 actor 的 ±10 frames，`97/200` 个 offset50 records 与训练集接近；
- 两边都包含 `lbn1/lbn2/zzr`、`thuman00/thuman02`、全部五个 MVHuman100 actors
  和全部五个 MVHuman200 actors。

五个完全相同 RGB observation 均来自 MVHuman200：

```text
200002/22327091 frame 59
200002/22327116 frame 60
200002/22327116 frame 61
200003/22327109 frame 195
200003/22327109 frame 196
```

因此 offset50 最多能叫：

```text
later usable manifest records / row-disjoint smoke evaluation
```

不能叫 actor-disjoint、camera-disjoint、event-disjoint、timestamp-disjoint 或严格 held-out。
尤其 token 很容易编码人物外观、pose、camera/domain identity；在该 split 上的提升不能证明
per-person root residual 能泛化。

## 8. 能否用现有 archive 训练 per-person root residual

### 8.1 可以复用的部分

- 400 个 sample 的 frozen strict-Human3R token array；
- 单人 B0 current frame 对应的 `human_token_out[2,:768]`；
- history/current/difference 的 controlled feature construction；
- 保存的 raw Human3R camera/SMPL prediction；
- `selected_records` 到原 GT 数据的确定性索引；
- 用于 feature/label 两阶段物理隔离的小规模 observability probe。

### 8.2 不可直接复用为 label 或正式评测的部分

- `t_fixed` / `t_learned`：shared segment rigid correction，不是 person residual；
- `learned_weights`：19 个 anchor 的权重，不是 GT root；
- raw `target_joints - pred_joints`：包含未被 B0 消除的 camera/gauge jump；
- `smpl.transl`：是 head center，不是 pelvis；
- offset50 数字：存在 actor/time/image 泄漏，不能作为正式 held-out；
- pooled human token：不能直接用于多人 identity-conditioned per-person head。

### 8.3 最小安全转换流程

建议新建一个派生 manifest，不修改 archive 本身：

```text
Phase A: prediction-only feature freeze
  selected record
  + token_features.npz
  + saved raw Human3R prediction
  + frozen B0 transform
  -> per-sample/person immutable feature payload + hash

Phase B: GT-only label build
  reload record and GT views
  -> exact target_joints in A gauge
  -> compare against B0-aligned predicted pelvis
  -> bounded delta_ray / delta_xyz label

Phase C: leakage-free evaluation
  actor/group-disjoint outer folds
  + camera-pair/event blocking inside train
  + exact `(source,seq,frame)` intersection assertion == 0
  + actor-time intersection assertion == 0
```

当前 15 个 actor/group 可以做 leave-one-actor/group-out，而不是继续用 offset。若要评测
新人物泛化，任何同 actor 的 camera 和 timestamp 都只能落在同一个 fold。若只研究同人物
跨相机适配，也必须单独命名为 within-actor setting，不能与 actor-generalization 混报。

模型输入第一版建议只使用：

```text
human_token_out[0:2,:768] history summary
human_token_out[2,:768] current B0 frame
prediction-only invariant geometry/confidence
```

不使用 token 的恒零 std 半维，不使用 source/group/camera/frame ID。当前 archive row 3 可以
保留作 B1 propagation/evaluation，但不能进入“一张 first-post frame”的 causal feature。

## 9. 最终 go / no-go

| 问题 | 结论 |
|---|---|
| token shape 和当前时间步能否无歧义恢复？ | **可以** |
| 当前 archive 是否真的是逐人 token？ | **仅在当前单人数据上近似可以；格式本身已做人物 pooling** |
| selected_records 是否能重新找到 GT？ | **可以，前提是原 data/code/calibration 仍在** |
| token/PTH 是否已经保存可直接使用的 GT root label？ | **没有** |
| V10 alignment target 是否等于 B0 后的 person root residual？ | **不等于** |
| offset0/offset50 是否严格 held-out？ | **不是** |
| 是否可以立刻把 archive 原样用于正式 residual-head 训练和结论？ | **不可以** |
| 是否值得复用 token 做重标注后的 controlled probe？ | **可以** |

最安全的下一步不是重新跑 Human3R token inference，而是复用这 400 个 token archive，先
构造 actor-disjoint 派生 split，离线重建 GT target，并对每条样本补算 frozen-B0 pelvis
residual。只有这个新 label payload 通过 root convention、camera bit-exact、feature-before-GT
和 split-intersection 自动审计后，才进入小 head 训练。
