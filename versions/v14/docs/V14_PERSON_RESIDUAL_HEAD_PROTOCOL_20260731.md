# V14 Person Residual Head 严格训练与冻结评测协议（2026-07-31）

## 1. 目标与边界

目标不是让一个网络重新预测 camera，也不是回到 Human3R 全模型微调，而是在已经冻结的 upstream camera/scene 结果上，为每个匹配人物预测一个很小的标量：

```text
delta_i = person i 沿当前 predicted root ray 的刚性平移量
```

唯一允许的几何更新：

```text
r_i       = normalize(root_i - camera_center)
root_i'   = root_i   + delta_i r_i
joints_i' = joints_i + delta_i r_i
verts_i'  = verts_i  + delta_i r_i
```

camera、scene pointmap、人体 pose、shape、orientation、人体尺寸和其他人物不得被该 head 修改。`delta_i` 最终限制在 `[-0.30 m,+0.30 m]`。gate 失败必须逐人 bit-exact fallback。

本协议把 upstream 状态视为有版本号的冻结输入。正式主线应使用已经 promoted 的 `B0 + frozen da3_safe camera`；如果某次实验仍使用 B0-only，必须作为单独 ablation 命名，不能在训练和冻结评测间静默切换 camera baseline。

当前严格 mesh-depth v2 在 three 前 3 cuts 上 `0/9 accepted`。8 个有 surface 的样本中，raw root-ray residual 只有 3 个与 GT oracle 同号，说明单一 median 本身没有可靠信号。学习式 head 必须通过 out-of-fold 证明它能从 residual 的空间结构和可靠性信息中恢复信号；不能因为 oracle headroom 很大就默认它会成功。

## 2. 因果输入与 GT 隔离

### 2.1 Runtime 可见信息

每个 boundary 只允许读取：

- source shot 在 cut 前已经看到的历史帧；
- target shot 的第一张当前帧；
- 冻结 Human3R/B0/DA3 的预测；
- 部署时可获得的 anonymous track/detection matching 输出；
- 静态 SMPL-X topology/part labels；
- 从上述预测量计算的 confidence、coverage、一致性和几何 residual。

禁止读取 target shot 的未来帧。训练 controlled synthetic 时也必须模拟同样的 `history + first-post` 截断，不能因为完整序列可用就给 head 双向信息。

### 2.2 GT 只能做监督、split 和评价

GT 可用于：

1. 构造训练 label；
2. 构造 actor/event group split；
3. 训练 loss；
4. 训练完成后的指标评价。

GT 不可用于：

- feature；
- candidate；
- runtime gate；
- sample acceptance/rejection；
- identity matching 输入；
- feature normalization；
- threshold 的 test-time 自适应。

实现必须拆成两个物理阶段：

```text
FeatureBuilder(prediction-only) -> immutable X + prediction provenance
LabelBuilder(GT-only)           -> y / split group / evaluator fields
```

`FeatureBuilder` 的函数签名和序列化 payload 中不得出现 `gt`。在 X 文件 hash 固定后，才允许 LabelBuilder 打开 GT。GT identity 可用于把监督 label 配给 detection，也可用于 person-disjoint split，但不得作为数值或 one-hot 进入 X。

### 2.3 Label

在冻结 predicted camera/world gauge 中：

```text
p_i       = predicted root
g_i       = GT root，经 GT pre-camera 只用于监督地对齐到冻结 prediction gauge
r_i       = normalize(p_i - predicted_camera_center)
delta_i*  = dot(g_i - p_i, r_i)
y_i       = clip(delta_i*, -0.30 m, +0.30 m)
```

同时保存 evaluator-only：

- unclipped `delta_i*`；
- tangential error；
- root/joint/vertex before/after oracle；
- label sign；
- GT visibility/identity（仅 evaluator/split）。

当 `|delta_i*| < 0.02 m` 时，sign label 记为 neutral，不强迫网络学习噪声方向。

## 3. 明确禁止的 ID 与泄漏特征

模型输入中明确禁止：

- camera ID、source camera ID、target camera ID、camera-pair ID；
- dataset/source/domain/sequence ID；
- actor/person/track GT ID；
- timestamp、frame index、cut index、文件名、目录名、视频名；
- synthetic asset ID、renderer ID、motion-clip ID、camera-rig ID；
- GT camera/extrinsics/intrinsics/baseline；
- GT root/joints/mesh/SMPL、GT bbox/mask/depth；
- GT residual、GT error、oracle gain、matching correctness；
- target-shot future frame、双向 temporal feature；
- raw global world coordinates或 raw camera/extrinsic matrix；
- 由全数据（含 dance/box/test fold）计算的 normalization/statistics。

以下量虽然来自预测，也高度容易变成 ID proxy，默认不进入主模型：

- raw camera K/focal/principal point；
- raw B0/DA3 pose matrix、baseline direction；
- absolute pixel centroid/bbox coordinates；
- raw SMPL shape/beta（人体指纹）；
- RGB/appearance embedding、CLIP/DINO feature；
- refined human token、CUT3R/MHMR head token；
- raw完整 SMPL pose vector。

这些高风险特征若未来做 ablation，必须单独命名，并同时通过 person-disjoint、event-disjoint 和 camera-pair-disjoint audit。不能因为 three 内部指标提高就并入主线。

允许 dataloader 使用 source provenance 做 batch balancing，但该字段必须在进入模型前删除，并通过单元测试确认模型 forward 无法访问。

## 4. 最可能有信号的 causal features

### 4.1 P0：同像素 strict root-ray residual 的空间结构

不要只输入全人 median。对 mesh z-buffer 与 DA3 同像素的：

```text
e(p) = dot(x_DA3(p) - x_mesh(p), root_ray)
```

提取：

- q05/q10/q25/q50/q75/q90/q95；
- MAD、IQR、trimmed mean；
- positive/negative/near-zero fraction；
- `|median|/MAD`；
- confidence-weighted 与 unweighted 两套统计；
- spatial block/SMPL-X body-part median 与 sign agreement；
- torso/pelvis/head/limb 各自 coverage 与 residual；
- image 四象限或 3×3 region 的 sign coherence。

理由：v2 的失败不是都没有像素，而是同一 silhouette 内 residual 高度多模态。网络最可能利用的是“哪些部位/区域一致”，而不是一个全局值。

静态 SMPL-X face/part label 是 topology，不是 GT，允许使用。第一版建议只使用 6--8 个粗部位，避免在小数据上用每个 vertex ID 记忆人体。

### 4.2 P0：forward/reverse 与独立几何一致性

对 forward/reverse 分别生成严格 scalar/part statistics，再输入：

- forward/reverse median difference；
- sign agreement；
- scale ratio与 relative baseline consistency；
- rotation/direction spread；
- DA3-vs-B0 direction/rotation diagnostics；
- forward/reverse valid coverage ratio。

这些量来自部署可见预测，且直接衡量 gauge 是否可靠。只输入 consistency/invariant summary，不输入 raw camera pose、camera ID 或 baseline direction。

### 4.3 P0：mask、遮挡、截断与几何质量

- silhouette/visible/valid pixel count 和比例；
- image-boundary contact/truncation ratio；
- 多人 z-buffer overlap/occlusion ratio；
- combined predicted semantic-person-mask IoU；
- mesh projection completeness；
- DA3 invalid/sky fraction；
- mesh与DA3 3D difference 的 tangential median/MAD；
- observed ray 与 root ray 的 angle quantiles；
- projected mesh 与 observed-person support 的 spatial overlap。

旧 bbox 和 mesh 实验已经证明：像素数量多不等于像素属于该人；truncation/overlap/tangential mismatch 很可能首先是 reliability feature。

### 4.4 P1：弱质量 cue

- Human3R head score/completeness；
- DA3 confidence 的人内 percentile、entropy、tail ratio；
- automatic identity matcher cost、margin、cycle consistency；
- 当前 root-ray range 与预测人体高度的无量纲比例；
- root-ray surface offset 除以人体高度后的统计；
- causal history 内 root/torso motion 的 normalized dispersion。

这些 cue 只能辅助。已有实验表明 head confidence 本身不足以识别坏人物，history anchor 只有约 `7.1 mm` root gain，native token Re-ID 也不可靠。它们不得单独决定 correction sign。

### 4.5 暂不推荐

- raw appearance/token：更可能回答 WHO，而不是精确 WHERE，也容易记住 actor/camera；
- raw beta/shape：three 只有三个人，极易把 person identity 当 residual prior；
- absolute bbox position/root range/camera baseline：固定 camera rig 下会成为 camera-pair lookup table；
- old bbox surface-change和纯 history range difference：已出现错误符号，不应作为强 feature。

## 5. Residual head 最小架构

为防止小数据过拟合，第一版不使用大型 Transformer 或端到端 RGB backbone。

建议：

```text
per-part robust feature (6--8 parts, each 12--20 dims)
  -> shared 2-layer MLP (32 dims)
  -> masked mean + max pooling

global consistency/quality feature (约 32--64 dims)
  -> concatenate
  -> 2-layer MLP (64 hidden)
  -> mu_delta, log_sigma, sign_logit
```

约束：

- shared part encoder，禁止 part-specific大参数表记忆固定位置；
- head 不接 camera token；
- 输出只有 scalar residual/uncertainty/sign，不输出 SE(3)、scale、pose 或 shape；
- 最终 `delta=0.30*tanh(raw)`；
- 参数量应控制在约 10k--50k；
- dropout/weight decay 固定，不能在 dance/box 调；
- 同时报告 ridge/Huber linear baseline，确认收益不是网络容量记忆。

训练 loss：

```text
L = Huber(mu_delta, y)
  + lambda_nll * calibrated_Laplace_or_Gaussian_NLL(mu_delta, sigma, y)
  + lambda_sign * sign_CE(sign_logit, sign(y)) for |delta*| >= 0.02m
  + lambda_zero * |mu_delta|
```

uncertainty 和 sign head 可用 GT label 训练，但 runtime gate 只能读取预测的 `mu/sigma/sign` 与 causal quality feature。

## 6. three 的严格 train/CV

### 6.1 数据现状

`three` 当前有 41 cuts、7 个 timestamp：

```text
500, 700, 900, 1000, 1100, 1300, 1500
```

同一 timestamp 包含多个 camera pair，并共享相同人物、场景和物理时刻。把 camera pair 随机拆 train/test 会造成严重 cut leakage。每个 timestamp 的所有 camera pairs、所有人和由它们生成的 augmentation 必须属于同一个 event group。

### 6.2 双重阻断 OOF

主 OOF 使用 `7 event groups × 3 actor groups = 21` 个外层组合：

对 test `(timestamp=t, actor=a)`：

```text
test  = actor a 在 timestamp t 的全部 camera pairs
train = actor != a 且 timestamp != t 的样本
```

因此：

- test actor 的任何其他 cut 不在 train；
- test timestamp 的其他人物/相机对不在 train；
- 同一物理 cut、相邻 camera pair和同一人体都不会泄漏；
- 每个可评价 `(actor,timestamp,camera-pair)` 恰好得到一次 OOF prediction。

inner validation 只能从 `train` 内再按完整 timestamp group 留一组，不能随机按 person-row 切分。模型 architecture、feature schema 和优化器尽量预先固定；若做超参选择，只能使用 inner event-group validation 的聚合结果。

GT actor ID 仅参与 split manifest，进入 feature 文件前替换成不可逆 group hash，并在 model batch 中删除。

### 6.3 相邻/重复 cut 去重

若未来加入非零 offset 或更多 timestamp，先对每个样本构造其使用的 source-history/post-frame interval。任意两个样本只要：

- history frame 区间重叠；
- post frame相同/相邻；
- 属于同一 physical event 的 camera permutation；

就通过 union-find 合并为同一 event group。augmentation、forward/reverse、不同 mask erosion 版本继承原 group，不得被当独立样本跨 fold。

### 6.4 Normalization 与采样

- robust median/IQR 只在每个 train fold 拟合；
- validation/test 缺失值使用 train-fold常量与 missing flag；
- 不使用全 `three` statistics；
- person/cut 权重先聚合到 event/actor，避免某个 camera pair 或像素多的人支配 loss；
- 不按 GT residual 大小删除困难样本；可做 label-balanced sampler，但 sampler 配置必须只基于 train labels并记录。

### 6.5 three OOF 通过门槛

只有同时满足以下预注册门槛，才允许训练 final head 并消耗 dance+box：

1. camera bit-exact `100%`；
2. accepted coverage `>=15%`，防止全 fallback；
3. accepted sign accuracy `>=85%`（忽略 `|GT delta|<2 cm` neutral）；
4. accepted root improvement rate `>=80%`；
5. harm `>5 cm` rate `<=5%`；
6. overall root mean 至少改善 `max(20 mm, 5%)`；
7. root P95 不退化；
8. joint/vertex mean 各自不得退化超过 `2 mm`；
9. 每个 actor 和每个 timestamp 的 paired mean 不允许出现 `>20 mm` 退化；
10. 相比 ridge/Huber linear baseline 必须有可复现增益，且不是只靠一个 fold。

门槛失败时停止，不查看 dance/box residual-head 指标。可以继续在 three/controlled train-CV 上形成新版本，但不能用 holdout 反馈调 feature/gate。

## 7. Controlled synthetic 的合法训练方式

AvatarReX、THuman、MVHuman controlled data 可以用于 residual-head 训练和 mechanism supervision，但不是最终真实 holdout。

必须：

- 按 actor/avatar asset group split；
- 同一 motion clip、render event、camera rig 的所有 views/cuts归同一 group；
- 同一 base mesh 的服装/纹理变体不得跨 train/validation；
- 保留至少一组完全 asset-disjoint synthetic validation；
- 生成与 runtime 相同的 B0-qualified coarse state、history长度、first-post因果输入；
- GT camera/body只进入 LabelBuilder/loss/evaluator；
- 不输入 source name、asset ID、renderer/camera rig参数；
- dataloader 可按 source balance，但 source字段在 forward 前删除；
- controlled 样本和 three 的 loss 权重按 event group平衡，不能让百万 synthetic pixel/sample 淹没真实数据。

主 head 应只在 upstream B0/DA3 qualification gate 通过的样本上产生非零 candidate。B0 明显不是 coarse 的 MVHuman case用于训练“外部资格 gate/fallback”或 hard negative，不应要求 person head在错误 camera下修正人体。

Controlled data 可参与每个 three OOF fold 的 train，但任何 synthetic hyperparameter仍只能由 synthetic-heldout + three-inner-CV选择。不能使用 dance/box。

## 8. Final head 与 dance+box 一次冻结评测

### 8.1 Freeze 顺序

在打开 residual-head 的 dance/box report 前固定并 hash：

1. feature schema与禁用字段表；
2. upstream camera版本；
3. split manifest与 OOF prediction；
4. architecture/optimizer/seed ensemble策略；
5. train-only normalizer；
6. residual cap；
7. uncertainty/sign/quality gate threshold；
8. identity matcher版本；
9. checkpoint hash与源码 commit；
10. acceptance criteria。

之后用所有允许的 `three + controlled train` 训练一次 final head。`dance` 61 cuts、`box` 78 cuts只运行一次，不做 threshold sweep、不选 checkpoint、不单独调 sequence参数。

### 8.2 Frozen 评价

必须分别报告 dance、box、pooled，不能用 pooled gain掩盖单 sequence失败：

- root/joint/vertex mean、median、P90、P95；
- paired mean delta与 improvement rate；
- accepted coverage、sign accuracy；
- harm >2 cm、>5 cm、>10 cm；
- boundary catastrophic rate；
- automatic identity coverage/swap；
- camera input/output hash。

冻结通过条件沿用 three OOF 门槛，并额外要求：

- dance 和 box 各自 overall root mean 改善 `max(20 mm,5%)`；
- 两个 sequence 的 root P95 都不退化；
- camera 全部 bit-exact；
- catastrophic 和 identity swap 不增加。

若一次冻结评测失败，完整记录后停止。后续根据 dance/box反馈修改的版本不得再次把同一 dance/box称为 untouched holdout；必须换新 capture/sequence，或明确降级为 dev-contaminated evaluation。

如果 `three/dance/box` 的实际 performer 有重合，dance/box只能称为 capture/motion holdout，不能称为 person holdout。应在训练前用 dataset metadata 固化 global actor-group关系，但 actor ID仍不得进入模型。

## 9. Gate 与 bit-exact 输出协议

runtime 接受必须同时满足：

- upstream B0/DA3 camera qualification；
- predicted identity match有效且 margin/cycle gate通过；
- mesh/person support、truncation、overlap gate通过；
- forward/reverse consistency通过；
- predicted interval不跨 0；
- sign head与 `mu_delta` 同号；
- `|mu_delta|` 超过最小可观测 deadzone；
- correction cap内。

禁止 GT gate。所有 rejection reason结构化输出。

每个 case 自动检查：

```text
hash(camera_before) == hash(camera_after)
rejected person: hash(all SMPL params before) == hash(after)
accepted person: pose/shape/rotation unchanged
accepted person: all vertex pairwise distances unchanged
only translation along predicted root ray changes
```

head 应在 boundary 做一次 causal commit。若 correction 被传播到 shot 后续帧，使用同一 identity state中的刚性 offset，不能用未来帧平滑；Multi-THuMBS Accel评测会惩罚 boundary jump和逐帧抖动。

## 10. Required baselines 与消融

同一 split/protocol 至少比较：

1. zero head / frozen upstream；
2. raw strict mesh-depth median + precision gate；
3. history-anchor estimator；
4. ridge/Huber linear residual head；
5. small MLP residual head；
6. MLP without partwise features；
7. MLP without forward/reverse；
8. MLP without quality/visibility；
9. evaluator-only GT ray oracle与 capped oracle。

禁止把 oracle、GT identity feature或 test-tuned threshold列为可部署方法。所有方法使用相同 auto identity support和相同 denominator。

## 11. 与 Multi-THuMBS 最终指标的衔接

最终需要按 Multi-THuMBS 官方同数据、同 frame/visibility/identity、同 missing/FP处理和同 aggregation协议报告：

| 指标 | Residual head 能否直接影响 | 要求 |
|---|---|---|
| W-MPJPE | 是，person world translation直接影响 | 必须相对冻结 upstream下降 |
| WA-MPJPE | 是 | 必须下降，证明不是只靠全局 gauge |
| MPJPE | 是 | 必须下降 |
| MPVPE | 是 | rigid translation应与 root/joint一致改善 |
| Accel | 是 | 不得因 boundary jump/逐帧 gate抖动退化 |
| ATE | **否**，camera bit-exact | 必须与冻结 upstream逐字节/数值相同；变化即 bug |
| IDs | 间接，若 tracker受位置影响 | 不得增加；identity matcher需单独达标 |

论文当前记录的 EgoHumans reference 为：

```text
W-MPJPE 279.0 mm
WA-MPJPE 166.0 mm
MPJPE 228.3 mm
MPVPE 262.2 mm
Accel 27.3 m/s^2
ATE 0.7 m
IDs 0.97
```

最终“打过 Multi-THuMBS”必须在官方匹配协议下逐项比较。Residual head只能负责人体位置/平滑相关指标；若冻结 upstream ATE仍高于 `0.7 m`，person head不可能解决 ATE，必须由独立且已经冻结验证的 camera模块完成。若 IDs仍高，必须由 identity/tracking模块完成，不能让 residual head偷偷学习 person ID。

当前本地三条 EgoHumans provisional raw-Human3R 为 W/WA/MPJPE/MPVPE/Accel `1088.3/405.1/109.3/130.0/52.49`、ATE `1.848`、IDs/stream `4.00`。由于 split、visibility、missing/FP和官方 supplementary protocol未对齐，这些数字只用于流水线诊断，不能据此声称已经在 MPJPE/MPVPE 上超过论文。

最终 full-stream evaluator 还必须验证：

- 每个 boundary causal commit；
- correction在 track内的持续方式；
- miss/entry/exit/occlusion；
- acceleration与 identity switch；
- camera ATE bit-exact不变。

## 12. 必须保存的复现实物

```text
feature_schema.json
forbidden_feature_audit.json
group_split_manifest.json
controlled_asset_split_manifest.json
train_config.json
normalizer_train_only.npz
checkpoint + sha256
three_oof_predictions.json
three_oof_report.md
dance_box_frozen_predictions.json
dance_box_frozen_report.md
camera_and_fallback_hash_audit.json
multithumbs_protocol_report.json
```

自动 leakage tests：

1. 删除所有 GT payload 后 FeatureBuilder仍能运行且 X hash不变；
2. 随机重命名 camera/source/person/file字段，prediction不变；
3. camera pair顺序变化时，只按显式 forward/reverse规则变化，不依赖 ID；
4. train/test event-group与actor-group交集为空；
5. train-only normalizer不读取 held-out路径；
6. 插入未来帧 sentinel不改变 feature；
7. rejected输出的人体和 camera hash完全不变。

## 13. 停止规则与当前建议

当前最值得尝试的不是把 raw token或 actor shape交给网络，而是一个小型、强分组、只读几何统计的 residual/uncertainty head：

```text
partwise strict root-ray residual
+ forward/reverse consistency
+ truncation/overlap/tangential quality
-> bounded scalar delta + calibrated uncertainty/sign
```

若严格 three OOF仍达不到 coverage、sign precision、20 mm/5% material gain和 tail safety门槛，应停止这条学习式 person-depth主线，保留 B0/DA3 camera与人体 fallback，不消耗 dance/box。只有 OOF 先证明跨 timestamp、跨 actor的因果信号后，才值得进行一次冻结真实序列评测。
