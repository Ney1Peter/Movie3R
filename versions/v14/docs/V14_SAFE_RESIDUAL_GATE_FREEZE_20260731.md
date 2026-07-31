# V14 Safe Person Residual Gate：冻结准入、Invariant Feature 与 OOF 选择协议

> 日期：2026-07-31
> 范围：只审计 `three` development；本轮没有读取或运行 `dance`、`box`。
> 当前结论：**NO-GO，不允许消耗一次 `dance/box` frozen evaluation。**

## 1. 当前 frozen 候选

当前已经生成的候选为：

```text
model       = Ridge(alpha=10)
features    = 172 dimensions
OOF split   = timestamp x actor double block, 21 folds
gate        = abs(raw OOF prediction) >= 0.20 m
action      = clip(raw prediction, -0.05 m, +0.05 m)
fallback    = rejected person bit-exact B0
camera      = bit-exact unchanged
development = three only
```

对应结果：

```text
41 cuts / 122 people / 81 accepted
coverage                         66.4%
B0 root mean                     0.3789 m
locked root mean                 0.3583 m
root gain                        20.6 mm / 5.4%
joint gain                       19.8 mm
vertex gain                      19.8 mm
accepted sign accuracy           85.2%
accepted root improve rate       84.0%
root/joint/vertex harm > 5 cm    0.0%
root P95                         0.8429 -> 0.8055 m
joint P95                        0.8755 -> 0.8308 m
vertex P95                       0.9232 -> 0.8784 m
camera                           bit-exact
rejected fallback                bit-exact, 41/41
```

结果文件及审计时 hash：

```text
be3a0546e15e25a8e655bab0e6e333327c4bf0146b0e3a57ac1d2848f733a968
output/v14/fine_alignment_research/person_residual_head_cv_three_double_block_ridge/v14_person_residual_head_cv.json

62dabee4148f0b2a4d2bcb2192024a446bfaf6d0000668630d21d336ea5f48b2
output/v14/fine_alignment_research/person_residual_head_locked_three/v14_person_residual_head_locked.json
```

## 2. 为什么结论仍然是 NO-GO

这组结果证明了“小幅、逐人、沿 ray 的刚性 correction 可以安全工作”，但尚未证明它是可泛化的主线。不能因为新协议中的 `max(20 mm, 5%)` 刚好通过，就覆盖更早已经写入总任务的 material-gain 条件。

### 2.1 更早的总任务门槛没有通过

`V14_B0_FINE_ALIGNMENT_RESEARCH_TASK_20260730.md` Section 6 对 person-only 路径要求：

```text
human root error 至少下降 8%
layout error 至少下降 5%
camera non-inferior
```

当前实际是：

| 指标 | B0 | Locked | 相对改善 | 结论 |
|---|---:|---:|---:|---|
| root mean | 0.3789 | 0.3583 | 5.4% | 未达到 8% |
| pairwise distance mean | 0.1345 | 0.1300 | 3.3% | 未达到 5% |
| pairwise vector mean | 0.3371 | 0.3306 | 1.9% | 未达到 5% |

此外 pairwise distance P95 从 `0.4793` 轻微升至 `0.4818 m`。它不是灾难性退化，但不能支持“layout 已显著改善”。

阈值冲突时必须使用较早且更严格的预注册条件，不能在看过 OOF 结果后把 `8%` 降为 `5%`。后续如需修改总任务门槛，必须由用户明确批准并记录为协议变更，不能由实验结果反推。

### 2.2 172 维输入违反当前 feature contract

旧模型虽然没有直接输入字符串 camera/person ID，但包含大量可以等价充当 lookup key 的绝对量：

- `camera_baseline_local_{x,y,z}`、`camera_baseline_norm`；
- `camera_relative_rotvec_*`、`camera_rotation_deg`；
- `b0_translation_norm`、`b0_rotation_deg`；
- absolute bbox center/width/height/area；
- absolute camera-local root、root range、root ray 分量；
- absolute torso/root-orientation rotvec；
- raw metric matcher root/joint/total cost；
- raw pre/post body size和 extent。

`three` 只有固定 rig、3 个 actor 和 7 个 timestamp。上述量即使不叫 ID，也可能让 Ridge 学到 camera-pair、actor 或 capture prior，而不是通用的人体精对齐机制。timestamp × actor double blocking 能阻断同 actor、同 physical event，但不能自动阻断同 camera pair 在其他 timestamp 中形成的 proxy。

因此当前结果只能称为：

```text
promising development safe-step with proxy risk
```

不能称为 invariant person-alignment model，也不能据此打开 frozen holdout。

### 2.3 新协议尚缺的证据

当前 locked report 后处理重算显示，三个 actor 和七个 timestamp 的 root mean 都未退化：

| Unit | B0 root | Locked root | Delta |
|---|---:|---:|---:|
| person0 | 0.3512 | 0.3311 | -0.0200 |
| person1 | 0.3589 | 0.3340 | -0.0249 |
| person2 | 0.4277 | 0.4109 | -0.0168 |
| t500 | 0.2963 | 0.2811 | -0.0152 |
| t700 | 0.3604 | 0.3604 | -0.0000 |
| t900 | 0.3859 | 0.3516 | -0.0343 |
| t1000 | 0.3230 | 0.2965 | -0.0265 |
| t1100 | 0.3294 | 0.3056 | -0.0238 |
| t1300 | 0.3783 | 0.3571 | -0.0212 |
| t1500 | 0.5754 | 0.5645 | -0.0109 |

这补上了旧 172 维 locked candidate 的 root group safety 检查。当时它仍缺：

1. invariant-only 模型相对 constant、Ridge、Huber、Logistic 的真实增益；
2. camera-pair proxy adversary 或 leave-camera-pair-out 审计；
3. 用同一 frozen action 重建 joint、vertex、layout 后的完整 gate grid；
4. invariant final feature/schema/code hash；
5. 旧 `8% root + 5% layout` material gain。

下一节的 invariant OOF 已补做第 1 项和 leave-camera-pair-out部分，但给出了负结果；第 3--5 项仍未满足。

### 2.4 去 proxy 后的首轮 invariant OOF 已给出更明确的负结果

data-agent 按上述审计先修正了 shared-world ray、raw matcher cost 和 neutral sign，随后只在 `three` 跑了 67 维 strict-invariant constant/Ridge/Huber/Logistic double-block OOF：

```text
output/v14/fine_alignment_research/
person_residual_head_invariant_cv_three/
```

审计时结果 hash：

```text
cbdca0d5492f717ebad049bf82aa974c4a7db2d6589c4cc951331477299ccb01
v14_person_residual_head_invariant_cv.json
```

| Method | Root mean | Gain vs B0 | Accepted sign | Improve | Harm >1 cm |
|---|---:|---:|---:|---:|---:|
| B0 | 0.3789 | 0.0 mm | N/A | 0.0% | 0.0% |
| train-fold mean constant | **0.3554** | 23.5 mm | 75.4% | 71.3% | 26.2% |
| train-fold median constant | **0.3554** | 23.5 mm | 75.4% | 71.3% | 26.2% |
| train-fold majority-sign constant | **0.3554** | 23.5 mm | 75.4% | 71.3% | 26.2% |
| invariant Ridge | 0.3558 | 23.1 mm | 74.6% | 71.3% | 23.0% |
| invariant Huber | 0.3648 | 14.0 mm | 63.6% | 60.7% | 34.4% |
| invariant Logistic sign | 0.3585 | 20.4 mm | 73.7% | 70.5% | 26.2% |

核心不是这些方法都比 B0 mean 低，而是：

```text
Ridge - constant = -0.4 mm feature gain
```

即 Ridge 比不看任何 person feature、只用 train-fold label mean 的 constant prior 还差 `0.4 mm`。Logistic confidence 从 `0.0` 扫到 `0.95` 后，accepted sign 最高只有 `80.7%`，没有达到 `85%`；严格规则下 `selected gate = None`。

Secondary leave-one-camera-pair-out 得到：

```text
constant  0.3554 m
Ridge     0.3551 m，较 constant 仅 +0.3 mm
Logistic  0.3626 m
```

即使额外阻断 camera pair，Ridge 的 individual-feature increment 也只有 `0.3 mm`，远低于可视为通用机制的量级。

这给出比 proxy-risk 推断更直接的证据：旧 172 维 `85.2%` sign 很可能主要来自绝对 camera/root/bbox shortcut。去掉这些 shortcut 后，当前 67 维相对特征没有证明 person-conditioned 增量，残差主要仍由 capture-level constant signed prior解释。

该首轮输出已包含 constant mean/median/majority-sign、Ridge、Huber、Logistic 和 camera-pair LOGO。它仍未重建 full joint/vertex/layout，也没有 camera-pair classification adversary；但由于所有 invariant feature model 都未超过 constant prior、且没有任何 strict eligible confidence gate，其 `None/NO-GO` 结论已经足够阻止打开 `dance/box`，无需为一个已失败的候选继续消耗 full-geometry 或 holdout 预算。

## 3. Invariant feature whitelist 的判定原则

一个量来自 prediction，并不自动代表它可进入模型。合法 feature 必须满足至少一类：

1. 对共同 world SE(3) 变化不变；
2. 由同一人的 pre/post 比值或差分构成，消除绝对 actor/camera尺度；
3. 在 predicted body frame 中表达，并用 predicted body size 归一化；
4. 是 confidence、coverage、agreement 等无量纲可靠性统计；
5. 只使用部署时可见的 causal prediction，GT 只在 feature hash 固定后进入 label/evaluator。

### 3.1 可直接进入第一版 whitelist

#### A. Strict person pointmap residual 的相对统计

对同像素 mesh z-buffer 和 DA3 point：

```text
e(p) = dot(x_DA3(p) - x_mesh(p), predicted_root_ray)
u(p) = e(p) / predicted_body_height
```

允许：

- `u(p)` 的 q05/q10/q25/q50/q75/q90/q95、trimmed mean；
- MAD、IQR、positive/negative/near-zero fraction；
- torso/pelvis/head/limb 分区的 normalized median 和 sign agreement；
- valid/visible/semantic-support coverage；
- confidence-weighted 与 unweighted 结果的差异；
- forward/reverse normalized residual 的差、ratio、sign agreement。

这里 predicted body height 只作为除数，不能同时把原始 height 作为 feature 输出。

#### B. Cross-view 相对比例

允许：

```text
log(post bbox width / pre bbox width)
log(post bbox height / pre bbox height)
log(post predicted-body extent / pre extent)
log(post body RMS / pre body RMS)
post/pre completeness ratio and difference
```

只保留 ratio/difference；不保留 absolute bbox center、width、height、area 或 absolute body size。

#### C. Body-frame normalized motion

以 last-pre predicted torso frame为基：

```text
jump_body = R_torso_pre^T (root_post_aligned - root_pre) / body_scale_pre
velocity_body = R_torso_pre^T velocity_pre / body_scale_pre
accel_body = R_torso_pre^T accel_pre / body_scale_pre
```

这些 vector component 合法，因为坐标轴来自同一人的 predicted body frame，不是 camera/world 固定轴；body scale 仅用于归一化。

#### D. 同一 shared frame 中的 ray 与 relative rotation

ray cosine 必须先把两个 ray 放入同一个坐标系：

```text
r_pre_world  = normalize(root_pre - camera_pre_center)
r_post_world = normalize(root_post_B0 - camera_post_B0_center)
ray_cos      = dot(r_pre_world, r_post_world)
```

禁止把 `pre_camera_local_ray` 和 `post_camera_local_ray` 直接点乘；二者原本不在同一坐标系。

相对 torso/root rotation 可写成：

```text
R_rel = R_pre_body^T R_post_body_in_pre_world
```

`R_rel` 的 angle、`sin(angle)`、`cos(angle)` 和在 pre-body frame 中表达的 rotvec 分量属于合法相对量。实现需要先将输入投影到合法 SO(3)，并测试共同 world rotation 后 feature 数值不变。

#### E. Mask、可见性和质量

允许：

- completeness、mesh support coverage、semantic-person IoU；
- truncation/boundary-contact ratio；
- z-buffer overlap/occlusion ratio；
- DA3 invalid/confidence quantile、entropy、tail ratio；
- forward/reverse valid coverage ratio；
- missing flags。

absolute completeness 虽是无量纲量，仍可能带 camera signature。可以作为弱质量 cue 暂留，但必须通过 camera-pair proxy audit；若 adversary 明显高于 chance，应只保留 pre/post difference、ratio 或 min-value。

#### F. Matcher feature

允许：

- selected rank fraction；
- row chosen-minus-best / row MAD；
- best-second margin / row MAD；
- assignment best-vs-second margin / assignment cost scale；
- cycle-consistency、valid fraction、dustbin/ambiguity flag。

不允许 raw meter root cost、raw joint cost、mixed-unit total cost和未归一化 margin。归一化分子与分母必须来自同一 cost definition，不能拿 torso degree 除 root meter。

### 3.2 明确禁止

下列字段不进入主模型，无论它们是否来自 prediction：

| 类别 | 禁止项 |
|---|---|
| 直接 ID | camera/source/target/pair、actor/person/track GT ID、timestamp、case/file/path、dataset/domain |
| Camera | raw K、focal、principal point、pose matrix、relative pose axis、baseline direction/norm、B0/DA3 raw transform |
| 绝对人体位置 | raw world/camera root、range/depth、absolute ray xyz、absolute bbox center/size |
| 人体指纹 | raw beta/shape、absolute body extent、absolute torso/root orientation、appearance/token embedding |
| GT/未来 | GT camera/body/mask/depth/residual/error/gain、future-post frames、双向未来 feature |
| 全数据统计 | 使用完整 `three` 或任何 holdout 拟合的 normalizer/imputer/threshold |

仅把字符串替换为不含 `camera` 或 `person` 的名字，不会使 feature 合法；validator 必须检查 feature 的构造公式和 provenance，而不仅是字段名。

## 4. 对 invariant data-agent 脚本的审计

审计对象：

```text
versions/v14/probe_person_residual_head_invariant_cv.py
```

当前方向正确的部分：

- 明确锁定 `three`，脚本没有 `dance/box` 输入入口；
- timestamp × actor double-block OOF；
- B0 只用于把 post geometry 放到 shared frame，不直接输出 B0 参数；
- absolute camera/root/range/bbox center 已列入禁止项；
- bbox/body 特征只计划输出 pre/post log-ratio；
- motion 在 last-pre body frame 中表达并除以 predicted body RMS；
- cross-view ray 已改为 shared-world ray 后再计算 cosine；
- matcher raw metric cost 已改为 rank/self-normalized margin；
- body-relative rotvec xyz 在严格 pre-body frame 中，原则上合法；
- Logistic 已排除 `|label|<2 cm` neutral；
- 已包含 train-fold mean/median/majority constant、Ridge、Huber、Logistic；
- 已增加 leave-one-camera-pair-out secondary proxy audit。

运行并把其结果用于候选选择前，仍必须修正或补齐：

1. 若未来出现 root-eligible gate，selection 不能只看 root；每个 threshold 必须重建 root/joint/vertex/layout并执行完整准入条件。
2. `root_protocol_eligible` 只能标记诊断资格；只要 `full_geometry_eligible=False`，final selection必须返回 `None`。
3. Leave-camera-pair-out 已完成；若未来有非零 feature增量，再增加 camera-pair classifier adversary，报告 balanced accuracy/macro-F1 与 permutation chance。camera pair仅供 audit grouping，不进入 X。
4. 对所有 claimed invariant feature 做数值单元测试：共同 world SE(3) 随机扰动后 feature max difference `<1e-10`；predicted body uniform scale后 dimensionless feature不变。
5. relative rotation 输入在每次计算前显式投影 SO(3)，不能只在异常时修复。
6. absolute completeness 是否保留由 proxy audit决定；禁止根据它和 label 的单变量相关性临时增删。
7. final whitelist、normalizer、imputer、model hyperparameter、threshold grid 与 tie-break 需要在读取本轮候选选择结果前 hash。

在上述检查完成前，该脚本是合格的开发实现草案，不是可冻结 final candidate。

## 5. 必须报告的 baseline

所有 baseline 必须使用完全相同的 21 个 double-block folds、相同 denominator、相同 correction cap、相同 full-geometry evaluator。normalization、imputation、label prior只能在每个 train fold拟合。

### 5.1 Zero / B0

```text
delta = 0
```

这是所有 mean/tail/harm 的基准，也验证 rejection 和 camera bit-exact。

### 5.2 Constant prior

至少报告：

1. train-fold label mean，clip 到 `±0.05 m`；
2. train-fold label median，clip 到 `±0.05 m`；
3. train-fold majority sign × 固定 `0.05 m`。

constant 必须对整个 test fold相同，不能按 actor、timestamp、camera pair 或 sample condition变化。它用于判断模型是否只学到了 capture-level “多数人向同一方向偏”的 prior。

### 5.3 Ridge

```text
X_invariant -> StandardScaler(train only) -> Ridge(alpha=10) -> scalar delta
```

Ridge 是最低容量的 feature-use baseline。必须报告 raw prediction 和 clipped action；不能只报告经过最优 gate 后的数字。

### 5.4 Huber

```text
X_invariant -> StandardScaler(train only) -> HuberRegressor(fixed epsilon/alpha)
```

它检查 Ridge 的收益是否由少量大 residual 支配。epsilon、alpha 必须固定或只在 inner event-group CV选择。

### 5.5 Logistic sign/confidence

训练样本：

```text
positive: delta* >= +0.02 m
negative: delta* <= -0.02 m
neutral:  |delta*| < 0.02 m，禁止进入二分类 loss
```

输出 `p_pos`，基础 action 固定为：

```text
sign = +1 if p_pos >= 0.5 else -1
delta = sign * 0.05 m
confidence = abs(2 p_pos - 1)
```

若 Logistic 用作 Ridge/Huber 的 gate，而不是独立 sign baseline，必须 nested/cross-fit：训练 gate 时使用的 base-regressor prediction也必须是 train 内 OOF，不能使用 in-sample fitted prediction构造“是否正确”标签。

### 5.6 报告要求

每种 baseline 都必须报告：

- raw prediction distribution、applied action、coverage；
- root/joint/vertex mean、P50/P90/P95；
- pairwise root-distance 和 root-vector mean/P95；
- accepted sign accuracy、accepted improvement rate；
- harm >1/2/5/10 cm；
- per actor、timestamp、camera pair；
- prediction 与 train-fold constant 的差值；
- camera、rigid-shift、fallback hash。

若 invariant Ridge/Huber/Logistic 没有稳定超过 constant prior，则结论必须是“没有发现 person-conditioned通用信号”，不能凭其相对 B0 改善晋级。

## 6. Confidence gate 在 `three` 上唯一合法的选择方式

本项目允许把 `three` 当 development set。因此可以使用 **OOF prediction + OOF label 做一次候选选择**，但只能执行一次，完成后永久锁定。

### 6.1 选择前冻结

在查看 gate grid 的 label-based metrics 前固定并 hash：

1. feature whitelist与构造代码；
2. forbidden/provenance audit；
3. 21-fold split manifest；
4. train-only preprocessing；
5. constant/Ridge/Huber/Logistic hyperparameters；
6. action cap；
7. confidence definition；
8. 有限 threshold grid；
9. full-geometry evaluator；
10. eligibility rules与 deterministic tie-break。

然后一次性生成每个 sample 恰好一条 double-block OOF prediction。OOF 文件写出并 hash 后，才允许 LabelBuilder 打开 GT并对 grid评分。

### 6.2 Eligibility 使用完整且更严格的条件

任一 `(model, confidence threshold)` 只有同时满足以下条件才是 eligible：

```text
camera bit-exact                         100%
rejected fallback bit-exact              100%
accepted coverage                        >= 15%
accepted sign accuracy                   >= 85%, neutral excluded
accepted root improvement                >= 80%
root/joint/vertex harm >5 cm             <= 5%
root P95                                 non-inferior
joint/vertex mean                        each improves or degrades <=2 mm
each actor/timestamp mean degradation    <=20 mm; recommended non-inferior
root mean gain                           >=20 mm AND >=8%
primary layout mean gain                 >=5%
secondary layout mean/P95                non-inferior
invariant model                          reproducibly beats constant prior
```

为消除“layout”歧义，下一次运行前必须固定一个 primary layout metric。推荐使用 `pairwise root-vector mean` 作为 primary，因为它同时保留距离和方向；`pairwise root-distance mean/P95` 作为 secondary safety。不能看完结果后在 distance/vector 中挑更好的一项。

这里使用 `>=8%` 而不是仅 `>=5%`，因为更早的总任务协议更严格。若没有 candidate eligible，选择结果就是 `None/NO-GO`；禁止把门槛降低后重选。

### 6.3 唯一选择与 tie-break

在预声明的有限 candidate 集中：

1. 先删除所有不满足 eligibility 的 candidate；
2. 在剩余候选中选择 overall root mean 最低者；
3. 若差异 `<1 mm`，选择 worst actor/timestamp gain 更好者；
4. 再相同则选择 coverage 更高者；
5. 再相同则选择参数更少、threshold 更严格者。

这一步完成后记录：

```text
selected model
selected threshold
action cap
feature schema hash
OOF prediction hash
split hash
preprocessing hash
source code hash
selection table and rejected reasons
```

之后允许在全部 `three` 上训练一次 final model，但不得再修改 feature、model、cap、confidence、threshold 或 tie-break。final train prediction不能重新参与 threshold选择。

### 6.4 明确禁止

- 使用 in-sample prediction选 gate；
- 对每 actor/timestamp/camera pair 使用不同 threshold；
- 根据 `dance/box` coverage、sign 或 gain 修改 gate；
- 运行多个 frozen版本后挑最好者；
- 在看到 grid 后添加恰好卡住样本的新 threshold；
- 只保存 winner、不保存完整 grid和失败项；
- 用 GT residual magnitude、GT error 或 oracle gain作为 runtime confidence。

## 7. 下一步执行顺序

1. 修正 invariant script 的 neutral、full-geometry selection 与 invariant tests。
2. 冻结 whitelist、四类 baseline、threshold grid 和所有 hash。
3. 只在 `three` 生成一次新的 invariant double-block OOF。
4. 一次性按本协议选择 candidate；若无 eligible candidate，记录 NO-GO。
5. 只有通过旧 `8% root + 5% layout`、新 safety gate、constant-baseline 和 proxy audit 后，才可以申请消耗一次 `dance/box` frozen evaluation。

当前步骤停在第 1 步。`dance/box` 继续保持 untouched。
