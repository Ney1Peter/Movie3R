# V14 Controlled Human-Token 标签与 held-out 协议审计

> 日期：2026-07-31
> 审计对象：`versions/v14/probe_controlled_token_person_residual.py` 及其已有产物
> 审计方式：只读检查 archive、manifest、源码和结果；未修改 controlled-token data agent 脚本，未读取 `dance/box`。

## 1. 审计结论

结论分成两层：

```text
标签内部几何/anchor 语义：PASS
direct GT convention 独立对照：CONDITIONAL / 尚未完成
作为“可泛化精对齐主线”的 held-out 证据：FAIL / 尚未成立
```

当前标签已经把预测和 GT 放在同一个目标相机坐标系，并且两端都使用 SMPL-X joint 0
pelvis。`boundary=2`、因果帧语义和 camera isolation 也正确。现有标签不再混用 torso
centroid 与 GT pelvis。

但是 offset0 与 offset50 只是 manifest row offset，不是 actor、asset、sequence 或
camera-pair 隔离的 split。两边共享全部 15 个 `source::group`、140 个 individual
sequence，并有 20 个 unordered camera pair 和 5 个精确 RGB observation 重合。四个 source
的标签均值也差异巨大。因此当前 Ridge 的高相关性主要证明 frozen Human3R token 能读取
这些受控数据域中的绝对 camera-local 深度偏差；它还不能证明学到了一个对新人物、新资产、
新场景都成立的 universal cut residual。

## 2. 标签到底在计算什么

每条 archive sample 有 4 帧：

```text
frame 0,1 = pre-shot / seqA history
frame 2   = first post-shot / seqB camera，当前 boundary frame
frame 3   = post-shot future，本标签和 token feature 不使用
```

在 frame 2：

```python
pred_root_local = predicted_smplx_joint0_in_predicted_camera
gt_root_local   = world_to_gt_target_camera @ gt_smplx_joint0_world
ray_local       = normalize(pred_root_local)

label = dot(gt_root_local - pred_root_local, ray_local)
```

允许的 correction 只有：

```python
corrected_root_local = pred_root_local + clip(predicted_label) * ray_local
```

因此它只移动当前 person，不更新 camera，不改变 body pose/shape，也不允许横向 correction。

### 2.1 与 `Tcam` 形式的等价性

若把 Human3R 的预测 camera-local root 先放回预测 world，再用

```text
Tcam = target_c2w @ inverse(pred_c2w)
```

映射到目标 world，结果等价于直接使用 target camera local 坐标：

```text
pred world -> pred camera -> target world
```

脚本现采用 local 形式，避免显式依赖 predicted camera pose；最终 scalar 对共同 target c2w
变换保持不变。已有数值检查的 local/world label 最大差为：

| Split | 最大绝对差 |
|---|---:|
| offset0 | `2.44e-6 m` |
| offset50 | `2.87e-6 m` |

这是 GT camera 文件 float32 精度下的合理误差，camera isolation 通过。

## 3. Joint topology 与 anchor 审计

预测端不是简单读取 `smpl_transl` 当 pelvis，而是按 Human3R 的 head-centered SMPL-X
定义重建 joints：

```text
SMPL-X neutral body
-> 围绕 pelvis 应用 predicted global orientation
-> 减 predicted head joint
-> 加 Human3R head-centered translation
-> 读取 SMPL-X joint 0 pelvis
```

GT 端读取相同定义的 SMPL-X pelvis：

- 有 `smplx_body25_world` 时，将 Body25 pelvis id 8 映射到 SMPL-X pelvis；
- 否则从 GT SMPL-X 参数重新前向并读取 joint 0；
- 再用相应 target camera 的 w2c 变为 camera-local。

主标签两端因此是 homologous pelvis-to-pelvis。pelvis+hips+shoulders 的 torso-5 centroid
只作为敏感性审计，不进入训练 target。结果也支持 anchor 语义稳定：

| Split | pelvis/torso label Pearson | sign agreement |
|---|---:|---:|
| offset0 | `0.99987` | `100%` |
| offset50 | `0.99991` | `99%` |

结论：旧方案中“pred torso centroid 减 GT pelvis”会混入骨架 offset 的风险，在当前脚本中
已经消除。

## 4. Boundary 与 feature 因果性

archive `run_args.json` 均满足：

```text
boundary = 2
offset0 source_offset = 0
offset50 source_offset = 50
```

token feature 为：

```python
history = mean(token[0:2])
current = token[2]
feature = concat(history, current, current-history, abs(current-history))
```

三类候选 feature 都只读 frozen raw token：

- `human_token_out`；
- `human_token_in`；
- `human_token_out - human_token_in`。

没有读取旧 learned anchor weights、gate、checkpoint prediction 或 frame 3。因此单 boundary
因果性和 raw-token 合同通过。

## 5. GT convention 仍需独立确认的硬条件

当前 local/world invariance 只能证明数学实现内部一致，不能证明输入 camera metadata 本身
一定采用正确 convention。正式使用前仍应抽样把 direct extractor 与旧
`extract_gt_world()` 路径对比，至少确认：

- `pose` / AvatarReX raw calibration 是 c2w，不是 w2c；
- camera axis convention 与 Human3R 数据处理一致；
- `smplx_body25_world`、SMPL-X parameter fallback 和 camera translation 单位均为米；
- `smplx_world_scale` 在 precomputed 与 fallback 两条路径中没有漏乘或重复乘；
- direct GT local pelvis、旧 loader GT pelvis 和最终 scalar label 在 float tolerance 内一致。

这个检查尚未由当前产物证明。若不通过，现有所有绝对 root 数字都必须作废；通过后才可把
标签几何升级为完全确认。

## 6. offset0 / offset50 split 重合审计

每个 split 各 200 records。重合统计如下：

| 层级 | offset0 unique | offset50 unique | intersection | 结论 |
|---|---:|---:|---:|---|
| exact `(source,group,seqA,seqB,start)` | 200 | 200 | 0 | 没有 exact record 重复 |
| unordered sequence pair | 195 | 190 | 20 | camera/sequence pair 有重合 |
| individual sequence | 227 | 220 | 140 | 大量相同 sequence/asset |
| `source::group` | 15 | 15 | 15 | actor/group 100% 重合 |
| exact RGB `(source,seq,frame)` | 800 | 800 | 5 | 有精确图像 observation 重合 |
| motion `(source,seqA,frame)` | 800 | 798 | 0 | seqA motion frame 无精确重合 |

单个 archive 内没有 exact record duplicate，但有相同 unordered pair 的不同 start：

| Split | repeated pair keys | extra rows |
|---|---:|---:|
| offset0 | 5 | 5 |
| offset50 | 10 | 10 |

因此 offset50 最多可称：

```text
record/start-frame offset holdout
```

不能称：

```text
actor-disjoint / asset-disjoint / sequence-disjoint / camera-pair-disjoint held-out
```

另外，offset50 label marginals 在冻结最终 prediction 之前已为语义审计查看过。现有代码确实
记录了这项 disclosure，并声明没有 offset100 archive。因此当前结果不是 pristine first
reveal。

## 7. 标签分布与 source bias

exact-camera pelvis root error 与沿 ray label：

| Source | offset0 root mean | offset0 label mean | offset50 root mean | offset50 label mean |
|---|---:|---:|---:|---:|
| AvatarReX | 0.6812 m | -0.6791 m | 0.6896 m | -0.6877 m |
| MVHuman100 | 1.7224 m | -1.7216 m | 1.8415 m | -1.8408 m |
| MVHuman200 | 0.9179 m | -0.9147 m | 0.8779 m | -0.8743 m |
| THuman | 0.1541 m | -0.0066 m | 0.1874 m | +0.0591 m |

总体 tangential floor 只有 `0.0473 / 0.0487 m`，说明这批受控 synthetic label 的误差几乎
完全沿相机 ray；但四个 source 的 bias 形态完全不同。尤其前三个 source 几乎总是负 label，
THuman 则接近零且正负混合。

这带来两个解释：

1. 积极解释：Human3R token 的确包含预测人体绝对深度/尺度信息；
2. 风险解释：模型只需从 token 识别 source/domain，就能输出一个 source-specific 常数。

第二种解释不能靠当前 offset50 排除，因为两边共享全部 actor/group 和大量 sequence。

## 8. 已有模型结果应如何解释

offset0 GroupKFold 选择：

```text
ridge::human_out_pair::pca64::alpha0.1
```

冻结 policy：

```text
accept if abs(raw) >= 0.30 m
clip correction to +/-0.10 m
```

offset50 结果：

| Method | Root mean | Relative gain | Improve | Harm >5 cm | Corr | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| no-op | 0.8991 m | 0.0% | 0.0% | 0.0% | n/a | 100% |
| train mean | 0.5658 m | 37.1% | 74.0% | 25.5% | n/a | 100% |
| raw Ridge | 0.1482 m | 83.5% | 87.5% | 9.0% | 0.973 | 100% |
| locked capped policy | 0.8221 m | 8.6% | 78.5% | 0.5% | 0.664 | 79.5% |

现有 `RESULTS.md` 按预设 provisional gate 把 locked policy 标为 PASS，这个内部判断本身没有
计算错误。但它只能解释为：

```text
PASS：受控、共享 domain 的 record-offset split 上，token correction 有低风险改善。
```

不能解释为：

```text
PASS：已经找到真实多人、真实跨 shot、跨 actor/asset 泛化的精对齐方法。
```

尤其 `train mean` 不看 token 就有 37.1% gain，进一步说明 source/absolute bias 是主要信号。
locked policy 的 8.6% gain 主要来自对前三个 source 施加统一 `-0.1 m` 小步修正；这是一条
有用的安全性证据，但离 main-line fine alignment 仍有明显距离。

## 9. 最终判定与下一步

### 已通过

- boundary frame 和 causal token feature 正确；
- camera 与 person correction 隔离；
- pred/GT 使用 homologous SMPL-X pelvis；
- local/world scalar 数值一致；
- raw token Ridge 在当前 controlled split 有强相关性；
- capped policy 将 >5 cm harm 压到 0.5%。

### 尚未通过

- direct GT extractor 与旧 dataset loader 的独立 convention 对照；
- actor/asset/sequence-disjoint held-out；
- source-balanced residual baseline；
- 在真实 B0 post-cut output 上验证，而不是 synthetic controlled local reset；
- 多人 identity、遮挡、漏检和 pairwise layout；
- correction 后 world joint/vertex，而不只是 pelvis root；
- 与 B0 no-op、常数 correction、source-only oracle 的公平比较。

### 建议的最小确认实验

1. 先抽样验证 direct GT vs 旧 loader，未通过前停止使用标签结果。
2. 重新按 actor/asset/sequence 建 split；同一 `source::group` 和 individual sequence 只能出现
   在一个 split。
3. 同时报告 no-op、global constant、source-only constant、token Ridge；token 必须显著超过
   source-only baseline 才证明 token 提供 sample-specific signal。
4. 冻结该模型后只在一次真实 B0 cut benchmark 上揭盲，camera bit-exact 不变，报告
   root/joint/vertex、harm rate 和按 source/人物分层结果。

在完成以上四项前，controlled-token 方法应保留为“有潜力的受控证据”，不应替代当前
Person-Conditioned Boundary Scene Registration 主线。
