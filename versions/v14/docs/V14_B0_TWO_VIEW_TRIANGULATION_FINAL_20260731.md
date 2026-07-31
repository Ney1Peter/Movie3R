# V14 冻结 B0 后的人体精对齐：两视图射线三角化 + 多人布局共识

日期：2026-07-31

状态：**已得到第一个通过开发集与新确认集的 root/layout 精对齐主线；尚未完成 Multi-THuMBS 官方协议对榜。**

## 1. 最终结论

目前已经有一个明确、可实现、效果显著的方法：

```text
Camera-Frozen Boundary Ray Triangulation with Layout Consensus
简称：BRTC-LC

冻结 B0 相机
-> 自动匹配切镜前后的人
-> 用最后一帧 pre 与第一帧 post 的 5 个躯干关节视线做两视图三角化
-> 得到每个人的显式深度/位置修正
-> 把多人位移分成“全组共享平移 + 个体残差”
-> 用切镜前的预测多人布局选择个体残差强度
-> 只刚性平移 post 人体；相机、场景、姿态和形状不变
-> 证据不可靠的人精确回退 B0
```

它解决的是当前最主要的 **camera-local 人体 root/深度和多人布局** 问题。它不是重新训练
B0，也不是再预测一个相机 Boundary，更不是让 DA3 给人体表面深度。它把“相机已经比较准”
直接转化成跨镜头的多视图几何约束。

在新的 `three offset1` 确认集上，固定策略得到：

| 指标 | B0 | BRTC-LC | 相对改善 |
|---|---:|---:|---:|
| Root | 0.3779 m | **0.2314 m** | **38.8%** |
| World joint | 0.4117 m | **0.2745 m** | **33.3%** |
| World vertex | 0.3891 m | **0.2525 m** | **35.1%** |
| Pairwise distance | 0.1341 m | **0.0984 m** | **26.7%** |
| Pairwise vector | 0.3297 m | **0.2588 m** | **21.5%** |

其他安全指标：

- 42 个 cut、125 人；
- 覆盖率 `88.0%`；
- accepted residual 符号正确率 `87.3%`；
- root improve rate `67.2%`；
- root 恶化超过 5 cm 的比例 `7.2%`；
- 自动 `root+torso+joints` Hungarian 匹配准确率 `100%`；
- 相机最大数值改动 `0.0`。

因此，它超过了预先使用的主线门槛：

```text
root gain >= 8%
layout gain >= 5%
harm >5cm <= 10%
coverage >= 20%
camera bit-exact
```

## 2. 为什么这个方法与当前问题匹配

前面的 GT 可视化和 oracle 分解已经证明：

```text
相机/shot 坐标系对齐正确
不等于
人体相对相机的 root 深度正确
```

冻结相机后，剩余 root 误差有很大一部分沿人体观察射线。GT-ray oracle 能把现有约
`0.3821 m` 的 root 降到约 `0.1600 m`。真正缺失的不是另一个 shared camera SE(3)，而是
逐人的 signed depth evidence。

切镜前后相机虽然不同，但只要 B0 给出了足够准确的相对相机位置，同一人的同一躯干关节
在两幅图中就对应两条世界射线。两条射线的交会位置直接提供深度，不需要猜尺度先验，也不
需要把上一帧的 3D root 硬拷到下一帧。

这也是 BRTC-LC 比历史 root anchor 更强的根本原因：

- history anchor 使用上一段 Human3R 自己可能有偏差的 3D root；
- BRTC-LC 使用两幅图中的角度观测和已冻结的相机基线重新解深度；
- 多个躯干关节提供冗余，ray gap、parallax 和 joint MAD 提供可观察的可靠性判断。

## 3. 输入、模块和输出

### 3.1 输入

部署时只需要：

1. 最后一张切镜前图像对应的 Human3R 人体；
2. 第一张切镜后图像对应的 raw-reset Human3R 人体；
3. 已冻结的 B0，用于把 post camera/person 放到 pre 世界坐标；
4. 已有关联模块输出的一对一 pre/post 人物匹配；
5. 每个人的 pelvis、hips、shoulders 五个 SMPL-X 关节世界坐标。

不用：

- GT camera；
- GT identity；
- GT depth/root；
- DA3；
- segmentation GT；
- 未来 post 帧；
- 离线全序列优化。

### 3.2 自动人物匹配

先用 B0 把 post 的匿名人体放到 pre 坐标系，再用：

```text
root distance + torso orientation + centred joint geometry
```

构造 Hungarian cost。匹配只决定哪一个 pre 人和哪一个 post detection 构成 ray pair，
不参与深度数值拟合。

在新确认集 42 个 cut 上，`root+torso+joints` 匹配为 `125/125` 正确；因此报告的确认集
结果已经是自动匹配路径，不是 GT-ID 数字替代。

### 3.3 五关节射线三角化

对每个已匹配的人和每个 core joint：

```text
pre camera centre  C_a
pre predicted joint J_a -> ray d_a = normalize(J_a - C_a)

post camera centre C_b
post predicted joint J_b -> ray d_b = normalize(J_b - C_b)
```

求两条空间直线的 closest points：

```text
C_a + s_a d_a
C_b + s_b d_b
```

两点中点是该关节的三角化位置。再减去 post mesh 中“该关节相对 pelvis”的向量，得到一个
post pelvis 候选。最后只取候选相对当前 pelvis 在当前 pelvis ray 上的 signed residual。

五个关节的 residual 用 median 汇聚，避免单个关节姿态或检测噪声控制结果。

### 3.4 完全可观察的 gate

冻结策略为：

```text
joint set              = pelvis + left/right hip + left/right shoulder
minimum valid joints   = 1
median ray gap         <= 0.20 m
joint residual MAD     <= 0.40 m
median parallax sine   >= 0.025
residual cap           = +/-2.0 m
```

所有 gate 都来自预测相机和预测人体；GT 不进入候选或 gate。未通过的 person 保持 bit-exact
B0 geometry。

### 3.5 多人布局共识

最初的逐人独立版本虽然能显著降低 root，但在 `dance/box` 首次冻结评测中破坏了布局。这是
本轮最关键的失败记录：

| 冻结 `dance+box` 独立修正 | B0 | 逐人三角化 | 变化 |
|---|---:|---:|---:|
| Root | 0.4798 m | 0.2912 m | 改善 39.3% |
| Pairwise vector | 0.3088 m | 0.3894 m | **恶化 26.1%** |

原因是每个人的三角化方向大体正确，但个体噪声不同。于是最终方法把位移写成：

```text
s_i = g + lambda * (s_i_individual - g)
```

其中：

- `g` 是同一 cut 内所有 accepted individual shifts 的逐坐标 median；
- `lambda` 从固定集合 `{0, 0.25, 0.5, 0.75, 1}` 选择；
- 选择标准是：修正后的 post pairwise root vectors 与最后一帧 pre 预测布局的差最小；
- 这个选择只看预测布局，不看 GT；
- rejected person 仍然严格零位移，不会被同组其他人带着移动。

这一步同时保留了三角化的共同深度信号，并抑制了会破坏多人相对位置的个体噪声。

### 3.6 输出

输出包括：

- 原 B0 camera：数值完全不变；
- 原场景 pointmap：不变；
- 每个 accepted post person：root/joints/vertices 加同一个刚性 shift；
- pose、shape、global orientation：不变；
- rejected/unmatched post person：bit-exact 原值；
- debug：ray gap、parallax、MAD、individual shift、group shift、lambda 和 gate 状态。

## 4. 实验演化与完整结果

### 4.1 Exact-camera 机制验证

先在四来源 controlled archive 上用 exact camera 隔离“人体机制是否成立”。

| Split | N | B0/raw root | 两视图三角化 | Gain | Coverage | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| offset0 dev | 200 | 0.8689 | 0.0871 | 90.0% | 100% | 2.0% |
| offset50 confirm | 200 | 0.8991 | 0.0978 | 89.1% | 100% | 3.5% |

这证明两视图关节射线确实包含强 signed root-depth signal。不过 offset0/50 共享 actor、资产和
部分图像，所以这一步只能叫 controlled mechanism confirmation，不能叫严格跨资产泛化。

### 4.2 真实 B0 开发集

在 `three offset0` 41 cut / 122 人上：

| 指标 | B0 | 独立三角化 | 布局共识版 |
|---|---:|---:|---:|
| Root | 0.3789 | 0.2331 | **0.2251** |
| World joint | 0.4138 | 0.2760 | **0.2704** |
| World vertex | 0.3913 | 0.2550 | **0.2486** |
| Pairwise vector | 0.3331 | 0.2933 | **0.2605** |

布局共识版：

- root gain `40.6%`；
- layout-vector gain `21.8%`；
- coverage `88.5%`；
- harm >5 cm `6.6%`；
- accepted sign accuracy `87.0%`。

### 4.3 新确认集

在布局共识规则、gate 和代码冻结后，重新运行 Human3R 得到之前未做过该 depth/layout 评测的
`three offset1`：pre 到 frame `t`，post 第一帧为 `t+1`。

确认结果：

| 指标 | B0 mean | BRTC-LC mean | B0 P95 | BRTC-LC P95 |
|---|---:|---:|---:|---:|
| Root | 0.3779 | **0.2314** | 0.8549 | **0.6867** |
| World joint | 0.4117 | **0.2745** | 0.8796 | **0.6663** |
| World vertex | 0.3891 | **0.2525** | 0.9196 | **0.6568** |
| Pairwise distance | 0.1341 | **0.0984** | 0.4275 | **0.3001** |
| Pairwise vector | 0.3297 | **0.2588** | 0.9613 | **0.9332** |

开发集与确认集结果高度一致，说明主要增益不是偶然阈值命中。

### 4.4 `dance/box` 的正确解释

`dance/box` 已被独立逐人版本消费并暴露 layout failure。因此布局共识版在相同数据上的结果
只能叫 **post-hoc support**，不能冒充第二次 pristine frozen evaluation。

| Post-hoc support | B0 root | BRTC-LC root | Root gain | B0 layout vector | BRTC-LC | Layout gain |
|---|---:|---:|---:|---:|---:|---:|
| dance | 0.3827 | **0.1251** | 67.3% | 0.1042 | **0.0783** | 24.8% |
| box | 0.5557 | **0.3723** | 33.0% | 0.4688 | **0.4273** | 8.8% |
| combined | 0.4798 | **0.2639** | 45.0% | 0.3088 | **0.2742** | 11.2% |

combined coverage `98.9%`，harm >5 cm `5.0%`。这些数字说明布局共识确实修复了第一次失败所
揭示的问题，但正式论文结论仍应以新 offset1 确认为主。

## 5. 实现与验证

部署模块：

```text
versions/v14/b0_person_triangulation.py
```

核心 API：

```python
refine_matched_people(
    pre_camera,
    post_camera,
    pre_people,
    post_people,
    matches,
)
```

这个文件不 import dataset、GT、evaluator 或 DA3。`matches` 是匿名数组索引对，不要求 GT
identity 名称。

测试：

```text
versions/v14/tests/test_b0_person_triangulation.py
4 passed
```

测试覆盖：

- 已知射线交点；
- single-person depth correction；
- camera 不被修改；
- rejected 和 unmatched bit-exact fallback；
- common world gauge equivariance。

部署模块与 42-cut 确认 probe 的 final shift 数值一致：

```text
max absolute shift difference = 1.10e-15 m
```

## 6. 与 Multi-THuMBS 的关系

Multi-THuMBS 的 Table 3 给出了非常支持当前方向的证据：去掉 boundary human-scene
alignment 后，EgoHumans W-MPJPE 从 `278.8` 恶化到 `882.7`。这说明仅相机统一不够，逐人
human-scene alignment 是核心。

论文的 EgoHumans 参考线是：

| W-MPJPE | WA-MPJPE | MPJPE | MPVPE | Accel | ATE | IDs |
|---:|---:|---:|---:|---:|---:|---:|
| 279.0 | 166.0 | 228.3 | 262.2 | 27.3 | 0.7 | 0.97 |

BRTC-LC 和 Multi-THuMBS 方向相同，但实现范式不同：

- Multi-THuMBS：全序列/离线、逐人 2D joint + silhouette + scene depth 优化，并有后处理；
- BRTC-LC：严格 boundary 两帧、显式三角化、一次前向几何求解、相机冻结、精确 fallback。

目前**不能宣称已经正式打过 Multi-THuMBS**，原因不是本方法没有效果，而是：

1. 论文主文没有公开完整 W/WA/MPJPE/MPVPE/Accel/ATE/IDs 公式与 aggregation；
2. 本地只有 EgoHumans `001_legoassemble` 的自建短 chain，不是论文官方 split；
3. 当前 BRTC-LC 报告的是 fixed-world root/joint/vertex 与 pairwise layout，不等同论文
   pelvis-aligned MPJPE/MPVPE；
4. BRTC-LC 尚未在本地 EgoHumans chain 上按 provisional evaluator 完整重跑；
5. Harmony4D Table 1 的 MPVPE 真正最佳线是 HSfM† `257.6`，不是 Multi-THuMBS 的
   `278.3`。

完整论文数字与协议限制见：

```text
versions/v14/docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md
```

## 7. 现在有多明确的答案

### 已经明确

1. B0 应冻结为 camera coarse alignment，不应回退到“让 B0 一步修好人体”。
2. DA3 person surface depth、CUT3R virtual memory 和纯 invariant hand-crafted feature 都不是
   当前主线。
3. 最有效的 signed evidence 是：**冻结 B0 后同一人物跨 boundary 的多关节两视图射线**。
4. 多人不能完全独立修正；必须保留共享 group shift，并用 pre layout 约束个体残差。
5. BRTC-LC 已在开发和新确认集上显著超过 root/layout 门槛，并有 GT-free runtime。

### 仍未明确或尚未完成

1. 跨 capture、跨数据集、鱼眼 EgoHumans 的严格泛化；
2. 人员进出、遮挡、只有一个人和错误匹配时的完整 dustbin/uncertainty policy；
3. pelvis-aligned 内部 pose/shape error：刚性 root 平移不会修复错误肢体结构；
4. global orientation 的精修；
5. Multi-THuMBS 官方 split 和官方 evaluator 下的胜负；
6. 将 runtime callable 接入最终 demo/stream commit 的工程 wiring。

所以最准确的项目结论是：

> **root/layout 精对齐主线已经找到，可以停止继续盲试 DA3 depth 或纯 token proxy；完整人体精对齐与 Multi-THuMBS 对榜还需要在这条主线上继续完成 orientation、内部 pose 和官方协议验证。**

## 8. 下一步优先级

### P0：接入正式流式路径

把 `refine_matched_people` 接到：

```text
B0 commit
-> automatic association
-> BRTC-LC
-> corrected post humans commit
```

并导出原 demo 风格多人可视化，检查 rejected/unmatched/camera parity。

### P1：真正跨域确认

- 新 capture 或 actor/camera-disjoint synthetic split；
- EgoHumans fisheye 的明确 undistortion/intrinsics protocol；
- 不再使用已经被消费的 `dance/box` 调参数。

### P2：补 global orientation

在固定 root 和 camera 后，用同一套两视图 core joints 增加小角度 global orientation solver；
仍采用 bounded residual、cue agreement 和 exact fallback。

### P3：Multi-THuMBS 指标闭环

- 对 B0 和 B0+BRTC-LC 输出连续 track；
- 计算 provisional W/WA、world root、pelvis-aligned MPJPE/MPVPE、Accel、ATE、IDs；
- 获取 supplementary/evaluator 后再做官方结论；
- 目标不仅是 `<279.0` W-MPJPE，还要确保 pose、Accel、ATE 和 IDs 不退化。

## 9. 复现入口

```bash
# controlled exact-camera mechanism
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  versions/v14/probe_two_view_person_triangulation.py --phase dev --device cuda:0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  versions/v14/probe_two_view_person_triangulation.py --phase confirm --device cuda:0

# B0 development
.venv/bin/python versions/v14/probe_b0_two_view_person_triangulation.py --phase dev

# fresh offset1 B0 + automatic matching
CUDA_VISIBLE_DEVICES=0 .venv/bin/python versions/v14/probe_b0_identity_matching.py \
  --sequence three --offsets 1 --device cuda:0 \
  --output_dir output/v14/b0_identity_matching_offset1_confirm

# locked automatic-ID confirmation
.venv/bin/python versions/v14/probe_b0_two_view_person_triangulation.py --phase confirm

# runtime unit tests
.venv/bin/pytest -q versions/v14/tests/test_b0_person_triangulation.py
```

主要结果：

```text
output/v14/fine_alignment_research/two_view_person_triangulation/
output/v14/fine_alignment_research/b0_two_view_person_triangulation/
output/v14/b0_identity_matching_offset1_confirm/
```
