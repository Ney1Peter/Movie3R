# V14 B0 + DA3 精对齐最终方案与实验结论

日期：2026-07-30
结论状态：已经得到可作为主线使用、可运行、可回退、通过冻结集验证的精对齐方法。

## 1. 最终答案

现在已经有比较明确的答案：

> 保留并冻结 V14 `B0` 作为粗对齐；精对齐不再依靠 Human3R 自身的人体 root、torso
> 或 pointmap 去猜残差，而是引入冻结的 `DA3-Base`，用 cut 前最后一张 RGB 和 cut 后
> 第一张 RGB 在一个独立 shared space 中估计双向相对相机 pose。只提取其旋转与
> camera-baseline 方向，不使用 DA3 的任意尺度；保留 B0 的 metric baseline 长度，
> 再通过双向一致性和 B0 prior gate，只施加最多 3 度旋转、5 度方向的小残差。所有
> 证据不可靠时，逐元素返回原 B0。

正式名称可写为：

```text
B0-centered Bidirectional DA3 Shared-Pose Fine Alignment
简称：B0 + DA3 Safe / da3_safe
```

这不是“重新退回只用 B0”。B0 的职责仍然只是把两个 shot 放进近似同一个世界；
DA3 是在 B0 周围观测并修正最后几度、几十厘米相机误差的独立显式精对齐模块。

## 2. 为什么最终方案不是继续训练 B0

此前已经确认：V9 是从原始 Human3R 出发，在 AvatarReX、THuman、MVHuman 上经过充分
训练后得到的结果；它的隐式跨 shot 能力上限就是粗对齐。继续要求相同隐式表示一步
输出最终精确 Boundary，相当于重复已经充分验证过的路线。

现有残差审计也说明，B0 后的问题不能靠简单的人体连续性方程解决：

- B0 camera residual 仍约为数度和 0.2--0.3 m；
- translation 剩余量主要落在 Human3R camera-local depth 方向；
- Human3R 自身的 human root depth 同时带有局部重建 bias；
- 用 root continuity 修 camera，会把人体深度偏差吸收到整个世界 Boundary 中；
- 即使 oracle camera gauge 正确，Human3R local human 也不一定自动正确。

所以相机精对齐需要一个独立于 Human3R local-human bias 的 shared-space 观测。DA3
提供的正是这类证据。

## 3. 部署时的输入和输出

### 3.1 输入

精对齐模块只需要：

1. `pre_rgb`：cut 前最后一张完整 RGB；
2. `post_rgb`：cut 后第一张完整 RGB；
3. `C_pre`：Human3R 在 pre-shot world 中最后一帧 camera-to-world pose；
4. `C_raw`：cut 后 fresh-reset Human3R 第一帧的 camera-to-world pose；
5. `B0`：冻结粗对齐 Boundary，满足 `C_shadow = B0 @ C_raw`；
6. 冻结 `DA3-Base` 权重。

部署时明确不需要：

- GT camera、GT depth、GT identity、GT SMPL-X；
- future post-cut frame；
- 人工指定 camera pair 或数据源；
- per-human Boundary；
- DA3 translation 的绝对尺度。

### 3.2 输出

输出恰好是一个 `4 x 4` shared SE(3) Boundary：

```text
B_fine
```

它左乘到 post shot 的所有 world-space 内容：

```text
C_post_fine = B_fine @ C_post_raw
X_post_fine = R_fine @ X_post_raw + t_fine
```

同一个 `B_fine` 同时作用于：

- post-shot camera；
- Human3R pointmap / scene points；
- 所有人的 root、joints、vertices；
- 后续整段 post-shot trajectory。

不会为不同人估计不同变换，也不会改变 camera-local SMPL-X pose/shape。

## 4. 完整模型架构与数据流

```text
                         ┌─────────────────────────────┐
pre history + post RGB ─>│ frozen V9/V14 shadow branch │─> C_shadow
                         └─────────────────────────────┘
                                      │
post RGB + fresh state ───────────────>│ raw Human3R ──> C_raw + raw scene/humans
                                      │
                                      └─> B0 = C_shadow @ inv(C_raw)

pre RGB + post RGB ─> frozen DA3-Base forward [pre, post] ─> D_fwd_pre, D_fwd_post
post RGB + pre RGB ─> frozen DA3-Base reverse [post, pre] ─> D_rev_post, D_rev_pre
                                      │
                                      v
                   map both relative poses into C_pre world
                                      │
                   SO(3) consensus + baseline-direction consensus
                                      │
                   bidirectional agreement + B0-prior gate
                         ┌────────────┴────────────┐
                         │ accepted                │ rejected/invalid
                         v                         v
             cap rotation residual at 3°     return exact B0
             cap direction change at 5°
             retain B0 baseline length
                         │
                         v
                   one shared B_fine
                         │
        ┌────────────────┼──────────────────┐
        v                v                  v
      camera          pointmap           all humans
```

Shadow branch 仍然只负责产生 B0，state、human 和 pointmap 一律丢弃；post-shot 唯一被
提交的几何仍来自 fresh raw branch。

## 5. 几何原理

以下 pose 全部采用 camera-to-world 约定。

### 5.1 从 DA3 shared space 得到 Boundary proposal

对 forward 输入 `[pre, post]`，DA3 输出：

\[
D_p^f, D_q^f.
\]

DA3 world 到 Human3R pre world 的 gauge 为：

\[
G_f=C_{pre}(D_p^f)^{-1}.
\]

于是 DA3 建议的 post camera 是：

\[
\hat C_q^f=G_fD_q^f,
\]

对应 Boundary proposal：

\[
\hat B_f=\hat C_q^fC_{raw}^{-1}.
\]

reverse `[post, pre]` 完全独立再算一次 `B_r`。因为只使用相对 pose，这个构造对 DA3
自己的任意 world gauge 不变，相关单元测试已验证。

### 5.2 为什么丢弃 DA3 translation scale

DA3-Base 的 any-view reconstruction translation 只在其相对深度尺度中成立，不能直接
当作 Movie3R 的米制 baseline。直接拿该长度覆盖 B0，会重新引入 scale ambiguity。

因此这里只保留两件可观测量：

- proposal Boundary 的旋转；
- `C_pre -> desired post camera center` 的世界方向。

DA3 baseline magnitude 被明确丢弃。

### 5.3 forward/reverse 共识

两个旋转 proposal 用投影到 SO(3) 的 SVD 均值：

\[
R_c=\operatorname{Proj}_{SO(3)}(R_f+R_r).
\]

两个单位 baseline direction 归一化后求和：

\[
d_c=\frac{d_f+d_r}{\|d_f+d_r\|}.
\]

双向计算不是为了简单 ensemble，而是生成一个不依赖 GT 的失败检测器。顺序互换后
如果旋转或方向不能自洽，就没有理由相信这个显式 proposal。

### 5.4 冻结 acceptance gate

参数只在 `three` development split 选择，访问 `dance/box` 前已经冻结：

```text
forward/reverse rotation spread <= 5°
forward/reverse direction spread <= 5°
DA3 consensus vs B0 rotation <= 15°
DA3 consensus vs B0 direction <= 30°
forward/reverse pose finite and valid
B0 camera baseline non-degenerate
```

任意一项失败，返回逐元素相同的 B0。

### 5.5 B0-centered trust region

旋转 residual 按右乘形式计算：

\[
\Delta R=R_0^TR_c.
\]

将其 log-map rotvec 模长截断到 3 度：

\[
R^*=R_0\exp(\operatorname{clip}_{3^\circ}(\log \Delta R)).
\]

令 B0 对齐后的 raw camera center 为：

\[
c_0=R_0c_{raw}+t_0,
\]

B0 baseline 为：

\[
v_0=c_0-c_{pre},\quad l_0=\|v_0\|.
\]

从 `v0` 的方向朝 DA3 consensus `dc` 最多球面旋转 5 度，得到 `d*`，但长度始终保留
为 `l0`：

\[
c^*=c_{pre}+l_0d^*.
\]

最后根据 raw camera center 反求 Boundary translation：

\[
t^*=c^*-R^*c_{raw}.
\]

最终输出：

\[
B_{fine}=[R^*,t^*].
\]

## 6. 开发集与冻结协议

严格按以下顺序执行：

1. `three` 41 cuts：观察、选择 3°/5° cap 和 gate；
2. 冻结所有参数；
3. 一次性运行未触碰的 `dance` 61 cuts 与 `box` 78 cuts；
4. 冻结设置不变，再运行 AvatarReX/THuman/MVHuman 180-case source-diversity audit；
5. 没有根据 holdout 或四源结果修改阈值。

GT 只用于 output Boundary 已经产生后的评分；DA3 proposal、共识、gate 和 residual cap
都不读取 GT。

## 7. 最终实验结果

### 7.1 development：three，41 cuts

| 指标 | B0 | `da3_safe` | 结果 |
|---|---:|---:|---:|
| camera translation mean | 0.2558 m | 0.1948 m | -23.8% |
| camera rotation mean | 4.135° | 1.604° | -61.2% |
| camera composite mean | 0.3385 | 0.2269 | -33.0% |
| camera composite P95 | 0.5455 | 0.4702 | -13.8% |
| human root mean | 0.3152 m | 0.2658 m | -15.7% |
| human root P95 | 0.5351 m | 0.4169 m | -22.1% |
| catastrophic | 0/41 | 0/41 | 不增加 |

camera composite `41/41` 改善。

### 7.2 untouched frozen：dance + box，139 cuts

| 数据 | 方法 | Camera T | Camera R | Composite | Human root | Catastrophic |
|---|---|---:|---:|---:|---:|---:|
| dance, 61 | B0 | 0.2951 | 3.453° | 0.3641 | 0.3827 | 0 |
| dance, 61 | fine | 0.1773 | 1.239° | 0.2021 | 0.3256 | 0 |
| box, 78 | B0 | 0.2736 | 4.004° | 0.3537 | 0.5557 | 0 |
| box, 78 | fine | 0.1727 | 1.630° | 0.2053 | 0.4606 | 0 |
| combined, 139 | B0 | 0.2831 | 3.762° | 0.3583 | 0.4798 | 0 |
| combined, 139 | fine | 0.1747 | 1.458° | 0.2039 | 0.4014 | 0 |

冻结集关键结果：

- camera composite `139/139` 改善；
- combined composite 降低 `43.1%`；
- composite P95 `0.5775 -> 0.4399`；
- human root 平均降低 `16.3%`，`83.45%` case 改善；
- catastrophic 保持 `0/139`；
- dance 和 box 均独立大幅改善，不是由某一个 sequence 拉动均值。

### 7.3 MultiHuman 全部 180 cuts

将 development 与 frozen 仅用于汇总，不再选择参数：

| 指标 | B0 | Fine | 相对变化 |
|---|---:|---:|---:|
| camera translation mean | 0.2768 m | 0.1793 m | -35.2% |
| camera rotation mean | 3.847° | 1.491° | -61.2% |
| camera composite mean | 0.3538 | 0.2091 | -40.9% |
| camera composite P95 | 0.5767 | 0.4536 | -21.3% |
| human root mean | 0.4423 m | 0.3705 m | -16.2% |
| human root P95 | 1.2212 m | 1.1469 m | -6.1% |
| pairwise distance mean | 0.0728484 | 0.0728484 | rigid invariant |
| pairwise vector mean | 0.2684 | 0.2498 | -6.9% |
| catastrophic | 0/180 | 0/180 | 不增加 |

composite 在 `180/180` cuts 上改善。pairwise distance 的差只有约 `3e-10 m`，直接证明
代码没有对不同人做各自 fitting。

### 7.4 AvatarReX / THuman / MVHuman source-diversity，180 cuts

这一组实验的主要作用是检查 gate 和适用域。当前活动 V14 checkpoint 只在一个
AvatarReX event 上微调，不是广泛四源训练后的正式 B0，因此不能把其绝对 B0 指标当作
最终四源模型能力。

| 来源 | N | Gate 接受 | B0 composite | Fine composite | B0 head | Fine head |
|---|---:|---:|---:|---:|---:|---:|
| overall | 180 | 24.4% | 2.2875 | 2.2451 | 1.4555 | 1.4189 |
| AvatarReX | 48 | 10.4% | 2.4905 | 2.4794 | 1.1671 | 1.1640 |
| THuman | 48 | 81.2% | 0.8461 | 0.6980 | 0.7510 | 0.6167 |
| MVHuman100 | 48 | 0% | 2.9257 | 2.9257 | 2.4173 | 2.4173 |
| MVHuman200 | 36 | 0% | 3.0879 | 3.0879 | 1.4972 | 1.4972 |

这里最关键的不是 overall 小幅下降，而是 gate 行为：

- 接受 44 cases；这些 case 上 composite `0.5311 -> 0.3575`，降低 `32.7%`；
- 接受集 `42/44` composite 改善；
- human-head proxy `0.5885 -> 0.4386`，`39/44` 改善；
- 拒绝的 `136/136` cases 输出逐元素相同的 B0；
- overall catastrophic `107 -> 106`，没有扩散失败；
- MVHuman 当前 B0 rotation mean 为 `86°--90°`，根本不属于 fine residual，安全门全部拒绝。

这说明 `da3_safe` 能识别自己的可靠子集，但也说明正式四源系统仍需要一个真正广泛
训练的 B0。精对齐 trust region 不应被放大到几十度去替代失败的粗对齐，否则又会变成
一个无约束 full-Boundary estimator。

## 8. 已失败并排除的路线

### 8.1 直接人体 torso rotation

```text
camera R: 4.135° -> 7.059°
human root: 0.315 m -> 0.400 m
```

人体局部姿态、动作和预测误差会污染相机旋转，拒绝。

### 8.2 Human3R scene ICP / mutual NN translation

```text
ICP composite: 0.455
mutual translation composite: 0.396
improve rate: 5.9% / 3.3%
```

跨 shot pointmap 自身存在尺度、深度和重复结构 bias，最近邻不是可靠 correspondence。

### 8.3 SIFT Essential

无约束版本：

```text
camera R: 53.59°
composite: 1.318
human root: 1.868 m
```

有界 rotation 只有很小 camera gain，但 human root 变差；有界 translation direction
也没有超过 B0，均不足以成为主线。

### 8.4 root continuity / old multi-human Boundary / scale

- 直接 root translation 在 97.2% cuts 上恶化 camera composite；
- old uniform multi-human full Boundary composite `0.713`，明显差于 B0 `0.354`；
- shared scale 有少量均值收益但 tail 和跨源 cue 不稳定；
- token scale 读出了 capture identity，跨 sequence 不泛化。

这些失败共同指向同一个规律：不能让 Human3R 自己有 bias 的 local human/scene 直接决定
camera gauge；精对齐证据必须来自独立 shared-space prior，并且只能在 B0 周围小幅修正。

## 9. 与本地论文方法的关系

本地 `paper/` 中的 Multi-THuMBS、Trophies、UniCon3R 和 CUT3R 都强化了一个共同方向：
多视图或跨时间几何需要进入同一个 shared coordinate system，并通过独立的几何先验、
对应或统一表示来约束，而不能只依赖单帧人体 root 连续性。

本方案没有照搬某篇论文的完整系统，而是把这一原则落到 Movie3R 的特殊约束上：

- Human3R/V9 B0 保留其已经训练出的粗 shared gauge；
- DA3 只提供独立 relative pose evidence；
- B0 提供米制 baseline 长度；
- 双向一致性提供无需 GT 的 uncertainty proxy；
- trust region 防止 foundation pose prior 覆盖 Movie3R 已有 metric geometry。

## 10. 正式代码与调用方式

### 10.1 无评测依赖的 runtime

```text
versions/v14/b0_da3_fine_alignment.py
```

主要接口：

```python
from depth_anything_3.api import DepthAnything3
from versions.v14.b0_da3_fine_alignment import DA3FineAligner

da3 = DepthAnything3.from_pretrained(
    "/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/"
    "checkpoints/DAE-base"
).to("cuda:0").eval()

aligner = DA3FineAligner(da3, process_res=504)

B_fine, diagnostics = aligner.refine_images(
    b0=B0,
    pre_pose=C_pre,
    raw_post_pose=C_raw,
    pre_rgb=last_pre_rgb,
    post_rgb=first_post_rgb,
)
```

`diagnostics["accepted"]` 为 false 时，`B_fine` 就是 bit-exact `B0`。

如果 DA3 pose 已由统一服务预计算，可直接调用：

```python
refine_b0_with_da3(
    B0, C_pre, C_raw,
    forward_camera_to_world,
    reverse_camera_to_world,
)
```

### 10.2 实验代码与 artifacts

```text
versions/v14/probe_b0_da3_shared_pose.py
versions/v14/probe_b0_da3_four_source.py
tests/test_v14_da3_shared_pose.py

output/v14/fine_alignment_research/da3_shared_pose_three_dev/
output/v14/fine_alignment_research/da3_shared_pose_dance_box_frozen/
output/v14/fine_alignment_research/da3_shared_pose_four_source_frozen/
```

### 10.3 验证状态

正式 runtime 使用缓存的 180 个 DA3 forward/reverse pose 重放，与原冻结 probe 的 Boundary
最大绝对差：

```text
1.11e-16
```

测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_v14_da3_shared_pose.py \
  tests/test_v14_b0_identity_matching.py \
  tests/test_v14_segment_boundary.py
```

结果：`21 passed`。

覆盖内容包括：

- DA3 world gauge invariance；
- forward/reverse pose mapping；
- SO(3)/direction consensus；
- 3°/5° cap；
- B0 baseline-length preservation；
- camera/points 使用同一 Boundary；
- missing、NaN、冲突、退化 baseline、DA3 runtime exception 的 bit-exact B0 fallback。

## 11. 运行代价

使用 DA3-Base（约 0.12B 参数）、`process_res=504`，NVIDIA L20 上 warm-up 后：

```text
forward mean: 0.086 s
reverse mean: 0.083 s
total extra latency: about 0.17 s per cut
```

它只在 cut 时运行两次，不是每个视频帧都运行。进一步工程化可以把两个顺序组成 batch，
或缓存 pair encoding，但当前结果已经满足离线/准在线 Movie3R cut 精对齐的可落地性。

## 12. 方法适用域与未解决问题

### 已解决

- 合格 B0 后残留的 shared camera rotation 和 baseline-direction 误差；
- first-post-frame causal refinement；
- shared camera/scene/human rigid propagation；
- 无 GT 的异常检测与精确回退；
- MultiHuman three/dance/box 上稳定泛化。

### 尚未由该模块解决

1. **B0 自身完全失败。** 3°/5° fine stage 不应修 90° 粗对齐错误。需要把 V9/V14 B0
   在 AvatarReX、THuman、MVHuman 上做正式 broad event training，而不是放大 fine cap。
2. **Human3R camera-local human depth bias。** shared camera Boundary 改善 root 均值，但
   不能单独消除每个 local reconstruction 的人体深度 bias；这应作为独立 local-human
   calibration 问题研究，不能写回 shared Boundary。
3. **DA3 无共同视觉证据。** 极低 overlap、严重遮挡、纹理缺失或 dynamic-human 主导时，
   forward/reverse 可能不一致；当前正确行为是回退 B0。
4. **四源实验不是最终 held-out B0 benchmark。** 它验证 gate 和 source shift，当前
   checkpoint 的 broad B0 能力不足；正式训练后的 capture-disjoint 四源结果仍需补跑。
5. **自动 cut / 人数变化 / identity dustbin** 属于上游事件检测和下游匹配问题，不改变
   本精对齐模块的 shared Boundary 定义。

## 13. 后续主线建议

优先级已经很明确：

1. 将 `DA3FineAligner.refine_images` 接入真实 cut runtime，Boundary commit 前运行；
2. 固定本文件的 gate 和 cap，不再在 dance/box 上调参；
3. 用 broad AvatarReX + THuman + MVHuman event 数据正式训练/恢复合格 B0，然后原样重跑
   四源 frozen evaluation；
4. 在 automatic-ID、人数变化、多 cut 长序列上验证同一个 Boundary 的传播稳定性；
5. 把 Human3R local-human root depth calibration 作为独立模块，不允许它反向改写 camera
   Boundary；
6. 做 DA3 两方向 batching、显存和端到端 latency 工程优化。

最重要的研究判断已经完成：

```text
B0 不是最终答案，但它是正确的粗对齐底座；
DA3 shared-pose 不是替代 B0，而是在独立坐标系中观测 B0 剩余的小残差；
双向共识 + trust region + exact fallback 使它能够真正成为主线精对齐器。
```
