# V14.1 Shot-Aware Modality-Selective State Routing

## Question

Camera cut 后，Human3R 是否应该继续让 scene、camera 和 human 使用同一个
历史 state？V14.1 将 Boundary 对齐固定不变，只比较 state read/write 和人体
memory commit 规则。

## Streaming Design

Cut 前旧 scene/camera state 在 cut 到达时冻结。第一张 post-cut frame 在解码前
建立 fresh recurrent state，因此它不会先读取一次旧镜头再 reset。

实现同时通过了 restart-equivalence sanity check：将完整序列在 cut 处 pre-decode
reset，与只把 post-cut 子序列单独启动 Human3R 比较，cut 第一帧的 camera、
pointmap、SMPL-X shape、pose 和 translation 最大差异均为 `0`。

信息路由如下：

```text
scene/camera: read fresh state, write fresh state
human:       read fresh state + isolated human memory
old state:   frozen and isolated
world root:  Boundary SE(3) first, long-term commit second
```

Human3R 主干和全部权重冻结。实验使用 GT cut index；每个方法使用完全相同的
GT Boundary 或 Fixed Explicit Boundary。Human memory 不修改 camera token、
image token、pointmap、recurrent scene state 或 pose retriever memory。

## Compared Variants

- Original Continue：继续读写旧 state。
- Full Old-State Read：fresh branch 仍读取完整旧 state 的负面对照。
- Hard Reset：scene、camera、human 历史全部重置。
- Hard Reset + Tracklet：只保留原生 tracklet。
- Token Only：只混合 raw human token。
- Shape Memory：保留 SMPL-X beta/body scale。
- Shape + Local Pose：额外保留 root-centered local rotations。
- Immediate Commit：未对齐的 local world root 立即写回。
- Align-Then-Commit：应用 Boundary SE(3) 后再写回。
- Verify-Then-Commit：再用固定质量阈值筛选。
- Zero/Wrong-Video Memory：因果负面对照。

当前四源 loader 使用 `max_humans=1`。因此 track ID 数值只验证代码通路，不能
作为多人 IDF1、ID switch 或 Re-ID 的正式结论。

## Main Single-Cut Result

16 个真实 cross-camera cuts，覆盖 AvatarReX、THuman、MVHuman100 和
MVHuman200。主报告：

```text
output/v14_1_shot_aware_state_routing/stage3_true_reset/
```

Scene/camera isolation：

| Method | Camera max diff vs Hard Reset | Pointmap max diff |
|---|---:|---:|
| Selective Routing | 0 | 0 |
| Original Continue | 1.870 | 2.229 |
| Full Old-State Read | 1.870 | 2.229 |

这说明旧 state 的错误读取本身就足以重新引入 camera/scene 污染；人体记忆隔离
后可以保持 Hard Reset 的 scene/camera 输出不变。

人体信息消融：

| Method | Shape jump | Scale jump | Local-pose boundary residual |
|---|---:|---:|---:|
| Hard Reset | 0.745 | 0.0074 | 4.72 deg |
| Token Only | 0.742 | 0.0073 | 4.72 deg |
| Shape Memory, alpha=0.75 | 0.259 | 0.0032 | 4.72 deg |
| Shape + Local Pose | 0.259 | 0.0031 | 3.88 deg |

Raw human token 单独几乎没有收益。真正有效的是具有明确含义的 canonical
shape 和 root-independent local pose。

Shape jump 在 16/16 样本下降，local-pose boundary residual 在 15/16 样本
下降。四个来源的 shape jump 均为同方向改善：

```text
AvatarReX:   0.606 -> 0.165
MVHuman100:  1.010 -> 0.356
MVHuman200:  0.461 -> 0.225
THuman:      0.904 -> 0.289
```

## Continuity Is Not Accuracy

强 shape locking 可以稳定一个错误 shape。`shape_alpha=0.75` 的总体 GT beta
误差由 `1.557` 变为 `1.575`，其中 THuman 由 `1.698` 变为 `1.990`。

因此使用独立开发样本扫描 `0/0.25/0.5/0.75/1.0`，再在原 16 个评测样本上
复验保守配置。推荐 `shape_alpha=0.25`：

| Metric | Hard Reset | Recommended |
|---|---:|---:|
| Shape jump | 0.745 | 0.573 |
| Body-scale jump | 0.00742 | 0.00597 |
| Local-pose residual | 4.72 deg | 3.88 deg |
| GT beta error | 1.557 | 1.536 |
| GT body-scale error | 0.06005 | 0.05961 |

推荐配置在四个来源都降低 shape/scale jump，并在总体上略微改善 GT shape 和
scale。但 THuman 的 GT beta error 仍由 `1.698` 增至 `1.760`。所以 V14.1
当前证明的是跨 shot 连续性，而不是每个数据源上都提高绝对人体精度。

推荐在线报告：

```text
output/v14_1_shot_aware_state_routing/recommended_true_reset/
```

## Causal Controls

| Memory | Shape jump | GT beta error |
|---|---:|---:|
| Correct memory | 0.259 | 1.575 |
| Zero-valued memory | 1.230 | 1.142 |
| Wrong-video memory | 0.799 | 1.621 |

正确 memory 明显比 wrong-video memory 更连续且更准确，错误 memory 会带来
可测量伤害，说明收益不是普通平滑正则化。Zero-valued beta 偶然更接近部分 GT
平均 shape，但产生最大的边界跳变，不能作为有效跨 shot memory。

## Commit Protocol

GT Boundary 下，单 cut 的长期 memory world-root error：

```text
Immediate Commit: 0.997 m
Align-Then-Commit: 0.747 m
Verify-Then-Commit: 0.901 m
```

Immediate Commit 把新 camera gauge 中的位置直接写入旧世界，坐标语义错误。
Align-Then-Commit 在写回前应用统一 Boundary SE(3)，因此是必须步骤。

固定 world-root jump 阈值 `1.5/2.0/2.5/3.0 m` 的 8-cut 消融表明：

- `1.5 m` 在 MVHuman100 上误拒绝一半更新，memory root error 由 `1.394 m`
  恶化到 `1.727 m`；
- `2.0-3.0 m` 没有拒绝样本，结果等同 Align-Then-Commit。

当前没有证据支持固定 Verify 阈值。缺少可靠 motion prediction 时，推荐直接
Align-Then-Commit，不让陈旧 world root 因误拒绝长期保留。

## Multi-Cut Rollout

四个来源各一个交替 camera-pair rollout，共 8 cuts、18 frames。

原 `shape_alpha=0.75` 诊断：

```text
8-cut shape drift:
Hard Reset          0.582
Selective Align     0.229

Memory world-root error:
Immediate Commit    1.135 m
Align-Then-Commit   0.931 m
```

保守 `shape_alpha=0.25`：

```text
8-cut shape drift:
Hard Reset          0.582
Selective Align     0.484

8-cut shape jump:
Hard Reset          0.592
Selective Align     0.435
```

更保守的融合牺牲一部分长期平滑，但降低了锁定错误 shape 的风险。Align commit
在四个来源均优于 Immediate commit。

正常无 cut 检查中，camera、pointmap 和 SMPL-X shape 的最大差异均为 `0`。

## Final Answers

1. Scene/camera 和 human 应采用不同 state transition。
2. Scene/camera 在 cut frame 解码前使用 fresh state，可保持 Hard Reset 稳定性。
3. 隔离的人体 memory 可以稳定改善 shape、body scale 和 local-pose continuity。
4. 当前有用信息是 canonical shape 和 root-centered local pose；raw token 和仅
   tracklet 没有可测量收益。多人 identity 尚未正式验证。
5. World root/global orientation 依赖 camera gauge，必须在 Boundary SE(3)
   确定后再提交，否则长期 memory 混入两个坐标系。
6. Two-stage 的 Align-Then-Commit 明确优于 Immediate Commit；当前固定 Verify
   规则没有额外收益。
7. 8-cut rollout 中 selective memory 的 shape drift 低于全部 reset。
8. Correct memory 明显优于 wrong-video memory，具有因果作用。
9. 使用 GT 或 Fixed Explicit Boundary 时，human continuity 收益都存在，且不
   改变 scene/camera 输出；它可以与 V16/V18 或当前显式 SE(3) 安全组合。
10. 该方向值得保留为 Shot-Aware Modality-Selective State Transition，但论文
    目前只能主张单人人体连续性；Boundary 精度和多人 Re-ID 是独立问题。

## Recommended Configuration

```text
pre-decode scene/camera hard reset
raw human token alpha = 0
shape alpha = 0.25
local pose alpha = 0.15
commit = align
fixed world-root verify threshold = disabled
```

Run the final single-cut evaluation with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/v14_1_shot_aware_state_routing_probe.py \
  --device cuda:0 \
  --cases_per_source 4 \
  --recommended_only \
  --skip_wrong_memory \
  --output_dir output/v14_1_shot_aware_state_routing/recommended_true_reset
```
