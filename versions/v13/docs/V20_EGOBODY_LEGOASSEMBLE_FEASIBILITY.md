# V20 EgoHumans `001_legoassemble` 多人调试可行性与首轮实验

> 命名勘误（2026-07-27）：`001_legoassemble` 实际属于 EgoHumans。本文文件名中的
> `EGOBODY` 和本地父目录 `data/EgoBody/` 是历史命名，为保持旧引用和实验路径可复现而保留。
> 独立的 EgoBody release 位于本地 `data/EgoHuman/`。

## 1. 结论

`/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble` **适合作为 V20 多人方法的首要调试集**，尤其适合验证：

- Human3R 原生多人检测与 within-shot track ID；
- hard reset 后的跨视角 token Re-ID；
- 已知身份时，多人物 root/orientation 是否能形成更稳定的共同 Boundary；
- 某些人物漏检时，是否能只使用剩余可见人物；
- 多人共识失败时，能否退回单人或 Fixed Explicit。

它目前只应作为开发与几何诊断集，不能单独承担最终论文 benchmark。主要原因是 GT body 为 SMPL、Human3R 输出为 SMPL-X，原始外部相机图像为鱼眼，而且这里的 GT body 是优化拟合结果而非独立 mocap 真值。

## 2. 数据结构

序列包含：

| 项目 | 内容 |
|---|---|
| 同步帧 | 601 帧，`00001` 到 `00601` |
| 外部相机 | `cam01` 到 `cam08`，共 8 个固定视角 |
| Aria 相机 | `aria01` 到 `aria03` |
| 稳定人物身份 | `aria01`、`aria02`、`aria03` |
| 外部相机图像 | `3840 x 2160` |
| 相机模型 | `OPENCV_FISHEYE` |
| 人体 GT | bbox、17 个 2D joints、3D pose、SMPL 参数、6890 vertices、45 joints |

关键目录：

```text
exo/cam01...cam08/images/
processed_data/bboxes/<camera>/rgb/
processed_data/2d_joints/<camera>/rgb/
processed_data/3d_joints/
processed_data/smpl/
colmap/workplace/cameras.txt
colmap/workplace/images.txt
colmap/workplace/colmap_from_aria_transforms.pkl
```

八个 exo 相机都位于同一个 COLMAP world。`processed_data/smpl` 位于 `aria01` metric world，转换到 COLMAP world 时必须使用：

```text
colmap_from_aria_transforms.pkl["aria01"]
```

该 Sim(3) 的尺度为：

```text
1.7142045310549123
```

这一步只用于 GT evaluation，不进入 Human3R candidate generation。

## 3. 标定与 GT 一致性审计

同一固定 exo 相机在 COLMAP 中有 11 份 pose record。相机中心最大离散通常为 `1-5 mm`；`cam06` 最大约 `30 mm`，仍可用于开发评测，但应该保留该误差说明。

将 frame 300 的 GT SMPL vertices 从 `aria01` world 转到 COLMAP world，再使用各相机的 fisheye 模型投影，与数据集 bbox 对比，三个身份的平均 bbox IoU 为：

| Camera | Mean bbox IoU |
|---|---:|
| cam01 | 0.846 |
| cam02 | 0.765 |
| cam03 | 0.808 |
| cam04 | 0.465 |
| cam05 | 0.715 |
| cam06 | 0.822 |
| cam07 | 0.782 |
| cam08 | 0.695 |

投影总体相符。`cam04` 由于遮挡、边缘畸变和拟合差异更困难，适合作为 fallback/漏检测试，不适合作为唯一精确 reprojection 样本。

## 4. 首轮严格 Lite 测试

本轮关闭：

- DA3；
- Keypoint R-CNN shared scale；
- V11.4 shared scale；
- VGGT；
- scene refinement。

执行逻辑为：

```text
RGB stream
-> frozen Human3R multi-human inference
-> pre-decode scene/camera hard reset at every camera cut
-> preserve native Human3R tracklet behavior
-> export per-person native tokens
-> GT identity only for controlled source prototypes, scoring, and GT-ID geometry Oracle
```

在当前 feature-discriminability probe 中，GT identity 用于把 source-shot detections 组成正确的 per-ID prototype，并用于 assignment scoring；它不改变导出的 feature 或 L2 distance。该步骤是受控 probe，不是可部署 Re-ID。GT camera 不进入 Boundary candidate，只在最后评价 camera error。

测试了三条三相机链：

```text
cam01 296-300 -> cam06 300-304 -> cam07 304-308
cam02 176-180 -> cam05 180-184 -> cam08 184-188
cam03 416-420 -> cam04 420-424 -> cam01 424-428
```

每个 Boundary 前后重复同一个同步时间戳，先排除真实人体运动带来的身份和几何歧义。共 45 张图、6 个 cuts。

## 5. 多人检测与原生 Track ID

第一条链中每帧均检测到 3 人，45 个 detection-to-GT assignments 全部成功。

第二条链的 `cam02` 前四帧只检测到 2 人，之后恢复 3 人。第三条链的 `cam04` 每帧只检测到 1 人。该现象说明 EgoHumans 能同时覆盖：

- 三人全部可见；
- 某个人短时漏检；
- 大面积遮挡后仅剩一个可靠人物；
- 人数变化后的 fallback。

Human3R 原生 track ID 在 camera cut 后并不稳定。例如第一条链：

```text
aria01: 1 -> 1 -> 1
aria02: 0 -> 2 -> 0
aria03: 2 -> 0 -> 3
```

也就是说，原 tracker 在两个 wide-view cuts 都发生身份交换或重新编号。当前输出数组下标和原生 `smpl_id` 均不能直接当作跨 shot 稳定身份。

## 6. 原生 Token 跨 Shot 探针

对每个 shot 的历史 feature 取 prototype，在下一 shot 第一帧使用 Hungarian matching。以下为 normalized L2，在 6 个 cut 上共有 14 个可评价 assignment：

| Feature | Correct | Accuracy |
|---|---:|---:|
| refined human token `H'` | 4 / 14 | 28.6% |
| fused human prompt | 6 / 14 | 42.9% |
| Multi-HMR head token | 6 / 14 | 42.9% |
| CUT3R head token | 8 / 14 | 57.1% |
| SMPL beta | 10 / 14 | 71.4% |
| root-centered local pose | 13 / 14 | 92.9% |

首轮结论：

1. 原生 refined `H'` 在该序列的大视角 cut 后不具备足够的身份稳定性。
2. CUT3R head token 比 `H'` 更好，但 `57.1%` 仍不能部署。
3. beta 在这三个固定人物上较强，但还需要验证跨人物、相似体型和跨 capture 泛化。
4. local pose 的高正确率不能解释成 identity 能力，因为 Boundary 两侧是同一时间戳，姿态几乎相同。它目前只是同步匹配 cue。
5. 暂时不应训练 Shot-ID Adapter。应先增加不同 timestamp、明显动作变化、进入/离开和遮挡测试，确认 raw token 是否至少存在可分性。

## 7. GT-ID 多人物几何 Smoke Test

当前几何 probe 是一个**人体 full-body orientation/root 诊断**：每个 GT-matched 人给出 rotation 和 translation candidate，再比较单人、普通平均和 head-confidence 加权平均。

它尚未实现最终的 `Fixed Explicit coarse + V16 torso-motion residual + 20 deg bound`，因此以下数字只证明多人共识值得继续，不是正式 V20 结果。

在 4 个确实有 3 个 shared humans 的 cut 上：

| Method | Camera T mean | Rotation mean |
|---|---:|---:|
| Oracle best single | 0.816 m | 13.48 deg |
| Three-human mean | 0.664 m | 12.68 deg |
| Confidence weighted | **0.575 m** | **9.96 deg** |

逐 cut 结果：

| Cut | Best single T/R | Mean T/R | Weighted T/R |
|---|---:|---:|---:|
| cam01 -> cam06 | 0.707 m / 12.02 deg | 0.615 m / 6.46 deg | 0.607 m / 6.49 deg |
| cam06 -> cam07 | 0.846 m / 12.76 deg | 0.530 m / 6.83 deg | 0.476 m / 6.02 deg |
| cam02 -> cam05 | 1.242 m / 21.31 deg | 0.647 m / 16.75 deg | 0.384 m / 9.60 deg |
| cam05 -> cam08 | **0.467 m / 7.84 deg** | 0.863 m / 20.69 deg | 0.835 m / 17.75 deg |

前三个 cut 中多人共识明显更好，但 `cam05 -> cam08` 明显退化。这说明：

- 多人确实提供了可利用的冗余约束；
- 不能无条件平均所有人物；
- head confidence 本身不足以识别几何异常人物；
- 下一版必须加入 translation-candidate dispersion、pairwise-layout residual、Huber/trimmed consensus 和 reject-then-resolve；
- 当只有一个 shared human（如 `cam04`）时，必须严格退化为原单人方法。

## 8. 对 V20 设计的判断

EgoHumans 的首轮结果支持 V20 的总体职责分离：

```text
Token / body feature answers WHO.
Explicit multi-human geometry answers WHERE.
All accepted humans share ONE Boundary.
```

但它同时否定了两个过早假设：

1. 不能假设 Human3R 原生 refined token 已经是可靠的跨镜头 Re-ID embedding。
2. 不能假设所有匹配人物直接求平均必然优于单人。

因此正确顺序仍然是：

1. 审计并导出原生人物表示；
2. 先做 GT-ID multi-human geometry gate；
3. 在真实 Fixed Explicit/V16 方程中验证 robust consensus；
4. 单独验证 token/shape/pose 的身份匹配；
5. 两个门槛都通过后，才做 deployable token-guided alignment。

## 9. 下一轮实现顺序

下一轮保持 DA3、V11.4 和 VGGT 关闭，避免混淆多人贡献：

1. 接入真实 Fixed Explicit coarse rotation。
2. 为每个 GT ID 独立计算 pre-cut torso velocity 和 post-cut V16 residual。
3. 所有人共享同一个 20 deg residual bound。
4. 比较 first/largest/highest-confidence/oracle-best single。
5. 比较 mean、confidence weighted、Huber、trimmed 和 RANSAC consensus。
6. 使用 pairwise-layout residual 做异常人物诊断。
7. 加入非同步 Boundary，验证 motion prediction。
8. 加入 entry/exit/occlusion，并验证单人和无人 fallback。
9. GT-ID 几何门槛通过后，再实现 external tentative identity bank、dustbin、TTL 和 Align-Then-Commit。

## 10. 代码和输出

实验入口：

```text
versions/v13/egobody_probe.py
versions/v13/native_token_probe.py
```

为了复测不同相机链，EgoHumans probe 支持：

```bash
.venv/bin/python versions/v13/egobody_probe.py \
  --device cuda:5 \
  --output_dir output/v20_egobody_legoassemble/example \
  --segments cam01:296-300 cam06:300-304 cam07:304-308
```

主要输出：

```text
output/v20_egobody_legoassemble/v20_egobody_three_person_probe.json
output/v20_egobody_legoassemble/v20_egobody_compact_tokens.pt
output/v20_egobody_legoassemble/egobody_token_distance_matrices.png
output/v20_egobody_legoassemble/frame00300_exo_contact.jpg
output/v20_egobody_legoassemble/cam02_cam05_cam08/
output/v20_egobody_legoassemble/cam03_cam04_cam01/
```
