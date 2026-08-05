# Movie3R-v15-final

## 一句话定位

Movie3R-v15 是一个因果、在线、可拒绝的跨镜头三维重建框架：它先用 Human3R/V9 给出粗坐标系，再用人物身份和人体几何判断是否可信，最后只在证据足够时联合修正相机与人体。

这个目录是当前主线的冻结版本。之后的大批量实验应以这里的配置、阈值、checkpoint 和输出契约为准，而不是临时修改 v14 脚本。

## 当前最终流程

```text
RGB 流
  → causal shot detector（或 manifest 给定的 cut）
  → V9 shadow branch：得到 post 的粗 B0 gauge
  → clean-reset Human3R raw branch：得到 post 的独立预测和 root rays
  → B0：把相机、背景和人体放入 pre-shot 坐标系
  → anonymous identity association：建立跨 shot 的 persistent person ID
  → BRTC-LC：冻结相机，用五个核心关节的两视线三角化修正 root/depth/layout
  → C1-EMA25：shot 内只稳定静止人体，运动/不确定 track 走 fallback
  → adaptive joint gate：验证通过才共同修正 camera + human
  → 输出标准 demo.py payload 和可审计 diagnostics
```

每一个边界只使用最后一帧 pre、第一帧 post 以及已经看到的状态；不会偷看未来 post 帧，也不会把 GT 放进推理。任何人物数量变化、遮挡、匹配歧义或几何残差过大，都会拒绝更新并保留上一个可信结果。

## 输入和输出

输入是两个相邻 shot 的 RGB 流。对于一个 batch case，至少要提供：

```json
{"case_id":"three_t1100_c1_c2","sequence":"three","frame":1100,"pre_camera":"1","post_camera":"2","pre_frames":5,"post_frames":25}
```

`frame` 是最后一帧 pre / 第一帧 post 的边界。`pre_camera` 和 `post_camera` 是数据集中的两个 RGB 流。AvatarReX 还需要 `avatarrex_group`。模型输入经过 Human3R 的图像预处理后，产生相机位姿、背景深度/置信度、SMPL-X 人体 mesh、关节和 native detection index。

输出目录包含：

- `original_human3r/`：严格原版 Human3R 的对照 payload；不传 cut 事件。
- `movie3r_raw_current_human3r/`：同一当前 checkpoint 的 clean-reset raw shadow payload，只供 adaptive joint 读取 root rays，不作为最终结果。
- `movie3r_b0_brtc_c1/`：冻结的粗对齐、身份匹配、BRTC-LC 和 C1 结果。
- `movie3r_final_adaptive_joint/`：在上一目录上运行自适应联合 gate 后的最终结果；gate 拒绝时与上一目录相同。
- `manifest.json`：输入、checkpoint、命令、运行时间和输出路径。
- `movie3r_final_adaptive_joint/adaptive_joint_boundary.json`：接受/拒绝理由、残差、匹配 margin 和应用的变换。

三个 payload 都兼容原有 `demo.py` 的三维可视化方式；runtime 固定使用 CPU，默认不会占用 GPU。

## 模块职责

### 1. Causal shot detector

它只从当前 RGB 帧和相邻帧的图像特征产生 cut probability。冻结的审计模型是 causal GRU，F1=0.982、FPR=0.031、Brier=0.015。实际部署可以把 detector 的 cut 提议交给几何 gate；即使图像 detector 误报，人体残差不可信时也会 abstain。为了进行可重复的指标比较，batch manifest 也可以直接给出已知的 `frame`，这不等于把 GT 放进 runtime。

### 2. V9 / B0 learned coarse gauge

V9 不是最终对齐器，而是一个 learned coarse gauge proposal：shadow branch 带着 pre-shot 状态读入第一帧 post，clean-reset branch 独立重建 post，二者的相机变换构成 B0。B0 同时完成三件事：给相机/背景一个可用的初始坐标系、把 post 人体搬到 pre 的大致位置、为跨 shot ID matching 提供较稳定的 body 形状和顺序。

V9 不能被删除。AvatarReX 单人低纹理上，直接 raw SE(3) 的相机误差约为 2.107 m / 64.53°，有 V9 后 adaptive joint 可降到 0.054 m / 0.44°；三人约 174° 跨镜头上，no-V9 raw SE(3) 为 4.265 m / 151.89°，B0+BRTC+C1 为 0.054 m / 1.82°。这些数字分别属于两个受控 case，不能替代大规模 benchmark，但足以说明 B0 是必要的粗初始化和身份预条件。

### 3. Anonymous identity association

原版 Human3R 的 `smpl_id` 只是每帧 native detection index，不保证跨 shot 的 persistent identity。v15 在 B0 坐标下用 root、torso 和 centered joints 做匿名一对一匹配，并保留 margin 和 unmatched 状态。已有 41 个多人 cuts 中，原版直接 root/torso matching 只有 41.5%--46.3% 正确，而 B0 后 matcher 为 100%。因此论文应把它写成 identity-preserving gauge correction，而不是声称 Human3R 原生已有 Re-ID。

### 4. BRTC-LC fine human alignment

相机先保持 B0 不动。对 pre 最后一帧和 post 第一帧的 pelvis、左右髋、左右肩五个关节，分别沿两台相机的射线求最近点。通过 ray gap、parallax sine 和 MAD gate 去掉不可靠关节，再用 group median 与 pre-layout residual 得到每个人的 root/depth shift。它只改 post 人体的 root、joints 和 vertices，不改相机、姿态和形状；拒绝的 track 保留 exact B0。

在 42 cuts / 125 people 的 MultiHuman three offset1 确认集上，root 误差从 0.3779 m 降到 0.2314 m，joint 从 0.4117 m 降到 0.2745 m，vertex 从 0.3891 m 降到 0.2525 m，自动关联准确率为 1.0，相机变化为 0。

### 5. C1-EMA25 within-shot stabilization

BRTC 只处理边界，C1 处理一个 shot 内的静止人体。它用 camera-local root/body step 做 causal EMA，并用 entering/exiting hysteresis 区分静止和运动。稳定时把同一个 bounded translation 同时加到 root、joints、vertices；运动、短历史、可见性变化、未匹配或 gate reject 都走 B0+BRTC fallback。C1 永远不改相机，且不使用未来帧。

### 6. Adaptive shared camera-human gate

低纹理单人场景的关键问题是背景无法可靠确定相机。v15 用 B0 人体的跨 shot 刚体残差提出旋转，用 B0 与同 checkpoint raw shadow 的人体 root rays 平均提出相机平移；人体绕已确定的 BRTC root 旋转，保证相机与人体相对位置不被任意拉开。只有同时满足旋转大、vertex RMS 小、归一化 RMS 小、person permutation margin 足够大时才提交更新；否则保留 B0+BRTC+C1。

## ICLR 论文叙事

论文的中心问题不是“再训练一个更大的 Human3R”，而是：流式三维重建在 shot cut 时会发生 gauge discontinuity；背景纹理强时相机可能对但人物 ID 错，低纹理时人物结构可能对但相机 gauge 错。单独优化相机或单独平移人体都无法覆盖这两个 failure mode。

建议题目方向：**Movie3R: Causal Adaptive Camera-Human Gauge Correction for Streaming 3D Reconstruction across Shot Cuts**。

三条主要贡献：

1. **Causal gauge transaction**：把跨 shot 处理定义成一个只使用 boundary evidence 的 transaction，显式区分 learned coarse proposal、human-only refinement 和 shared camera-human commit，并提供安全 abstention/fallback。
2. **Identity-preserving human gauge correction**：在 B0 坐标中用匿名 permutation-aware association 和 five-joint ray/layout consensus 做 camera-frozen 精对齐，解决相机看似正确但人物 180°/ID 错位的问题。
3. **Adaptive low-texture joint correction**：用同 checkpoint 的 raw shadow root rays 补充背景失效时的相机证据，在几何可信度通过时联合更新相机和人体；同一个规则覆盖多人高纹理和单人低纹理，而不引入额外预训练模型。

论文要强调的是“online + adaptive + camera-human consistency”，而不是遮挡处理或完整的人体重识别。当前版本没有解决 severe occlusion、new-person entry/exit 和任意长时 re-ID，因此摘要中不要过度声称通用遮挡鲁棒性。

## 需要报告的实验

最终论文表格应按同一 checkpoint、同一 split、同一 runtime protocol 比较：

1. strict original Human3R；
2. no-V9 / raw SE(3) control；
3. B0 only；
4. B0 + BRTC-LC；
5. B0 + BRTC-LC + C1；
6. B0 + BRTC-LC + C1 + adaptive joint；
7. oracle boundary 与 causal GRU detector。

指标至少包括 camera translation/rotation error、MPVPE、root/joint error、ID continuity、cut seam jump、within-shot drift、gate acceptance、abstention rate 和 CPU latency。Multi-THuMBS 的官方指标应单独按其协议复现；目前受控 MultiHuman、AvatarReX 和 EgoHuman 结果只能作为开发/泛化证据，不能冒充官方 Multi-THuMBS leaderboard。

## 批量运行

运行前请使用项目原有、已安装 `torch`、`smplx`、OpenCV 和 SciPy 的 Python 环境；发布脚本会沿用当前解释器，但会强制 `CUDA_VISIBLE_DEVICES=""`。本次 smoke test 使用系统 `python3` 时因没有 `torch` 而停在模型导入，这是环境依赖，不是方法结果。

先验证发布配置：

```bash
python3 versions/v15/validate_release.py
```

单个 case（CPU）：

```bash
python3 versions/v15/run_case.py \
  --case versions/v15/BATCH_MANIFEST_TEMPLATE.jsonl \
  --line 1 --overwrite
```

推荐用 batch：

```bash
python3 versions/v15/run_batch.py \
  --manifest versions/v15/BATCH_MANIFEST_TEMPLATE.jsonl \
  --max-cases 2 --continue-on-error --overwrite
```

所有结果都写到 `output/v15/`；`--checkpoint` 可以在 manifest 或命令行覆盖，但不同 checkpoint 的结果必须分开汇总。

## 冻结边界和限制

- 本版本冻结的是研究主线和可审计 runtime，不代表 ICLR 结果已经完成。
- detector 当前只在审计数据上验证过，必须在正式 train/val/test split 上重新报告。
- adaptive joint 是冻结的显式几何模块，不应写成已经训练好的神经网络。
- 官方 Multi-THuMBS 协议、遮挡/新人物和更大规模跨域泛化仍需要批量实验补齐。
- 不要把 GT、未来帧或原版 Human3R 的指标误写为 Movie3R 指标。
