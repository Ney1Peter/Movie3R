# MVHuman `test1/test2` 数据审计与正式实验准入结论

日期：2026-08-30  
范围：仅审计新上传数据；未启动 GPU 推理，未修改论文正文。

## 1. 结论摘要

两份上传数据可以作为 Bridge3R 的**新 capture 级别、单人、受控弱纹理背景、跨相机硬切换测试集**。数据具有完整同步 RGB、前景掩码、2D 标注、相机内外参、逐帧 SMPL-X 参数、逐帧 3D 关键点和逐帧 SMPL-X 网格，能够支持严格的 Camera--Human 定量评估以及从小视角到接近 $180^\circ$ 的分层实验。

关键限定如下：

1. 它不是未见过的 MVHuman source 或未见过的相机 rig。最终 correction checkpoint 的训练数据使用过 `mvhuman100` 和 `mvhuman200`，新数据沿用了各自的 48/16 相机 rig。
2. 它与 correction 训练清单在 **capture/event/frame-member** 层面无重叠：训练 capture 为 `100001--100005`、`200001--200005`；新 capture 为 `100021--100025`、`200021--200025`。
3. 因此论文中可写成 **held-out MVHuman captures/motions**，不能写成 unseen subject、unseen camera rig、unseen source，也不能在缺乏官方说明时称为 MVHuman 官方 test split。
4. 数据没有场景深度或场景网格真值，只适合已经锁定的 Camera--Human 任务，不适合 Scene--Human 定量评测。
5. 背景肉眼呈近似均匀绿色摄影棚，但仍存在灯板、墙缝和地面线。正式声称“low texture”前，应以人体掩码外的图像梯度统计给出可复现的弱纹理定义；在此之前建议称为 **controlled weak-texture studio setting**。

## 2. 归档完整性与安全解压

用户消息中的 `Mvhuman-test1.zi` 实际文件名为：

- `/data/wangzheng/iJCV-CODE/data/Mvhuman-test1.zip`
- `/data/wangzheng/iJCV-CODE/data/Mvhuman-test2.zip`

| 外层归档 | 字节数 | SHA-256 | 完整性结论 |
|---|---:|---|---|
| `Mvhuman-test1.zip` | 8,876,412,777 | `86ccbf4ed2970157de75faa4ff86950e38b0d89b45c4eed1361772c4d879f6df` | Python `zipfile.testzip()` 返回 `None`，5个成员 CRC 全部通过 |
| `Mvhuman-test2.zip` | 3,614,782,728 | `40c8a9f0ed0eca8b5dce0f4e495b277a7a1cfdab84eac7593cedaf80041aeafa` | `unzip -t` 与逐成员 `gzip -t` 均通过 |

服务器的旧版 `unzip/7z` 会将 `test1` 报成中央目录/CRC 错误。进一步用 `zipdetails` 和 Python 标准库核验后确认：它是合法的 Zip64 流式归档，5个成员、Zip64 EOCD 和成员偏移一致；该报错属于旧工具对超大、流式 Zip64 成员的兼容问题，不是上传损坏。没有对原归档做任何修复或覆盖。

外层归档已安全解到全新的目录，原文件保留：

```text
data/MVHuman_test12_audit_20260830/
├── test1_archives/       # 100021.tar.gz -- 100025.tar.gz, 8.3 GiB
├── test2_archives/       # 200021.tar.gz -- 200025.tar.gz, 3.4 GiB
├── metadata/             # 审计抽取的相机与完整 JSON GT, 51 MiB
└── visual_samples/       # 只读视觉核验样例
```

10个内层 `tar.gz` 均通过 `gzip -t`。其 SHA-256 为：

| 成员 | SHA-256 |
|---|---|
| `100021.tar.gz` | `dac963ef9c53c7d4abe6123db1b0d8f678d9feabda40814bce90c88d69e75a3c` |
| `100022.tar.gz` | `41bffb0ad99bd5cba8c4bbd9c60706c07084d1cc6d3eaab1ff6f772747933184` |
| `100023.tar.gz` | `50aecdbe2bfcdaf91027a03357b814b9a4eb845be4eb7c3bdb1d6e4d6581bde7` |
| `100024.tar.gz` | `4c2d1da5772d27f1faf7f3653c7ff924bea9610040b6df5cfc10aea4e4983ae0` |
| `100025.tar.gz` | `19cda14ac314fe3af866cd15255bb570382d134e8f9149da1015a0ac1100411a` |
| `200021.tar.gz` | `83e3e9ea188711360e5cf3b1503eb2223cd0b44375a0513915849ce7b69d52b9` |
| `200022.tar.gz` | `166a3fd27f4f95862d0326035a8018980e5e173457c380696a7d60f931e14f84` |
| `200023.tar.gz` | `fe389bef46472f16dc60dcc5673bf6698863c83d7e89664cbff0d95ce17be078` |
| `200024.tar.gz` | `fa5e4f016cffd6fa9ede182dab78b8e84a8dd66749f20e3b926177ba9819814b` |
| `200025.tar.gz` | `5264e1aaaf818e89c8cfdc432ab991b7c3216eb02aaf03ed490b612710658c15` |

## 3. 数据结构

每个内层归档包含一个 capture：

```text
<capture>/
├── images_lr/<camera>/<raw_tick>_img.jpg
├── fmask_lr/<camera>/<raw_tick>_img_fmask.png
├── annots/<camera>/<raw_tick>_img.json
├── openpose/<camera>/<raw_tick>_img_keypoints.json
├── camera_intrinsics.json
├── camera_extrinsics.json
├── camera_scale.pkl
├── smplx/smpl/<frame_index>.json
├── smplx/keypoints3d/<frame_index>.json
├── smplx/smplx_mesh/<frame_index>.obj
└── smpl_param/                         # 辅助 SMPL 转换结果，部分帧缺失
```

原始 tick 从 `0005` 开始、以5递增；SMPL-X frame index 与 RGB 的关系为

```text
frame_index = raw_tick / 5 - 1.
```

RGB、mask、2D annotation 和 OpenPose 的 `<camera, raw_tick>` 键完全一致，131,600个同步键中没有发现模态缺失。每个时间点的全部相机共享同一套世界坐标 3D GT。

## 4. Capture、相机和帧统计

| Capture | 相机数 | 同步时间点 | RGB帧数 | raw tick范围 | 内层展开大小 |
|---|---:|---:|---:|---|---:|
| 100021 | 48 | 418 | 20,064 | 0005--2090 | 2.92 GiB |
| 100022 | 48 | 378 | 18,144 | 0005--1890 | 2.60 GiB |
| 100023 | 48 | 442 | 21,216 | 0005--2210 | 3.31 GiB |
| 100024 | 48 | 387 | 18,576 | 0005--1935 | 2.80 GiB |
| 100025 | 48 | 355 | 17,040 | 0005--1775 | 2.58 GiB |
| 200021 | 16 | 457 | 7,312 | 0005--2285 | 1.17 GiB |
| 200022 | 16 | 452 | 7,232 | 0005--2260 | 1.10 GiB |
| 200023 | 16 | 481 | 7,696 | 0005--2405 | 1.31 GiB |
| 200024 | 16 | 467 | 7,472 | 0005--2335 | 1.34 GiB |
| 200025 | 16 | 428 | 6,848 | 0005--2140 | 1.10 GiB |
| **合计** | -- | **4,265** | **131,600** | -- | **20.23 GiB** |

每个 capture 的全部相机拥有相同帧集合，足以构造统一长度、同一物理时刻、同一人物的硬切换序列。最短 capture 也有355个同步时间点，足以构造150帧协议而无需循环或补帧。

## 5. 人体 GT、身份和相机标定审计

### 5.1 人体 GT

- 每个时间点均有且仅有1个人，局部 ID 恒为 `0`。
- `smplx/keypoints3d` 每帧有118个关键点；前25个 Body25 关键点的有效置信度比例为 106,623/106,625（99.998%）。
- `smplx/smpl` 每帧都有87维 pose 数据以及 `Rh`、`Th`、shape。
- `smplx/smplx_mesh` 每帧均存在，与同步时间点数严格一致。
- 所有 JSON 数值均有限，没有 NaN/Inf 或结构损坏。
- `smpl_param` 辅助转换结果不是逐帧完整，但它不是正式评估的必要真值；完整的 SMPL-X JSON、Body25 3D GT 和 SMPL-X mesh 已足够，转换脚本也使用这些完整目录。

这里的 `id=0` 是 capture 内的单人标签，不能据此证明10个 capture 之间的人物身份互异。因此本数据可以证明 held-out captures/motions，不能证明 held-out identities。

### 5.2 标定

- `10002x` 五个 capture 共用同一套48相机标定，`camera_scale=1.846153846...`，对应 world scale `0.541666...`。
- `20002x` 五个 capture 共用同一套16相机标定，`camera_scale=1.538461538...`，对应 world scale `0.65`。
- 所有 rotation matrix 的最大正交误差不超过 `1.3e-6`，行列式位于 `0.999999--1.000001`。
- `10002x` RGB 分辨率为 2048x1500，标定原分辨率为4096x3000；`20002x` RGB 为1224x1024，标定原分辨率为2448x2048。现有转换器会按 x/y 分辨率比例缩放内参。

使用 `100021/CC32871A020/frame 1000` 与 `200021/22327107/frame 1000` 做独立投影检查：

| 样例 | 有效2D/3D关节 | 低分辨率平均投影误差 | 中位投影误差 |
|---|---:|---:|---:|
| 100021 | 20 | 11.84 px | 10.05 px |
| 200021 | 23 | 7.24 px | 4.64 px |

这与检测/拟合标注误差的量级一致，未发现坐标轴、毫米/米、内参缩放或 world scale 方向错误。

## 6. 可构造的视角跨度

以下角度使用相机外参 rotation 的 SO(3) geodesic distance，而不是按相机编号猜测。两个 rig 都覆盖从小跨度到接近 $180^\circ$ 的硬切换。

| Rig | 相机对数 | $[0,60)$ | $[60,120)$ | $[120,150)$ | $[150,180]$ | 最小/最大 |
|---|---:|---:|---:|---:|---:|---:|
| 10002x（48相机） | 1,128 | 342 | 419 | 159 | 208 | 6.4 / 179.8 deg |
| 20002x（16相机） | 120 | 32 | 48 | 16 | 24 | 21.3 / 180.0 deg |

可复核的代表相机对如下：

| Rig | 目标跨度 | 实际角度 | 相机A | 相机B |
|---|---:|---:|---|---|
| 10002x | 30 | 29.9 | CC32871A021 | CC32871A030 |
| 10002x | 90 | 90.0 | CC32871A030 | CC32871A032 |
| 10002x | 120 | 119.2 | CC32871A020 | CC32871A033 |
| 10002x | 150 | 150.1 | CC32871A020 | CC32871A034 |
| 10002x | 170 | 170.1 | CC32871A008 | CC32871A060 |
| 20002x | 30 | 23.9 | 22236236 | 22327111 |
| 20002x | 90 | 90.0 | 22236234 | 22327084 |
| 20002x | 120 | 114.3 | 22327107 | 22327109 |
| 20002x | 150 | 156.0 | 22327107 | 22327117 |
| 20002x | 170 | 178.2 | 22327084 | 22327111 |

正式 manifest 不应直接手挑“看起来效果好”的相机对。建议先冻结4个角度桶，每个 capture 使用同一套、由“距离桶中心最近+GT可见性合格”确定的相机对，再运行任何方法。

## 7. 与 correction 训练数据的重叠审计

最终 correction checkpoint 对应的保守训练宇宙见：

```text
Movie3R/versions/v14/cut_first_cross_source/manifests/train96ps/
├── mvhuman100.jsonl
└── mvhuman200.jsonl
```

审计结果：

| 层级 | 训练 | 新测试 | 交集 |
|---|---|---|---|
| source/domain | mvhuman100, mvhuman200 | mvhuman100, mvhuman200 | 有 |
| camera rig / camera ID | 48相机rig、16相机rig | 相同两个rig | 有 |
| capture | 100001--100005, 200001--200005 | 100021--100025, 200021--200025 | **无** |
| qualified event | 由训练 `pattern_id` 定义 | 新capture中待冻结的cut event | **无** |
| qualified frame/member | `<capture>/<camera>@frame` | 新capture对应成员 | **无** |

另外对 `Movie3R/config`、`Movie3R/docs` 和现有论文审计材料全文检索 `100021--100025`、`200021--200025`，除本次新计划文档外没有发现这些 capture ID。该结果支持“训练期间没有使用这些 capture”的表述。

最严谨的论文措辞建议：

> We evaluate on ten held-out MVHuman captures whose capture and frame members are disjoint from the correction-module training manifests, while retaining the dataset's original camera rigs.

不得将其扩大成 subject-disjoint 或 rig-disjoint 泛化结论。

## 8. 方法准入与缺项

### 8.1 现在即可进入正式适配的主方法

| 方法 | 数据是否足够 | 备注 |
|---|---|---|
| Strict Human3R | 是 | RGB输入；使用现有 MVHuman 转换器与严格重置/状态规则 |
| Bridge3R | 是 | 与 Strict Human3R 使用完全相同的RGB、cut manifest和因果历史 |
| PromptHMR | 是，需适配/小批 smoke test | 单人视频适配度高；作为 offline/full-video baseline 必须显式标注非因果输入优势 |
| TRACE | 原则上是，需适配/小批 smoke test | 支持单人作为多人方法的退化情形；正式表前需验证世界坐标和跟踪输出覆盖率 |
| GVHMR | 条件准入 | 单人视频适配度高，但 AIST 曾出现跨cut跟踪支持不足；必须用预先固定的12例pilot和结果无关的coverage准入规则 |
| SPEC | 可作为补充参考 | 逐帧、相机感知但无跨镜头历史；适合 Supplement，不宜与因果流方法混为同等时序设定 |

HumanMM、Multi-THuMBS 因没有可执行代码/同输入输出，不能在这个自建协议上产生直接重跑数字。

### 8.2 数据层面没有阻塞，但正式实验前必须完成

1. 冻结唯一 `manifest.jsonl`：capture、相机对、SO(3)角度、pre/post帧、raw member、GT member和哈希全部写入。
2. 仅用 GT 可见性做预推理 QC；不得根据任何方法的误差筛选序列。
3. 明确每种 baseline 的输入可见性：causal、offline/full-video、single-frame。
4. 为每种方法冻结同一 denominator；失败必须计入 Coverage，不能换序列补齐。
5. 在正式推理前完成12例pilot，只检查管线、坐标、跟踪和显存，不以效果选超参数。
6. 用 provided mask 排除人体和黑色 padding，计算背景梯度/Laplacian 分数；没有该统计时只写 weak-texture studio setting。

## 9. 建议的正式协议

建议将本数据定位为**单人弱纹理+大视角专项实验**，而不是替代三个人体多人主数据集。

### 9.1 Case 构造

- 10个 capture 全部使用。
- 4个视角桶：`[0,60)`, `[60,120)`, `[120,150)`, `[150,180]`。
- 主协议最少为 `10 captures x 4 bins = 40 cases`；如计算预算允许，可在每个 rig 每桶冻结两个不重复相机对，形成80 cases。
- 每个 case 使用150个同步时间点，前75帧来自相机A，后75帧来自相机B；cut 位于固定中点。
- 同一 capture 的截取位置由确定性规则给出，且在运行方法之前冻结。
- 全部方法读取完全相同的 RGB member；GT mask、2D/3D标注只能用于 QC/评估，不能作为方法输入，除非该方法官方协议本来要求并在表中明确标注。

### 9.2 建议指标

正文或专项主表：

- PA-MPJPE：人体局部姿态；
- Anchor-MPJPE 或现有单次首镜头对齐后的 world joint error：跨cut全局人体位置；
- boundary root-position error：cut处根节点平移连续性；
- boundary orientation error：cut处人体朝向连续性；
- relative camera rotation error：跨镜头相机旋转；
- relative camera-center/translation error：跨镜头相机位置；
- Coverage：固定 denominator 上的有效输出率。

Supplement：

- 四角度桶分层结果与置信区间；
- camera ATE-Sim3/SE3（若沿用统一 evaluator）；
- 纹理分数分层曲线；
- 完整方法支持数。

这是单人数据，IDF1 基本退化为“是否持续保留唯一人物”，不应作为主要贡献指标。静态相机硬切换下，relative camera transform 和 anchored human error 比单独的长轨迹 ATE 更直接。

## 10. 最安全的选择性物化策略

不建议为了实验再次复制全部 RGB，也不建议每读取一帧就随机扫描 `tar.gz`。推荐流程：

1. 先由冻结 manifest 生成每个内层归档的**精确 member list**；列表只允许相对路径，拒绝 `/` 开头和 `..`。
2. 每个 `tar.gz` 只扫描一次，批量提取该 capture 的所有选中 RGB/mask/2D annotation，以及所需的共享 camera JSON、camera scale、SMPL-X JSON/keypoints/mesh。
3. 提取到新的只读 raw root，例如 `data/MVHuman_test12_formal_raw_v1/<capture>/`；使用 `--keep-old-files` 或脚本级“目标必须不存在”检查，禁止静默覆盖。
4. 调用现有 `Movie3R/scripts/convert_mvhuman_to_training.py` 的逻辑生成统一格式，但 RGB/mask 使用 symlink，避免第二份图像副本；输出到新的 `data/Training/mvhuman_test12_formal_v1/`，不要混入训练目录。
5. 对冻结 manifest、member list、转换摘要和结果文件生成 SHA-256 ledger。
6. 所有 baseline 只从同一个物化 root 建立输入，禁止各自重新采样。

如果40/80 case最终覆盖了大部分原始帧，完整展开全部10个内层归档也只约20.23 GiB，可以接受；但仍应保持 raw root 与 training/conversion root 分离，并用 symlink 避免重复。

## 11. 最终判断

**准入：通过，但必须以 capture-disjoint、same-rig 的弱纹理专项测试来表述。**

数据本身足够完整，足以运行 Strict Human3R、Bridge3R，并进入 PromptHMR、TRACE、GVHMR、SPEC 的统一适配和pilot阶段。当前唯一真正的事实性限制不是缺少 GT，而是：该数据与训练共享 MVHuman source 和相机 rig，且尚未冻结正式 case manifest/纹理定量定义。因此下一步应先冻结协议和 overlap ledger，再进行任何 GPU 推理。
