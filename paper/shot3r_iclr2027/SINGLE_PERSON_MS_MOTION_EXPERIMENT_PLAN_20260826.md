# Bridge3R 单人多镜头（ms-Motion）实验计划

更新日期：2026-08-26  
状态：**路线已锁定为“从原始 AIST++ / Human3.6M 自行构造”；规划与初步数据清点完成。** 尚未启动任何新模型推理、训练、下载或基线环境安装。  
目的：以可复现且不混淆协议的方式，补充 Bridge3R 在**单人、同一场景、跨视角镜头切换**下的证据，并与 HumanMM 所使用的 ms-AIST / ms-H3.6M 数据来源建立合理关联。

## 1. 结论与边界

### 1.1 是否可以作为实验数据？

**可以，但需要先通过标注时间对齐审计，且不能原样复用 EgoBody、EgoHumans 和 Harmony4D 的相机—人体联合评测器。**

这两个压缩包并非原始单镜头 AIST / H3.6M，而是 HumanMM 构造的 `ms-Motion` 风格数据：每个长视频由同一动作在不同物理视角下的连续片段拼接而成；`clip_info` 给出每一次真实剪切的边界。因此，它们正好检验 Bridge3R 要解决的单人版本问题：切镜后，模型是否能够保留连续的人体世界运动而不把旧视角的状态错误带入新视角。

但它们不适合直接回答“真实相机轨迹是否重建正确”：压缩包没有可直接使用的每帧相机内外参。故不应在本数据上报告或比较 EgoBody 协议中的 `ATE-SE3`、相机 `RPE`、camera seam；也不应把不同骨架、不同 world-gauge 的绝对数值与三个多人数据集求平均。

### 1.2 与 HumanMM 的比较边界

HumanMM 的论文使用相同的 ms-AIST / ms-H3.6M 来源，主表比较 SLAHMR、WHAM、GVHMR 和 HumanMM，并报告 PA-MPJPE、WA-MPJPE、RTE、ROE、Jitter、Foot Sliding。其公开数据包足以枚举输入视频、GT 人体标签与镜头边界，但没有公开完整的处理、基线推理和版本化 evaluator；特别是 H3.6M 的 world translation 按其补充材料来自后处理参考而非原始标注。

因此，本工作的正确表述是：

- 在**同一公开 ms-Motion 输入**上，以 Bridge3R 的冻结、可审计 evaluator 比较实际运行的开源方法；
- HumanMM 论文数字仅放在独立的 *source-reported / different evaluator* 参考表，不能与我们的数字混排、加粗排名，不能宣称“击败 HumanMM”；
- 若后续得到 HumanMM evaluator 或作者明确的完整评测定义，才可建立额外的复刻表；这不是当前全量实验的前提。

### 1.3 已锁定的数据构造选择：自行从原始数据构造

**正式实验选择从原始 AIST++ 和 Human3.6M 的官方视频、相机标定和人体 GT 自行构造 multi-shot 序列，而不将 HumanMM 打包的 ms-Motion 直接作为正式 benchmark。**

这是更严谨、也更契合 Bridge3R 现有三套多人实验的选择：

1. 原始相机内外参、人体 GT、RGB 与 frame index 可由同一来源构成闭环，能够复用 Camera--Human protocol 的坐标、manifest、coverage、cut seam 与统计原则；
2. cut 规则、相机跨度、时长、分辨率、训练/开发/正式 Test 划分均能在查看结果前冻结，且所有方法严格使用同一 RGB 序列；
3. 可避开 HumanMM 包中 AIST 缺 PTH、H3.6M frame-map 不清和相机文件失效的问题；
4. HumanMM 没有公开可复刻的完整 evaluator，沿用其打包格式也不能使我们的数值成为其 Table 1 的可直接比较数字。

HumanMM 的 ms-Motion 包保留为**只读协议参考和构造 sanity check**：我们可比较 shot 数、时长分布、相机选择和边界形式是否同类，也可在少量可对齐例上检查自建工具；它不产生正文的正式分数，也不用于与 HumanMM 表格混合排名。

### 1.4 最小化官方下载方案（已决定，尚未下载）

不下载数百 GB 的全量数据。正式数据按“**轻量官方标注 → 冻结 source manifest → 精确下载所需 RGB**”三步取得。

| 来源 | 正式 source pool | 必须下载的官方标注 | 只下载的 RGB | 正式 case 目标 | 估计新增空间 |
|---|---|---|---|---:|---:|
| AIST++ | 官方 `pose_test` split 中 100 个不同的原始动作序列 | `cameras.zip`（27 KB）、`motions.zip`（约 79 MB）、`splits.zip`、`ignore_list.txt`；建议另取 `keypoints3d.zip`（约 877 MB）用于独立投影审计 | 每个 source 的 4 个经 manifest 指定的 `cXX` 原始 MP4，即约 400 个 `10M` 视频 | CS150 100；同 source 构造 fixed-duration MC150-3 / MC150-4 | 通常数 GB，加标注约 1 GB；以实际 manifest 为准 |
| Human3.6M | 官方 standard-test subjects **S9、S11**；15 actions × 2 subactions，共至多 60 个独立 source sequences | 官方 camera calibration、world 3D joint positions（`D3_Positions`）及官方 frame/timestamp metadata | 每个 selected source 的 4 个同步 RGB 相机流，至多约 240 个视频；若门户仅按 subject 打包，则只下载 S9、S11 的 `Videos` 项 | CS150 至多 60；同 source 构造 fixed-duration MC150-3 / MC150-4 | 取决于门户封装；应为两个 subject 的视频包，远小于全库，下载前以页面大小确认 |

**AIST++ 的具体方式。** 官方 API 的 `downloader.py` 会下载全体视频，不能直接使用。我们将在取得 annotation 后，从冻结 manifest 生成一个 `aist_selected_urls.txt`，每行使用官方地址：

```text
https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/<video_name>.mp4
```

其中 `<video_name>` 由 AIST++ sequence name 的 `cAll` 替换为 manifest 指定的 `c01`--`c09`。下载前必须由用户确认同意 AIST Dance Video Database Terms of Use。RGB 到标签严格按官方规定的 60 FPS 时间轴映射，再统一重采样到 Bridge3R 的 30 FPS protocol；不会把容器帧号直接当作 GT frame。

**Human3.6M 的具体方式。** 使用持证的官方网页账号，只选 S9、S11 的 `Videos`、world 3D pose / camera calibration 所在的最小数据项；不下载训练 subjects、TOF、segmentation、mixed-reality、完整逐帧 image dump 或其他模态。若站点支持单文件选择，就按冻结的 240 条文件名下载；若只提供 subject archive，则下载 S9 与 S11 的 RGB archive 后仅保留 manifest 使用的文件。S9/S11 是通用 H3.6M test subjects，可避免使用通常的训练 subjects S1/S5/S6/S7/S8。

**case 数和独立性。** AIST++ 的 `pose_test` 中有 137 条满足 300 tick（5 秒）条件，足够以固定 hash 选择 100-case 主表；pilot 从不计入正式结果的 `pose_val`（必要时 `pose_train`）抽取，绝不占用或混入 `pose_test`。H3.6M 不人为把同一动作的不同窗口当作 100 个独立样本：主表最多 60 个 source-level case，统计以 source sequence 为独立单元；MC150-3 / MC150-4 是同一 source 的 fixed-duration 压力测试版本，不与 CS150 合并平均。

## 2. 数据清点（已完成）

本节清点的是已上传的 **HumanMM ms-Motion 打包数据**。在第 1.3 节锁定的新路线下，它仅作为协议参考，不是正式实验的 source of truth；正式构造所需的原始 AIST++ 与 Human3.6M 视频、官方 calibration 和原生人体标注需要单独取得并核验。

### 2.1 文件布局

```text
data/ms-aist/
  videos/shot{2,3,4}/aist_ms_*.mp4
  labels/aist_100_shot{2,3,4}.json
  labels/aist_100_shot{2,3,4}.pth

data/ms-h36m/
  videos/shot{2,3,4}/h36m_ms_*.mp4
  labels/h36m_100_shot{2,3,4}.json
  labels/h36m_100_shot{2,3,4}.pth
```

每个数据集、每种 shot 数都有 100 个 MP4 和 100 条 JSON 元数据；两个数据集合计 600 条多镜头视频。AIST 视频为 1920×1080、30 FPS；H3.6M 视频约为 1000×1000、30 FPS。`clip_info` 以闭区间帧索引记录各 shot 的来源相机与起止帧，例如 AIST 的首个 2-shot 视频共 645 帧，两个片段为 `[0,340]` 与 `[341,644]`。

| 子集 | 视频 / JSON | 可用 PTH GT | 可评严格交集 | 可用人体标签 | 已知限制 |
|---|---:|---:|---:|---|---|
| ms-AIST 2-shot | 100 / 100 | 92 | 92 | 17-joint 3D、SMPL pose / trans / scale | 8 条没有 PTH；相机 JSON 指向失效的作者机器绝对路径 |
| ms-AIST 3-shot | 100 / 100 | 89 | 89 | 同上 | 11 条没有 PTH；无本地相机标定 |
| ms-AIST 4-shot | 100 / 100 | 90 | 90 | 同上 | 10 条没有 PTH；无本地相机标定 |
| ms-H3.6M 2-shot | 100 / 100 | 100 | 100 | SMPL-X body pose / global orient / translation | `cam_json` 缺失；需要核验视频—标签 frame map |
| ms-H3.6M 3-shot | 100 / 100 | 100 | 100 | 同上 | 同上，且抽查显示长度并非天然逐帧对应 |
| ms-H3.6M 4-shot | 100 / 100 | 100 | 100 | 同上 | 同上 |

**缺失 GT 的 AIST 视频不以其他视频替代。** 每个 shot 子集固定使用其视频、JSON、PTH 的三者交集；所有方法在该子集使用完全相同的 `case_id` 列表。

### 2.2 已发现的必须先解决的问题

1. **AIST 的相机文件不可用。** `cam_json` 指向如 `/comp_robot/.../setting9_2.json` 的外部绝对路径，压缩包不含这些文件；不能假设已经有 GT camera trajectory。
2. **H3.6M 帧轴需要审计。** 抽查的 2-shot 首例中，MP4 有 604 帧，而 `clip_info` 与 PTH 人体序列为 605 帧；3-shot 首例的 `clip_info` 结束于第 1003 帧，但 PTH 人体数组仅有 580 帧。这可能是容器 frame-count、原片段索引或打包标签语义差异，**在解析清楚前禁止运行全量并计算数值**。
3. **两套 GT body model 不同。** AIST 是 SMPL 与 17 joints，H3.6M 是 SMPL-X 参数；要先固定统一的关节子集、root 定义、单位与 scale 处理，不能直接把原始 tensor 当成同一种标注。
4. **H3.6M 的全局平移并非独立原生 camera supervision。** HumanMM 补充材料说明其 H3.6M world translation 来自处理参考。因此该列可用于同一冻结 evaluator 下的相对方法比较，但必须标成 `provided reference trajectory`，不可升级为真实相机几何精度主张。

## 3. 最终实验问题与评测协议

### 3.1 本组实验回答什么

本组单人实验的核心问题是：在相同的多镜头 RGB 长序列上，Bridge3R 是否比其因果父方法和实际可运行的公开单人 world-HMR 方法更好地保持**跨 cut 的人体 root 位置、朝向和全局连续性**，同时不损害局部人体姿态。

它不回答多人 identity association，也不对相机轨迹作 GT 精度主张；这两部分由已经完成的三个多人数据集承载。

### 3.2 评测分两层，避免混淆

**层 A：Bridge3R 主评测（可直接、严格比较）。** 所有自身变体和实际跑通的外部方法使用同一 MP4、同一帧采样、同一自动 detector、同一输出 adapter 和同一 evaluator。正式表固定至少包含：

| 类别 | 指标 | 含义与约束 |
|---|---|---|
| 局部人体 | PA-MPJPE（共同关节集） | 去除全局位置、朝向和尺度差异后的姿态误差；用于验证 bridge 没有牺牲局部 HMR。必须明确 per-frame 或 sequence-level PA 定义。 |
| 全局人体 | Anchor-MPJPE（暂定名） | 仅在第一个 shot 以 GT 作一次全局 Sim(3) gauge 对齐，固定变换后评估整个长视频；不允许在每个 shot 重新对齐。它直接暴露跨镜头漂移。 |
| root 平移 | Anchor-RTE | 同一首 shot anchor 下的 root translation error，单位 m。 |
| root 朝向 | Anchor-ROE | 经同一首 shot 旋转 gauge 校正后的全局 root orientation error，单位 degree。 |
| 边界连续性 | Seam-root | 每个 cut 相邻帧的预测 root 增量与 GT root 增量的误差；按视频后再按 case 宏平均。 |
| 稳定性（补充） | Jitter、Foot Sliding | 仅在 SMPL/SMPL-X mesh、足部定义与 contact 判定均已验证后启用。 |
| 可用性 | Completion / valid-frame coverage | 解码、方法输出、adapter 和 evaluator 的成功情况必须独立报告，不能静默丢弃困难视频。 |
| cut detection（诊断） | Recall / Precision / F1、median boundary offset | `clip_info` 仅用于此 evaluator；模型输入不得读取它。 |

`Anchor-*` 是暂定、透明的名称，刻意不直接称为 HumanMM 的 `WA-MPJPE`。完成下一节的公式复核后，只有在定义逐项相同的情况下才映射为 `WA-MPJPE / RTE / ROE`；否则保留本名称并报告完整公式。这样既公平保留跨镜头难点，也不伪称复刻了未公开的 HumanMM evaluator。

**层 B：HumanMM 文献参考。** 单独表列 HumanMM 论文原始数值、其原始方法和评测器来源，表头明确为 `Source-reported; different processing/evaluator; not ranked`。这一层没有 `best` 样式，也不支撑“优于 HumanMM”的结论。

### 3.3 cut 与因果规则

- Bridge3R 与所有自身消融均使用同一个冻结的 causal detector；推理时不能读取 `clip_info`、GT 人体、原相机编号或 future frames。
- `clip_info` 只用于 detector 评估和 `Seam-root` 的边界索引。
- detector 阈值、视频 resize/FPS、track selection、帧缺失策略和 adapter 在看正式结果前冻结；若现有三数据集的 detector 配置已冻结，优先直接复用，不为本数据的正式指标调参。
- 外部基线运行其官方 continuous-video 设置。不得将 GT cut 传给外部方法，也不得为某一基线人工选择/修正单人 track；若其官方 tracker 输出多条轨迹，只使用在开发集预先固定的自动规则（最长且覆盖率最高的轨迹，平局按 track id）。

## 4. 方法队列与优先级

### 4.1 Bridge3R 及消融（必须完成）

每个正式子集均使用同一代码快照和同一输入 manifest，至少运行：

1. **Strict Human3R**：不断开 recurrent state 的原始 streaming backbone；
2. **Clean reset**：检测到 cut 后清空状态；
3. **Causal parent**：Bridge transaction 前的直接因果父方法；
4. **Bridge3R-Fixed**：固定的主 operating point；
5. **Bridge3R 消融**：去 coarse alignment、去 explicit correction-token branch、去 fine alignment / history transaction（按论文最终模块命名冻结）。

若 Bridge3R-Safe 在单人数据上也运行，必须独立成行，不能把它的 gate 结果与 Bridge3R-Fixed 混为同一个 `Ours`。

### 4.2 公开外部方法

| 优先级 | 方法 | 为什么值得比较 | 预期角色 | 首先要验证的条件 |
|---|---|---|---|---|
| P0 | **GVHMR** | HumanMM 主表中的公开 world-grounded 单人 HMR；本地已克隆 | 正式 direct baseline 候选 | 官方 checkpoint、连续视频推理、global root 输出和单位转换可验证 |
| P0 | **WHAM** | HumanMM 主表中的公开 world-HMR；本地已克隆 | 正式 direct baseline 候选 | 官方 checkpoint / SMPL 许可文件、SLAM 对切镜视频不崩溃、输出可适配 |
| P1 | **SLAHMR** | HumanMM 主表中的公开 camera-human joint optimization | 离线 external reference；与因果 Bridge3R 分开注明 | 自动单人 tracker、DROID-SLAM、长视频优化与输出语义可用；无法做到则只报告 pilot failure，不伪造结果 |
| P1 | **MultiShot (Pavlakos et al.)** | 与“多镜头单人 HMR”任务最直接相关，公开代码已在本地 | 离线任务相关 reference | 其 PHALP + OpenPose 前端必须全自动；不得人工给 `tracklet_id` 或给 GT shot；记录完整优化成本 |
| P2 | **PromptHMR** | 可在单人 RGB 上运行，可补充当前主线中已有的开放方法 | 局部人体 / availability 补充，不预设为 global leaderboard | 是否输出可验证的全局长程 root；否则仅报 PA-MPJPE、coverage 和定性图 |
| P2 | **TRACE** | 多人 tracking 方法也可处理单人输入 | 同上 | 没有独立物理相机轨迹时，不填 Anchor-RTE/ROE 或相机列 |

不将 SPEC、普通 image-HMR、仅 camera-coordinate pose 方法放入本组“跨镜头全局人体”主表：它们不解决跨切镜 world gauge，强行比较会稀释 Bridge3R 的问题定义。它们可在之后的 AIST / H3.6M 单镜头局部姿态附录中作为单帧/局部 baselines。

HumanMM 本身没有可执行的公开完整 pipeline，不进入运行队列；其数字仅按第 3.2 节进入文献参考表。

## 5. 执行阶段与停止条件

### 阶段 0：原始数据取得、保全与可复现清单（不跑模型）

1. 取得并记录原始 AIST++ 与 Human3.6M 的官方视频、每个相机的 calibration、原始人体 3D / SMPL（或可追溯 SMPL-X）标签及许可证说明。H3.6M 如受访问许可约束，必须使用持证账号/已有官方副本；不从非官方镜像补齐标定。
2. 对每个原始文件记录 SHA256、大小、来源版本、解压路径和相机/动作索引；原始数据只读使用，不修改文件。
3. 生成 `source_index_v1.json`，一行对应一个 `{sequence, time interval, camera}`，字段至少包括：`dataset`、`subject/action/sequence`、`camera_id`、`rgb_path`、`decoded_frame_count`、`rgb_to_gt_frame_map`、`intrinsics`、`extrinsics`、`gt_key`、`joint_convention`、`eligibility`、`exclusion_reason`。
4. 另行保留已上传的 HumanMM ZIP、其 JSON/PTH/MP4 的 hash 与数据清点结果，用于 protocol sanity check；不以其中 92 / 89 / 90 的缺失-GT 集合作为正式分母。

**停止条件：** 没有官方 RGB--GT--camera 闭环、hash 和完整 source index，不进入下一阶段。

### 阶段 1：自建 cut protocol、GT 帧轴与 evaluator 审计（不看方法胜负）

1. 从同一原始动作、连续时间轴的多视角视频中构造切镜序列：只切换相机，不改变人体时刻或场景；相邻 shot 之间保持连续帧索引。所有 RGB 帧、人体 GT 和 camera GT 均从同一个 `source_index_v1` 映射产生。
2. 正式主协议复用现有多人实验的时间结构：**CS150** = 两个连续 shot、每段 75 帧；第一个 shot 使用相机 A，第二个 shot 使用不同视角的相机 B。相机对由官方 extrinsics 的相对 viewing-angle 分层规则选择，并在查看方法输出前冻结。
3. repeated-cut 补充协议改为 fixed-duration：**MC150-3** = 三个 50 帧 shot；**MC150-4** = 四个 38/38/37/37 帧 shot。三种 protocol 均为 150 帧 / 5 秒、使用同一动作连续时段和 A→B→C（→D）相机序列，不能把不连续动作片段拼在一起。它们只用于隔离 cut 数量效应的补充压力测试。
4. 由确定性 hash 在符合长度、GT 和相机条件的 source index 中选择 AIST++ `pose_test` 内的 100 个正式 source，并从 `pose_val`（不足时 `pose_train`）选择 12 个不报告的 pilot source；Human3.6M 使用 S9/S11 的全部合格 source、上限为 60 case。若官方数据不足预定数目，先冻结实际最大数并在论文报告，不能运行后补换 case。
5. 对所有候选 case 解码并核验 RGB 帧、人体 GT 帧和相机 GT 帧的逐帧对应；保存至少 20 个 3D joints / mesh 到 RGB 的投影 overlay，检查重投影误差、坐标轴、crop 与每个 cut 的相机切换。
6. 在 AIST 验证 SMPL scaling / translation、17-joint 定义和单位；在 H3.6M 验证官方 body 3D、相机外参、SMPL-X conversion（如使用）的单位与语义。冻结共同最小关节集。
7. 明确 PA、首-shot anchor、RTE、ROE、camera RPE、seam、jitter、foot sliding 的公式、加权方式和缺失帧规则；与 HumanMM 论文逐项比对，把不相同的地方写入 protocol card。

**通过条件：** 每个要进入正式表的 case 都有唯一、可复现、人工 spot-check 通过的 RGB frame → human GT frame → camera GT frame mapping；joint/单位/坐标系通过可视化和数值 sanity check。  
**阻塞条件：** 任一原始数据源无法证实 frame map 或相机投影不通过。该数据源不以填充数字方式进入论文；另一数据源可独立继续。

### 阶段 1B：若仅能先取得 HumanMM 包时的上游标定恢复（备用分支）

若原始数据尚未完整取得而需先审计 HumanMM 包，当前压缩包没有本地相机标定，但这不等于标定必然不可恢复。恢复只接受**原始数据集的官方标定**；不得以 SLAM、COLMAP、拟合 GT 人体或模型预测得到的伪相机轨迹替代 GT。该分支仅用于审计 HumanMM 输入；正式构造仍遵循阶段 0--1。

1. **ms-AIST。** 下载/定位 AIST++ 官方 annotation API 中与标签所列 `cam_json` basename 对应的 `annotations_dir/cameras/setting*.json`。每个样本已有原动作名 `video_name`、对应 setting、以及每段的 `cXX` camera id，理论上可构造逐帧的 piecewise intrinsics / extrinsics。
2. **ms-H3.6M。** 定位官方 H3.6M camera calibration metadata；结合 `h36m_base_filename`、subject / action 信息、`h36m_processed_name` 与 `clip_info` 的 camera id，构造每段相机参数。该步骤依赖阶段 1 先解决的 frame map。
3. 对每种数据至少随机审计 20 个 case：将提供的 3D joints / SMPL body 用恢复的相机投影到 RGB，检查重投影误差、左右方向、分辨率/crop 和 cut 两侧相机切换是否一致；把 overlay 和数值写入 calibration recovery card。
4. 只有所有审计条件通过后，新增 `camera-RPE`、camera rotation seam 与分段 camera pose 定性图。由于 GT 相机在 cut 处本来就会跳变，指标评估的是每个 shot 内的 pose 与相邻 shot 的已知相对变换，不能把真实 cut 当成平滑 camera-path 的 ATE 任务。

**决策规则：**

- 官方标定与 RGB 的投影一致：数据升级为“单人 camera--human 多镜头评测”，但仍和多人数据分表；
- 只有部分 setting 可验证：仅使用已验证 case，并在 manifest 披露排除；
- 没有官方标定或投影无法通过：保留人体 root / pose / seam 评测，不报告 camera accuracy。数据仍然有效，只是它回答的是较窄、也依然直接相关的“跨视角人体连续性”问题。

### 阶段 2：冻结 pilot（12 个自建 case，不调正式结论）

每个 `{AIST++, Human3.6M} × {CS150, MC150-3, MC150-4}` 从 development pool，通过稳定 hash 选 2 条 case，合计 12 条。AIST++ pilot 必须与 `pose_test` source 不重叠；H3.6M 若没有额外已授权 source，则只做不调参的接口审计并记录其非独立性。pilot 只用于检查接口，绝不根据成绩挑案例或改阈值。

对每个 case：

1. Bridge3R-Fixed、Strict Human3R 和 causal parent 先跑，验证 detector 不读取标签且输出连续；
2. 将预测转换到共同 joint set，以首 shot 一次 anchor 计算局部、全局、root 和 seam 指标；
3. 输出 RGB/mesh/root trajectory/cut marker 的小型定性审计图；
4. 依次试跑 GVHMR、WHAM，再测试 SLAHMR、MultiShot、PromptHMR、TRACE 的官方输入输出链；
5. 每个方法记录 runtime、GPU peak memory、检测/跟踪输出、完成比例、失败日志和 adapter commit，不比较排名。

**通过条件：** 至少 Bridge3R 系列与 GVHMR / WHAM 在每个被保留子集能自动处理 pilot，输出单位和坐标语义明确，且 evaluator 对已知的 identity / shift sanity checks 有预期反应。  
**失败处理：** 外部方法若官方 pipeline 无法在 pilot 自动运行，只保留“availability audit”记录，不对 Test 手工修复；不会阻塞 Bridge3R 自身完整消融。

### 阶段 3：冻结正式 manifests 与运行卡

pilot 通过后，创建不可修改的 `v1` manifests、各方法 adapter freeze JSON、环境 lock、checkpoint SHA256 与输入预处理说明。

运行顺序：

1. **CS150 主单人表**：自建 AIST++ 100 条及自建 Human3.6M（S9/S11、至多 60 条）。这是与已有多人 CS150 最一致、最适合正文的单次 cut 证据。
2. **MC150-3 与 MC150-4 压力测试**：自建 AIST++ 各 100 条，以及 Human3.6M 的同一批合格 source（至多各 60 条）。在相同 150 帧总时长下作为 repeated-cut 补充表和 error-vs-cut-count 图；不把它们解释为 equal-shot-duration 长视频结果。
3. 任一方法均在每个 frozen manifest 的**所有** case 上运行一次；不可评估的输出保留在分母中并报告 completion / coverage。任何方法不因失败替换或缩小序列。

建议在上述三组内分开报告 AIST++ 与 Human3.6M；不把不同 shot 数视频混成同一个平均分母。若某源在预冻结的 eligibility audit 后少于 100 条，固定该实际数目并在表中明确分母，绝不为“凑数”事后替换。

### 阶段 4：全量 Bridge3R 消融与外部基线

1. 先完成 Bridge3R 系列的全部 2 / 3 / 4-shot 结果，得到最清楚的因果证据：Strict → reset → parent → Bridge3R 与关键模块消融。
2. 运行 P0 的 GVHMR 和 WHAM；每一种仅使用官方 pretrained checkpoint 和冻结 adapter。
3. SLAHMR 与 MultiShot 先完成完整 pilot，再决定是否进入所有 2-shot case。它们若进入，按“offline full-sequence optimization”单独标注，不能与 Bridge3R 的 causal runtime 混称。
4. PromptHMR 与 TRACE 在全部 2-shot case 运行；只有输出满足第 3.1 节的 global semantics 时才进入 global 表，否则放入 local-pose/coverage 附表和定性比较。
5. 只有运行过相同 manifest、相同 evaluator 的方法可以进入直接比较表；论文中每个 `N/A` 必须说明“没有该输出”还是“运行失败”。

### 阶段 5：统计、论文与发布物

1. 每条视频先得到 frame 指标，再 video-macro；AIST 与 H3.6M 分开做 nonparametric bootstrap CI（最高独立单元为 video）。
2. 对 Bridge3R 相对 causal parent 报 paired median / mean relative change、improvement fraction、worst-case ratio 和 95% CI；不只报平均值。
3. 正文放：2-shot 表、Bridge3R 关键消融、一个跨 cut 轨迹定性图；3/4-shot 表、detector、完整 coverage/失败、外部方法运行卡及 HumanMM source table 放补充材料。
4. 新建版本化实验目录，保存 manifest、代码 commit、环境、checkpoint hash、命令、stdout/stderr、逐 case NPZ、逐 case metric JSON、汇总 CSV、图、表和 claim ledger；论文仅引用由该目录自动生成的数值。

## 6. 计划中的交付物

```text
data/manifests/ms_motion/v1/                 # 冻结 case 列表；原始数据不被修改
Movie3R/experiments/ms_motion_v1/            # 所有方法的运行卡、预测、日志和逐 case 结果
Movie3R/evaluation/ms_motion_protocol_v1/    # frame map、joint conversion、指标公式和单元测试
ICLR-paper/bridge3r_iclr2027/                 # 自动生成 LaTex 表、图、protocol card、claim ledger
```

在第 1--2 阶段尚未通过之前，不创建“正式结果”表，不向摘要或主结论添加数值。

## 7. 论文中预期的合理呈现

正文的核心叙事仍以三个真实多人数据集为主。单人 ms-Motion 的作用是补强“任务本身并不限于多人关联，Bridge transaction 对跨视角切换的人体全局连续性同样有效”。建议：

- 正文新增一个紧凑的 `Single-person multi-shot transfer` 小节：2-shot 的 Bridge3R 消融和实际跑通的 P0 baselines；
- 补充材料给出 3/4-shot、逐切镜退化曲线、全部协议细节、运行失败与 HumanMM source-reported 表；
- 对离线方法措辞为“same-input offline reference”，对 Bridge3R 强调“causal / first post-cut observation / no future frames”；
- 绝不以缺失相机标定暗示 dense scene 或 camera-trajectory SOTA；这与论文已锁定的 Camera--Human 任务边界一致。

## 8. 下一步的唯一前置动作

下一轮开始执行时，先做**阶段 0 与阶段 1**，而非立即跑任何模型：先把 H3.6M 的 `frame map` 与统一 body evaluator 证实。该 gate 通过后，再按阶段 2 的 12-case pilot 开始；届时会先汇报 pilot 的可用性与协议验证，再请求/使用缺失 checkpoint 或 SMPL 许可文件。
