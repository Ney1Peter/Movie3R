# Bridge3R 单人多镜头数据：最小官方下载与构造规范

**用途：** 本文档可直接交给另一个 AI / 工程工具执行。它定义了应下载什么、绝不能下载什么、如何冻结数据清单、如何从官方原始多视角数据构造 Bridge3R 的单人跨镜头评测集，以及何时必须暂停向用户确认。  
**状态：** 仅规范，尚未执行下载、解压、构造或推理。  
**工作目录：** `/data/wangzheng/iJCV-CODE`  
**论文任务边界：** 单人、同一场景、同一连续动作、不同物理相机之间的视角切换；不构造跨场景 cut，不用伪相机 GT。

---

## A. 必须遵守的规则

1. 只使用官方数据源、官方 annotation 和用户拥有授权的数据访问方式。不得使用网盘、第三方镜像、爬虫绕过登录、泄漏账号或共享凭证。
2. **不得下载 AIST++ 或 Human3.6M 全量数据。** 先下载轻量标注并冻结 source manifest，再下载 manifest 列出的 RGB 文件。
3. 不得在模型推理、外部 baseline、用户指标或可视化结果出来后替换数据 case、相机对、时间窗口或 cut 边界。
4. 原始数据目录只读使用；不修改、重编码或覆盖原始 MP4。所有派生视频和标签写入新的 `data/bridge3r_singleperson_v1/`。
5. 不得把 SLAM、COLMAP、人体 GT 拟合或模型预测生成的相机轨迹称作 camera GT。camera GT 只可来自 AIST++ / Human3.6M 的官方 calibration。
6. 禁止向方法输入 GT cut、GT 人体、GT camera、相机编号、真实身份或 future frames。它们只服务于 dataset builder 和 evaluator。
7. 每一步都记录源 URL、版本、SHA256、命令、成功/失败日志和实际磁盘占用；发生缺失、下载失败或 GT 不一致时保留 failure record，不静默换样本。

---

## B. 最终要得到的数据集

### B.1 协议

所有派生视频的目标帧率为 **30 FPS**。视频保持原始空间分辨率；各方法在自己的官方前处理阶段 resize，不在数据构造阶段偷偷裁剪。

| Protocol | 输出帧数 | shot 数 | 每段长度 | 含义 | 论文位置 |
|---|---:|---:|---:|---|---|
| `CS150` | 150 | 2 | 75 帧 | 单次真实视角切换 | 正文单人主表 |
| `MC150-3` | 150 | 3 | 50 / 50 / 50 帧 | 固定 5 秒内的两次视角切换 | 补充材料压力测试 |
| `MC150-4` | 150 | 4 | 38 / 38 / 37 / 37 帧 | 固定 5 秒内的三次视角切换 | 补充材料压力测试 |

对同一 source sequence，三个 protocol 都使用相同的连续 5 秒窗口与同一起点。例如：

```text
CS150:  frame [0, 74]   from camera A; frame [75,149]   from camera B
MC150-3: frame [0, 49]   from camera A; frame [50, 99]  from camera B; frame [100,149] from camera C
MC150-4: frame [0, 37]   from camera A; frame [38, 75]  from camera B; frame [76,112] from camera C; frame [113,149] from camera D
```

`MC150-3/4` 的目的是在**相同 5 秒时间预算**下隔离 cut 数量对累积误差的影响，而不是保持每个 shot 的时长不变。因此它们只能作为补充压力证据，不能替代 CS150 主表，也不能将三种 protocol 的绝对分数直接混合平均。

同一输出帧位置的所有 camera view 必须对应同一个原始动作时刻；不能将动作的不同时间片段拼接为伪连续运动。每段在相机切换时改变 RGB 与 GT camera，但人体 GT 的时间轴连续。

### B.2 规模与统计独立性

| 数据来源 | Source-level Test | Pilot / audit | 说明 |
|---|---:|---:|---|
| AIST++ | 100 个不同原始动作序列 | 12 个与 Test source 不重叠的序列 | 三个 protocol 可由同一 100 source 生成；统计独立单位是 source sequence |
| Human3.6M | S9、S11 中所有合格 source，最多 60 个 | 仅作不调参接口审计；若需严格独立开发 source，使用额外已授权 source | 2 subjects × 15 actions × 2 subactions = 至多 60 个 source；不得为了凑 100 而重复同一 source 并当成独立样本 |

所有数值在 AIST++ 与 H3.6M 内部分开汇总、分开统计；不跨数据集平均绝对误差。CS150 与 MC150-3/MC150-4 也不合并为同一平均分母。

---

## C. 建议目录（不得混放原始与派生数据）

```text
/data/wangzheng/iJCV-CODE/data/
├── raw/
│   ├── aistplusplus_v1/
│   │   ├── downloads/                 # 官方 ZIP、下载日志、SHA256
│   │   ├── annotations/               # cameras, motions, splits, ignore_list, keypoints3d
│   │   └── videos_selected/           # manifest 指定的原始 cXX MP4；只读
│   └── h36m_v1/
│       ├── downloads/                 # 官方 portal 下载包；只读保留至验收完成
│       ├── metadata/                  # 官方 calibration, 3D world positions, sequence metadata
│       └── videos_selected/           # S9/S11 中实际被 manifest 使用的 RGB 文件；只读
├── manifests/
│   └── bridge3r_singleperson_v1/
│       ├── source_index_v1.json
│       ├── aist_source_selection_v1.json
│       ├── h36m_source_selection_v1.json
│       ├── aist_{cs150,mc150_3,mc150_4}_test_v1.json
│       ├── h36m_{cs150,mc150_3,mc150_4}_test_v1.json
│       └── DOWNLOAD_REPORT_v1.md
└── bridge3r_singleperson_v1/
    ├── videos/{aist,h36m}/{cs150,mc150_3,mc150_4}/
    ├── labels/{aist,h36m}/{cs150,mc150_3,mc150_4}/
    ├── frame_maps/{aist,h36m}/
    ├── calibration/{aist,h36m}/
    ├── audits/
    └── DATASET_CARD_v1.md
```

不要自动删除官方压缩包或原始视频；在校验完成后，由用户决定是否删除可再下载的 archive 以回收空间。

---

## D. AIST++：最小下载清单与方法

### D.1 用户确认前置条件

在下载任何 AIST RGB 视频前，必须让用户确认已阅读并接受 [AIST Dance Video Database Terms of Use](https://aistdancedb.ongaaccel.jp/terms_of_use/)。若未确认，**只能下载公开 annotation，不下载视频**。

官方参考：

- [AIST++ download page](https://google.github.io/aistplusplus_dataset/download.html)
- [AIST++ API](https://github.com/google/aistplusplus_api)
- [AIST++ annotations release v1.0](https://github.com/google/aistplusplus_dataset/releases/tag/v1.0)

### D.2 先下载的轻量 annotation（必须）

下载到 `data/raw/aistplusplus_v1/downloads/`，再解压到 `annotations/`：

| 文件 | 官方 URL | 大小（官方 release） | 用途 | 是否必须 |
|---|---|---:|---|---|
| `cameras.zip` | `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip` | 约 27 KB | 9 台相机的内参、外参、环境 mapping | 是 |
| `motions.zip` | `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip` | 约 79 MB | SMPL pose、root translation、scale | 是 |
| `splits.zip` | `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/splits.zip` | 约 16 KB | 官方 train / test split | 是 |
| `ignore_list.txt` | `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/ignore_list.txt` | 约 1 KB | 官方建议排除的劣质重建 | 是 |
| `keypoints3d.zip` | `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/keypoints3d.zip` | 约 877 MB | 独立 3D joint 与投影 sanity check | 强烈建议 |

**不要下载：** `keypoints2d.zip`（约 1.3 GB，当前非必需）、AIST++ 全体视频、全部图像帧、音乐和任何未被 manifest 使用的 video views。

完成后核验目录至少为：

```text
annotations/
├── cameras/
│   ├── mapping.txt
│   └── setting*.json
├── motions/*.pkl
├── keypoints3d/*.pkl                # 若下载推荐项
├── splits/
└── ignore_list.txt
```

### D.3 在下载 RGB 前冻结 AIST source manifest

1. 读取官方 `test` split；移除 `ignore_list.txt` 中的 sequence。
2. 仅保留 motion label 完整、所有必要 camera calibration 存在，且能提供至少 **5 秒**连续时间窗口的 source。AIST++ GT 时间轴是官方定义的 60 FPS，因此最低要求为 **300 个有效 GT tick**。
3. 在 `pose_test` 内按稳定排序键 `SHA256("bridge3r-aist-v1|<sequence_name>")` 由小到大排序；从 137 个合格 source 中取前 100 个为正式 Test。其余 `pose_test` source 不用于 adapter 调试或正式替换。12 个 pilot 必须从 `pose_val`（不足时 `pose_train`）中以同样的确定性规则另行抽取，且只用于接口/可视化审计、不报告任何论文数值。保存完整候选、split、排除原因和排序 hash。
4. 为每一个 source 从官方 9 台相机中确定有序四元组 `A,B,C,D`：
   - 从 official `setting*.json` 读取相机中心与光轴；
   - 优先选择覆盖低 / 中 / 高相对 viewing-angle 的 camera pair，且四台相机不重复；
   - 使用排序 hash 作为平局规则；记录相机编号、相对角度、设置文件与选择理由；
   - 不得根据任何方法预测质量、遮挡程度、最终指标或人工观感改变相机组合。
5. 为每个 source 由 hash 选择一个合法的 5 秒连续起点。记录起点的 60 FPS GT index 与绝对时间戳；CS150 / MC150-3 / MC150-4 共用该起点。
6. 生成并人工复核 `aist_source_selection_v1.json` 后加上文件 hash。**manifest freeze 之后才可下载 RGB。**

`source_index_v1.json` 的最小字段：

```json
{
  "source_id": "aist:<sequence_name>",
  "dataset": "AIST++",
  "split": "test",
  "motion_path": ".../motions/<sequence_name>.pkl",
  "keypoints3d_path": ".../keypoints3d/<sequence_name>.pkl",
  "camera_setting": "settingX.json",
  "camera_ids": ["c01", "c04", "c06", "c09"],
  "source_start_gt_index_60fps": 0,
  "source_start_seconds": 0.0,
  "output_fps": 30,
  "eligibility": true,
  "exclusion_reason": null
}
```

### D.4 只下载 manifest 中的 AIST RGB 文件

官方视频 URL 格式为：

```text
https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/<video_name>.mp4
```

`<video_name>` 的生成规则：将 source sequence 名称中的 `cAll` 替换为 manifest 所列 `c01`--`c09`。例如：

```text
gBR_sBM_cAll_d04_mBR0_ch01  +  c03
→ gBR_sBM_c03_d04_mBR0_ch01.mp4
```

为每个 100-case source 下载 4 个 camera views，约 400 个 MP4；pilot 若与 Test source 不重叠，额外下载 48 个 MP4。生成 `aist_selected_urls.txt` 后逐行下载、断点续传、逐文件 SHA256 校验。下载失败重试最多 3 次；仍失败时写入报告并停止该 source，不得换成另一条 source。

**时间对齐要求：** AIST++ annotation 的权威时间轴为 60 FPS。原始 MP4 的容器 FPS 可能略有不同，不能直接把“解码第 i 帧”当作“GT 第 i 帧”。必须保存 `RGB PTS ↔ 60 FPS GT index ↔ 输出 30 FPS frame` 的显式 frame map，并使用官方 API / ffmpeg 的时间戳方式验证。

---

## E. Human3.6M：最小下载清单与方法

### E.1 用户确认与官方访问前置条件

Human3.6M 的下载需要用户拥有的官方账号与 EULA 授权。官方入口：

- [Human3.6M official website](http://vision.imar.ro/human3.6m/description.php)
- [Human3.6M official login](http://vision.imar.ro/human3.6m/main_login.php)

不得要求用户向 AI 提供密码。用户应在浏览器/门户自行登录和确认 EULA；AI 只在用户已授权的下载会话或用户提供的合法本地文件路径范围内工作。

### E.2 在门户中只选择这些内容

| 项目 | 范围 | 是否必须 | 原因 |
|---|---|---|---|
| RGB videos | **S9、S11**，四台同步 DV camera | 是 | 标准 test subjects；用于生成 multi-shot RGB |
| 3D world joint positions | S9、S11；官方 world-coordinate / `D3_Positions` 数据 | 是 | 连续人体 3D GT 与 root trajectory |
| Camera calibration / camera parameters | S9、S11 对应的官方 calibration | 是 | 各 shot 的真实 intrinsics / extrinsics |
| Sequence / frame metadata | S9、S11 | 是 | 50 Hz video、pose 序列和文件名映射 |
| 2D poses / bbox / segmentation | 不下载 | 否 | 当前不需要；可由 3D + official calibration 验证投影 |
| Training subjects S1/S5/S6/S7/S8 | 不下载 | 否 | 不是正式 Test source |
| TOF、mixed reality、背景、全量 image frames、其他传感器 | 不下载 | 否 | 与当前 protocol 无关 |

H3.6M 的动作文件名和 camera suffix 必须从下载包内的 official metadata 自动枚举，不能在脚本中假设固定文件名。候选 pool 为 `S9,S11 × 15 actions × 2 subactions`，至多 60 个 source sequences；每个 source 最多使用 4 个同步相机视频。

若门户支持单文件/单序列下载：先从 metadata 制作精确的视频文件清单，下载最多 60 × 4 = 240 个 RGB 视频。  
若门户只提供按 subject 的 `Videos` archive：只下载 S9 和 S11 的 `Videos` 项，以及上表必要 metadata；下载后仅复制 manifest 使用的视频到 `videos_selected/`。不要下载任何其他 subject 或模态。

### E.3 冻结 H3.6M source manifest

1. 扫描 S9/S11 official 3D positions、camera metadata 与 RGB 文件名，建立 `{subject, action, subaction, camera}` source index。
2. 仅保留四个 camera stream 都存在、3D pose 与 RGB 有可验证的同步时间轴、可提供至少 5 秒连续窗口的 source。
3. 对合格 source 按 `SHA256("bridge3r-h36m-v1|<subject>|<action>|<subaction>")` 排序；使用所有合格 source，最多 60 条。不要挑选“更容易”的动作。
4. 每条 source 使用全部四个 official cameras，并由 hash 冻结有序 tuple `A,B,C,D`；相机顺序应覆盖视角跨度，不根据任何模型结果修改。
5. 使用 hash 冻结 5 秒起点。H3.6M RGB 通常是 50 Hz；必须从 official metadata 确认 3D pose 时间轴，构造 `RGB PTS ↔ GT tick ↔ 输出 30 FPS` frame map，而不是固定假设二者数组下标相同。
6. 保存 `h36m_source_selection_v1.json`、所有排除原因及文件 hash。

Human3.6M 只有最多 60 个独立 source，因此 Test 表的 `N` 应如实报告为实际合格数。若使用多个不重叠窗口构造不同 protocol，它们仍属于同一个 source cluster；bootstrap / 显著性统计的独立单位是 source sequence。

---

## F. 从冻结 source manifest 构造派生视频与标签

### F.1 统一的逐帧 manifest

每个派生 case 必须有一个不依赖方法输出的 JSON，例如：

```json
{
  "case_id": "aist_cs150_000",
  "dataset": "AIST++",
  "protocol": "CS150",
  "source_id": "aist:gBR_sBM_cAll_d04_mBR0_ch01",
  "output_fps": 30,
  "num_frames": 150,
  "cuts_after_output_frames": [74],
  "shots": [
    {"output_range_inclusive": [0, 74], "camera_id": "c01"},
    {"output_range_inclusive": [75, 149], "camera_id": "c06"}
  ],
  "frames": [
    {
      "output_frame": 0,
      "source_time_seconds": 12.500000,
      "source_gt_index": 750,
      "source_rgb_pts": 12.500000,
      "camera_id": "c01",
      "intrinsics_id": "...",
      "extrinsics_id": "..."
    }
  ],
  "uses_gt_at_inference": false
}
```

`cuts_after_output_frames` 记录 cut 前最后一帧，便于 evaluator；模型输入只能看派生 RGB 视频。

### F.2 视频构造

1. 以 manifest 的 `source_time_seconds` 从对应相机原始 RGB 提取一帧；保持时间戳记录。
2. 用原始动作连续时间轴生成 30 FPS 输出序列。AIST++ 使用其权威 60 FPS annotation 轴，H3.6M 按官方 metadata 确定采样轴；输出 frame map 必须可逆查到原始 timestamp 和 GT tick。
3. 按 shot 选择不同相机的**同一时刻** RGB frame，按 `A→B(→C→D)` 拼接为最终 MP4。不得跨相机做颜色、几何、人体或背景融合；不得在 cut 帧插值或 cross-fade。
4. 生成无声 MP4；保存原始分辨率、30 FPS、逐帧连续编号。视频 encoder / pixel format / command 写入 case metadata。若某帧无法精确解码，整个 case 标为 unavailable，不通过静默删帧改变 cut 位置。

### F.3 标签构造

每个 case 输出一个 `NPZ` 或 `PT` 标签文件，至少包含：

```text
joints_world[T, J, 3]          # 官方世界坐标人体 joints
root_world[T, 3]
root_orientation_world[T, 3, 3]  # 官方/可追溯 body orientation；若源没有，明确 N/A
camera_K[T, 3, 3]
camera_R_world_to_cam[T, 3, 3]
camera_t_world_to_cam[T, 3]
camera_id[T]
source_gt_index[T]
source_timestamp_sec[T]
cut_after[T-1 bool]
valid_frame[T bool]
joint_convention
units
```

AIST++ 可额外保存 `smpl_pose`、`smpl_trans`、`smpl_scaling` 与官方 17-joint 3D。H3.6M 保存官方 world 3D joints、camera data 与可验证的 root 定义；若需要 SMPL-X，只能作为方法预测的 adapter，不将 HumanMM 提供的 pseudo translation 注入正式 GT。

### F.4 保留的正式评测能力

官方 calibration 投影审计通过后，本数据支持：

- 局部人体：共同关节集 PA-MPJPE；
- 全局人体：首 shot 单次 gauge anchor 后的 full-sequence MPJPE / root translation / root orientation；
- 跨 cut：Seam-root、seam orientation、局部姿态 non-regression；
- 相机：各 shot 的 camera rotation / translation error、已知相邻 camera 相对变换误差；
- detector：以 `cuts_after_output_frames` 为 GT 的 recall / precision / F1；
- repeated cuts：CS150、MC150-3、MC150-4 的 fixed-duration error-vs-cut-count 曲线。

不得将本数据的绝对相机指标与 EgoBody、EgoHumans、Harmony4D 做无解释平均；分数据集报告，跨数据集只汇总相对 causal-parent 的 paired effect。

---

## G. 下载与构造验收清单

在开始任何模型推理前，必须全部通过：

- [ ] 用户明确确认 AIST Dance Video Database Terms of Use；Human3.6M 使用的是用户官方授权。
- [ ] 没有下载全量 AIST++ / H3.6M；`DOWNLOAD_REPORT_v1.md` 给出每类文件、数量、总字节数和被刻意省略的内容。
- [ ] 所有官方 annotation / RGB 文件都有 URL、下载日期、文件大小、SHA256。
- [ ] AIST++ 100 Test source 与 12 pilot source 不重叠；H3.6M 实际 Test source 数、排除原因、source cluster 均固定。
- [ ] 每个 Test case 的相机 tuple、时间起点、cut、原始帧索引均由 hash 在看模型前确定。
- [ ] 每个派生 MP4 都有等长 labels 与逐帧 `frame_map`；三个 protocol 的 frame count 都严格为 150。
- [ ] 至少随机抽查 20 个 AIST++ 和 20 个 H3.6M case：将 3D joints/mesh 按 official camera 投影到 RGB；确认无整体镜像、尺度、时间漂移或相机 id 错配。
- [ ] AIST++ 的 `smpl_scaling`、单位与坐标轴已记录；H3.6M 3D pose 的世界坐标、采样率、root 定义已记录。
- [ ] `DATASET_CARD_v1.md` 写清 dataset source、协议、case 数、相机选择、frame-map、许可、已知限制和禁止的结论。
- [ ] 所有原始数据与派生数据的路径在 manifest 中是相对工作区可复现路径；没有 hard-coded 其他机器绝对路径。

任一检查失败时：停止在数据审计阶段，输出问题报告；不得启动全量 Bridge3R、TRACE、PromptHMR、GVHMR、WHAM、SLAHMR 或 MultiShot 实验。

---

## H. 交付给后续实验的文件

完成后必须交付：

1. `source_index_v1.json` 与各 dataset / protocol 的 frozen test manifest；
2. `aist_selected_urls.txt`、H3.6M 官方下载项清单、下载日志和 SHA256；
3. 600 个以内的派生视频（AIST：100 × 3；H3.6M：最多 60 × 3）及等长标签；
4. 逐帧 RGB/GT/camera frame map；
5. calibration projection audit 图、数值报告与失败 case 列表；
6. `DATASET_CARD_v1.md` 和 `DOWNLOAD_REPORT_v1.md`；
7. 一段简短 handoff：实际 case 数、占用空间、未解决风险、是否可开始 12-case pilot。

只有上述交付物齐全，下一位执行实验的 AI 才可以开始 Bridge3R 与外部 baseline 的 pilot。
