# Bridge3R AIST++ 单人跨视角评测计划（v1）

**日期：** 2026-08-26  
**状态：** 数据输入、派生 CS150、共同 evaluator 与投影审计已验收。PromptHMR 的 12-case pilot 已通过，并已启动冻结的 100-case Test；正式数值仍待自动 ledger 完成。GVHMR 的 12-case pilot 暴露跨硬切 tracker 连续性失败，已按本计划降级，不启动其 global Test。  
**论文边界：** 单人、同一真实场景、同一连续动作、不同物理相机之间的视角切换（Camera--Human）。不声称重建稠密场景，不构造跨场景编辑。

## 1. 本轮结论

上传包 `data/HumanMM_AIST_Bridge3R_v1_20260826/` **可以使用，但当前是“经验证的原始冻结输入”，不是已经可打分的最终 benchmark。**

### 1.1 已通过的输入审计

| 项目 | 结果 | 证据 |
|---|---:|---|
| ZIP 安全性 | 4,186 个条目；无绝对路径、无 `../` 路径 | ZIP 成员路径检查 |
| 包内完整性 | `CHECKSUMS.sha256` 已完成全量校验 | 包内 checksum manifest |
| 冻结清单完整性 | 4 个 manifest 文件 hash 一致 | `aist_manifest_freeze_sha256.txt` |
| 正式 source | 100 个、全部来自官方 `pose_test` | `aist_source_selection_v1.json` |
| pilot source | 12 个、全部来自 `pose_val`，与 Test 不重叠 | 同上 |
| reserve | 37 个 `pose_test` source，冻结但不进入 Test | 同上 |
| RGB | 448 个官方 MP4 = 100 Test x 4 view + 12 Pilot x 4 view | `aist_selected_files_v1.tsv` 与本地目录 |
| GT | 每个冻结 source 都具备官方 motion、3D keypoints、相机 setting 和合法 300-tick 窗口 | `source_index_v1.json` |
| 容器/时长 | 448/448 通过 FFprobe、字节数和五秒窗口审计 | `aist_rgb_ffprobe_audit_v1.json` |

三个固定协议均在同一 5 秒、150 帧（30 FPS）窗口上定义：

| 协议 | 输出结构 | 真实 cut 后帧 | 论文位置 |
|---|---|---|---|
| `CS150` | 75 / 75 | 74 | 单人主表 |
| `MC150-3` | 50 / 50 / 50 | 49, 99 | 补充材料压力测试 |
| `MC150-4` | 38 / 38 / 37 / 37 | 37, 75, 112 | 补充材料压力测试 |

source、窗口起点和四相机 tuple 已经按 hash 冻结。**不得重新运行或覆盖 `aist_source_selection_v1.json`，不得根据任何方法的预测、失败或得分替换正式 case。** 早期 eligibility audit 中的“600 tick”记录只是历史审计；正式冻结选择正确使用 300 tick / 五秒规则。

### 1.2 仍未完成的硬门槛

以下项目未完成前，不得启动 formal Test，更不得把数值写入论文主结论。

1. 建立 `RGB PTS -> 60 FPS GT tick -> 30 FPS output frame` 的逐帧显式 frame map。
2. 基于该映射派生 `CS150`、`MC150-3`、`MC150-4` 视频与逐帧 human/camera 标签。
3. 用官方 calibration 在至少 20 个不同 source 上做 3D joints/mesh 到 RGB 的投影审计。
4. 冻结 evaluator 的关节定义、单位、gauge、缺失帧和聚合规则，并通过单元测试。
5. 编写 `DATASET_CARD_v1.md`、构造 checksum、版本/许可证记录和最终 handoff。

因此，本包回答了“**原始 AIST++ 输入是否完整、合规且可复现**”：答案是肯定的；尚未回答“**派生 benchmark 是否能被严谨评测**”。

### 1.3 2026-08-26 执行记录（只追加，不追溯改写）

- 派生根 `data/bridge3r_singleperson_v1/` 已生成 prediction-only runtime manifest 与 evaluator-only label manifest；AIST++ CS150 固定为 150 帧、30 FPS、隐藏 cut 位于 frame 74 后。所有外部方法只收到紧凑 RGB runtime row。
- PromptHMR 官方 whole-video pipeline 在 12 个 `pose_val` pilot source 上全部完成：每例 150/150 有效帧、同一 native track 覆盖 cut 两侧各 75 帧；camera-to-world 旋转通过刚体检查。pilot audit 已允许其进入 Test。
- GVHMR 官方 demo 在 12 个 pilot source 上均生成 150-frame human output；但其官方 `Tracker.get_one_track` 以全片最大面积 raw track 选人、随后插值缺帧。在 2/12 source 中，该原生 track 只存在于 cut 的一侧，另一侧完全没有 raw detection。故其平滑后的 150 帧输出不能被宣称为经验证的跨镜头同一人 global track。
- 依照“raw identity/coverage 不通过即不进 global table”的规则，GVHMR 不启动 100-case global Test；其 12-case availability audit 和不报告 camera trajectory 的原因保留至补充材料。该决定不基于任何 Test 成绩。
- PromptHMR Test 使用同一 100-source `pose_test` manifest，并在执行前锁定三段不相交的 line interval（1--25、26--50、51--100）及相应输出根。最终 ledger 仅消费该锁指定的路径，禁止从冗余输出中择优。

## 2. 不可改变的实验原则

1. 所有方法在同一 protocol 上读取同一个派生 MP4、同一输出帧时间轴与同一 frozen case manifest。
2. Test 只包含 100 个 `pose_test` source；pilot 只用于接口与可视化审计，绝不调正式阈值、选 case 或报告主表数值。
3. 输入中不提供 GT cut、GT camera、camera id、GT human、身份或 future frame。`cut_index` 仅供测评，Bridge3R 自身只使用冻结的因果 detector。
4. 任何方法失败都保留在分母中，以 `completion` 和 `valid-frame coverage` 报告；不能替换、静默过滤或人工修正难例。
5. 一条 source 是一个统计独立单元。三种 cut protocol 是同一 source 的不同压力版本，不相互混合平均。
6. 只把语义可验证的 world/root 输出放进全局几何表。不能验证 global gauge 的方法最多进入局部姿态、coverage 和定性附表。
7. HumanMM 及 Multi-THuMBS 没有可复现的完整 baseline/evaluator；它们的论文数字只能作为 **source-reported / different evaluator** 文献参考，不能与本计划产出的数字同表排名，也不能宣称数值上“打败 HumanMM”。

## 3. 目录、版本与冻结物

原始数据只读，派生数据和实验输出与原始输入严格分开：

```text
data/
  HumanMM_AIST_Bridge3R_v1_20260826/           # 原始上传包，只读，不改 manifest
  bridge3r_singleperson_v1/
    videos/aist/{cs150,mc150_3,mc150_4}/       # 派生 MP4
    labels/aist/{cs150,mc150_3,mc150_4}/       # per-frame GT / cut labels
    frame_maps/aist/                            # PTS--GT--output 映射
    calibration/aist/                           # 官方 K/R/t 的规范化副本
    audits/aist/                                # projection overlays、数值审计
    DATASET_CARD_v1.md
    CHECKSUMS.sha256
  manifests/bridge3r_singleperson_v1/          # 只追加 derived manifest；不重写输入 manifest
Movie3R/experiments/bridge3r_singleperson_aist_v1/
  manifests/ adapters/ run_cards/ logs/ predictions/ metrics/ figures/
ICLR-paper/bridge3r_iclr2027/
  generated/aist_singleperson_v1/              # 自动生成的表、图、claim ledger
```

每次实际运行前须保存：输入 manifest hash、Movie3R commit、baseline commit、环境 lock、checkpoint URL/sha256、命令、stdout/stderr、GPU/时长、逐 case 输出和聚合代码版本。

## 4. 阶段 A：构造可评测 AIST++ 派生集（尚未执行）

### A0. 保护与静态复核

1. 将上传目录挂载或约定为只读；不修改原始 MP4、annotation 或四个已冻结 manifest。
2. 在构造运行卡中写入上传包 hash、包内 checksum 状态、AIST 官方版本和许可范围；不把原始 RGB 或 ZIP 纳入 Git/Overleaf 包。
3. 从 `source_index_v1.json` 读取 Test/Pilot、四相机 tuple、`source_start_gt_index_60fps` 和 150 个 output 时间点；只允许生成新增的 derived manifests。

**通过条件：** 输入清单 hash 与当前冻结 hash 完全一致。任何 mismatch 都停止，而不是重新选择 source。

### A1. 逐帧时间映射

对每个 `{source, camera}`：

1. 从 MP4 解码获取每帧 PTS / time base，并以官方时间轴为准映射到 60 FPS GT tick。
2. 对 150 个输出时刻使用冻结规则 `gt_tick = source_start_gt_index_60fps + 2 * output_frame`；保存所选 RGB PTS、解码帧索引、GT tick、时间残差和是否精确匹配。
3. 逐 view 检查单调性、无重复/漏帧、末帧合法、四相机对应同一动作时刻；不得把容器帧号直接当 GT index。
4. 形成每 source/view 一个 JSON/Parquet frame map，并生成 machine-readable 汇总（最大/中位时间误差、失败数）。

**通过条件：** 150 个输出时刻全部有唯一映射，且任何不可解释的 PTS 偏差均有明确失败记录。未通过 source 从当前 derived manifest 排除并记录，不能由 reserve 补位，除非先发布新的、结果不可见时的 manifest 版本并由负责人确认。

### A2. 视频、GT 与 camera 标签派生

对每条 source 基于同一连续 150 个 output 时间点构建三种 protocol：

1. `CS150`: A 视角帧 0--74，B 视角帧 75--149。
2. `MC150-3`: A/B/C 视角帧 0--49/50--99/100--149。
3. `MC150-4`: A/B/C/D 视角帧 0--37/38--75/76--112/113--149。
3. 派生视频保持 30 FPS、150 帧和原始空间分辨率；使用固定、可复现的编码命令。不同方法不得各自抽帧或重建不同版本。
4. 每条 label 至少包含：`case_id`、source、protocol、output frame、GT tick、RGB source/path/PTS、shot id、evaluation-only cut index、SMPL motion、3D keypoints、root translation、root orientation、scale、camera `K,R,t`、关节/坐标系/单位声明。
5. 写入 `derived_manifest_v1.json`、每个派生 MP4/label 的 SHA256、帧数、FPS、分辨率和 source mapping。

**不可使用的捷径：** 不能用相邻视频帧替代 PTS 映射，不能用模型、SLAM、COLMAP 或人体拟合伪造 camera GT，不能把 GT cut 写进 baseline 输入。

### A3. 几何与标签审计

1. 从 Test/Pilot 中预先按 hash 选至少 20 个不同 source，覆盖 9 个 camera setting 和 2/3/4-shot protocol。
2. 用官方 camera 标定把官方 3D joints（并在 SMPL convention 通过后可加入 mesh）投影到 RGB；输出 overlay、每帧重投影误差、左右/坐标轴检查和 cut 两侧相机切换检查。
3. 检查 motion / keypoints 的长度、单位、root 定义、scale、joint order；定义所有方法共享的最小关节集。
4. 做下列 evaluator sanity tests：identity prediction 近零；固定 world shift 只影响 global/root 指标；逐 shot 重置 gauge 应被 Anchor 指标惩罚；PA 指标对合法相似变换不变；cut label 无法经模型输入读取。

**通过条件：** 20-case 审计与自动阈值均通过，或把失败 source 明确排除、冻结实际 `N` 后再继续。没有通过 projection audit 时，本数据不得报 camera accuracy。

### A4. Dataset card 与冻结

在通过 A1--A3 后补写 `DATASET_CARD_v1.md`：官方来源与许可、包含/排除内容、split、case selection、camera tuple、frame map、编码、标签语义、指标公式、已知局限、checksum 和禁止用途。只有卡片、checksums 和 derived manifest 齐备后才进入 pilot。

## 5. 阶段 B：统一 evaluator 与指标（尚未执行）

所有逐帧误差先在单个 case 聚合，再对 source 做 macro average；AIST++ 不与其他数据集或不同 protocol 混合平均。对 Bridge3R 相比 causal parent 使用 source-level paired bootstrap 95% CI，并报告完成率。

| 组别 | 指标 | 直观含义 | 进入位置 |
|---|---|---|---|
| 局部人体 | PA-MPJPE | 去掉位置/朝向/尺度后，姿势骨架是否准确 | 主表 |
| 全局人体 | First-shot Anchor MPJPE | 仅第一 shot 允许一次 gauge 对齐，之后不再对齐；检验跨 cut 全局连续性 | 主表 |
| root 平移 | Anchor-RTE (m) | 同一 anchor 后人体根节点的位置误差 | 主表 |
| root 朝向 | Anchor-ROE (deg) | 同一 anchor 后人体朝向误差 | 主表 |
| 边界 | Seam-root / Seam-orientation | cut 相邻帧的预测变化是否与真实连续运动一致 | 主表或补充 |
| camera | per-shot camera rotation / relative transform | 相机切换的几何是否对应官方标定 | 仅 A3 通过后报告 |
| detector | precision / recall / F1 / median boundary offset | 自动 detector 是否在真实 cut 附近触发 | 补充 |
| 可用性 | completion / valid-frame coverage | 整个冻结分母中自动完成和有效输出的比例 | 每张方法表 |
| 稳定性 | jitter / foot sliding | 抖动和脚底滑动；仅在 SMPL 足部/contact 定义核验后启用 | 补充（条件性） |

`First-shot Anchor` 不是未经核验地借用 HumanMM 的 WA-MPJPE 名称。只有当公式、joint set、尺度、对齐域和聚合方式逐项等价，才可在括号中说明对应关系；否则保持透明的本工作名称与完整公式。

## 6. 阶段 C：方法可运行性审计与运行队列（尚未执行）

### C1. 所有方法共用的准入检查

每个方法在 12 个 `pose_val` pilot 上依次完成以下检查，才有资格运行 100-source Test：

1. 固定代码 commit、许可证、环境 lock、官方 checkpoint 和 SHA256；不下载/替换任何未记录的权重。
2. 读取同一派生 MP4，确认不读取 GT labels、camera id 或 `cut_index`。
3. 验证输出的 frame count、时间顺序、track identity、单位、坐标系、SMPL/joint convention 与 global root 语义。
4. 固定自动 track rule：若官方 pipeline 产生多 track，使用开发前定义的“最长且有效覆盖最高；平局按 track id”规则；不手选最佳人体。
5. 记录输出覆盖、失败原因、wall time、GPU peak memory、adapter 版本和每帧有效性。
6. 只做接口和 evaluator sanity check；pilot 不用于选择 Bridge3R 超参数或外部基线的有利 case。

pilot 通过后，写入不可修改的 `adapter_freeze_<method>.json`；任何 adapter、checkpoint、前处理或 threshold 变化都必须新建版本，不能覆盖结果。

### C2. 必须完成的 Bridge3R 内部行

| 方法行 | 目的 | CS150 | MC150-3/4 |
|---|---|---:|---:|
| Strict Human3R | 不处理切镜的 streaming 父骨架 | 必须 | 必须 |
| Clean reset | 只清历史状态，量化“重置”本身的作用 | 必须 | 必须 |
| Causal parent | Bridge transaction 前的直接因果父方法 | 必须 | 必须 |
| Bridge3R-Fixed | 论文固定主 operating point | 必须 | 必须 |
| w/o coarse alignment | 验证粗相机--人体对齐 | 必须 | CS150 必须；多 cut 可补充 |
| w/o explicit correction-token branch | 验证修改 Human3R 的显式纠正 token 分支 | 必须 | CS150 必须；多 cut 可补充 |
| w/o fine alignment/history transaction | 验证精对齐与历史状态交易 | 必须 | CS150 必须；多 cut 可补充 |

所有内部行均使用同一自动 cut detector；GT cut 不会输入。若将来另做 oracle reset，它只能出现在**消融/诊断**表，不能作为外部 baseline 行或主要 Bridge3R 成绩。

### C3. 外部方法队列与当前本地可用性

| 优先级 | 方法 | 本地代码（冻结 commit） | 当前 checkpoint 状态 | 预期比较角色 | 进入全局表的额外条件 |
|---|---|---|---|---|---|
| P0 | GVHMR | `external_baselines/GVHMR` (`6ec3ca3`) | 本地发现 6 个权重文件 | 公开、直接的单人 world-HMR 基线 | root/world gauge 与单位可验证 |
| P0 | WHAM | `external_baselines/WHAM` (`2b54f77`) | 当前未发现 checkpoint | 公开、直接的单人 world-HMR 基线 | 官方权重、SMPL 许可、切镜视频能自动运行 |
| P1 | SLAHMR | `external_baselines/SLAHMR` (`58518fe`) | 当前未发现 checkpoint | 同输入的离线 camera--human reference | PHALP/OpenPose/DROID-SLAM 全自动、输出语义可验证 |
| P1 | MultiShot | `external_baselines/Multishot` (`745287e`) | 当前未发现 checkpoint | 最直接的多镜头单人离线 reference | 不使用 GT shot/tracklet、全自动前端通过 pilot |
| P2 | PromptHMR | `external_baselines/PromptHMR` (`bd3a7c4`) | 本地发现 6 个 checkpoint | 局部姿态/availability 补充 | 有可验证的连续 global root 才填全局列 |
| P2 | TRACE | `external_baselines/ROMP` (`a8558ae`) | 本地已有 checkpoint | 多人方法在单人输入上的补充 | 同上；否则只报 PA/coverage |
| P2 | TRAM | `external_baselines/TRAM` (`4861c11`) | 当前未发现 checkpoint | 可选 camera-aware reference | 官方权重与可验证输出 |
| P2 | Video-OnlineHMR | `external_baselines/Video-OnlineHMR` (`5236d22`) | 当前未发现 checkpoint | 可选在线局部 reference | 不作为 world-gauge 主表 |

不在此处强行加入普通 image-HMR、SPEC 或只输出 camera-coordinate pose 的单帧方法：它们可以作为将来的单镜头局部姿态附表，但不能检验跨 cut 的 global gauge。HumanMM 不进入运行队列，因其完整执行/评测链未公开。

### C4. 不同输出能力的公平呈现

1. **Direct global table：** Bridge3R 内部行 + 通过 C1 的 GVHMR/WHAM；符合 C3 条件的其他方法可加入。列为 PA-MPJPE、Anchor-MPJPE、Anchor-RTE、Anchor-ROE、Seam、completion。
2. **Offline reference table：** SLAHMR/MultiShot 若完整 pilot 成功，单列标注 `offline full-sequence optimization`；不把未来帧优化与 Bridge3R 的 causal runtime 混称。
3. **Local/availability table：** PromptHMR、TRACE、TRAM、Video-OnlineHMR 等无法证明 global semantics 的方法，仅列 PA-MPJPE、coverage、runtime/失败原因与定性结果。
4. **Literature-only table：** HumanMM 的原论文数字单独引用，表头显著标注 `source-reported; different processing/evaluator; not ranked`，不加粗、不排名。

## 7. 阶段 D：实际运行顺序（未来执行，不在本轮开始）

| 顺序 | 工作 | 成功标准 | 失败时的处理 |
|---:|---|---|---|
| D0 | 完成 A0--A4 | derived dataset card、projection audit、frozen v1 manifest | 停止，不跑模型 |
| D1 | 内部行的 12-case pilot | detector/adapter/evaluator 全通过 | 修接口；不修改 Test manifest |
| D2 | GVHMR 与 WHAM pilot | 可验证的连续输出与 coverage | 记录 availability failure，不阻塞内部表 |
| D3 | SLAHMR/MultiShot pilot | 全自动、不用 GT、输出可适配 | 只保留失败运行卡或降为定性参考 |
| D4 | P2 pilot | 明确哪些指标可合法报告 | 限制到 local/coverage table |
| D5 | 冻结 adapter/checkpoint/command | 所有 Test 配置可复跑 | 任何变化生成 v2，不覆盖 v1 |
| D6 | CS150：内部全量 100 source | 全部方法使用相同 manifest | 失败保留分母 |
| D7 | CS150：P0 全量 | 统一 evaluator | 与 D6 同规则 |
| D8 | MC150-3/4：内部全量 | 固定时长的 cut-count 证据 | 放补充材料 |
| D9 | P1/P2 全量（仅适格者） | 通过 pilot 的条件 | 不为填表硬跑不可靠系统 |
| D10 | 汇总、统计和论文生成 | result ledger 可追溯 | 不手工抄写数字 |

GPU 并行只在 D5 后使用，并按“一个方法 x 一个冻结 shard”的方式并行；同一 case 绝不由多个进程争写输出。运行前先估算磁盘：原始包约 6.5 GB，派生视频、逐 case prediction 和外部优化缓存必须设独立额度与清理策略；权重、原始视频和用户已有输出绝不被自动删除。

## 8. 阶段 E：统计、论文呈现与可发表标准

### E1. 正文

1. **单次切镜主表：** AIST++ `CS150`，Bridge3R 主行、必要内部消融与通过可验证性审计的 P0 基线。
2. **核心消融：** coarse alignment、explicit correction-token branch、fine alignment/history transaction、clean reset/causal parent。oracle 仅诊断，不以此主张真实运行优势。
3. **定性图：** 同一 source 的 RGB、GT/预测 overlay、root trajectory、自动 detector 触发点与切镜标记，展示运动连续但相机改变的情形。
4. 表注明确：100 `pose_test` source、150 frame / 5 s、相同 manifest、first-shot-only anchor、case-level macro average、自动 detector、无 future frames。

### E2. 补充材料

1. `MC150-3` 与 `MC150-4` 的 cut-count 压力表、error-vs-cut-count 图和完整内部消融。
2. detector 精度、coverage、失败日志、runtime、GPU memory、per-source paired statistics 和 95% CI。
3. 20-case calibration projection audit、frame map specification、joint conversion、全部指标公式和 dataset card 摘要。
4. 外部 baseline availability / offline table；HumanMM source-reported 文献参考表及不可直接比较的原因。

### E3. 可写入论文的最低证据门槛

只有同时满足以下各项，AIST++ 数值才可被标为“正式结果”：

- derived manifest、frame map、checksum、dataset card、calibration audit 均可复现；
- 每个直接比较方法用相同 Test cases；
- 输出语义和单位均经过 pilot 验证；
- 每种方法的 completion/coverage 与失败原因公开；
- 结果由逐 case artifacts 自动聚合，且 paired CI/分母与表格一致；
- claim ledger 中的每一句“更优”只针对相同输入、相同 evaluator、相同可验证输出语义的方法集合。

## 9. 本阶段交付物与下一步

本轮已交付：冻结 AIST++ 输入审计结论和本计划。未产生任何模型预测或论文数值。

下一步应从 **A0--A1（frame map）** 开始，而不是直接运行 Bridge3R、GVHMR 或其他 baseline。完成 A4 后再执行 C1 的 pilot；正式 100-source 结果在 D6 之前一律保持为空。Human3.6M 仍需要用户在官方许可范围内单独提供数据，故不与本 AIST++ v1 的启动条件混淆。
