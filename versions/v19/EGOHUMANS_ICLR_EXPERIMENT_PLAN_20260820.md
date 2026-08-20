# Movie3R-v19 × EgoHumans：ICLR 实验预注册与执行计划

日期：2026-08-20  
状态：数据结构已完成首个 capture 审计；在任何 Movie3R 指标产生前冻结本计划  
目标：在磁盘受限条件下完成可复现、可恢复、GT 隔离的 EgoHumans 多人跨镜头实验，并产出可直接进入 ICLR 论文的主表、消融、分层统计和限制说明。

## 1. 完成标准

本任务只有同时满足以下条件才算完成：

1. 原始 `/data/wangzheng/iJCV-CODE/data/EgoHuman.zip` 保持不变并通过 ZIP 完整性检查；
2. 每次只展开一个 capture，完成审计、推理、评测和结果校验后安全删除展开副本；
3. development、holdout、test 在查看任何 Movie3R 指标前固定，test 不参与阈值选择；
4. 至少报告 Strict Human3R、Movie3R-v15、冻结 v17 parent、冻结 v17 MultiCue-Safe 和最终 v19；
5. 统一计算 W-MPJPE、WA-MPJPE、MPJPE、MPVPE、Accel、ATE、IDs，并额外计算相机旋转、RPE、IDF1、Coverage、边界 seam、camera-human relative gauge、pairwise layout、Jitter、Foot Sliding 和运行成本；
6. 最终候选在独立 holdout 上通过安全门槛后才冻结，并以冻结配置运行整个 test；
7. 产出逐 case CSV、JSON、LaTeX 表、置信区间、分动作/镜头跨度统计、消融和最终中文报告；
8. 任何失败、统一排除和 fallback 都保留记录，不按结果删除难例。

## 2. 数据事实与磁盘策略

### 2.1 外层归档

- 文件：`/data/wangzheng/iJCV-CODE/data/EgoHuman.zip`
- 大小：237,157,095,656 bytes，约 221 GiB；
- 内容：7 类动作、44 个 `tar.gz` 条目；
- 动作：badminton、basketball、fencing、legoassemble、tagging、tennis、volleyball；
- 已发现 `badminton/003_badminton-003.tar.gz` 与 `badminton/003_badminton-004.tar.gz` 的未压缩长度和 CRC-32 完全相同，按数据级重复只保留字典序第一份；
- 去重后预计 43 个唯一 capture 归档。

### 2.2 已审计样例

`basketball/004_basketball.tar.gz` 展开后约 1.5 GiB，包含：

- 8 路固定 exo 相机，3840×2160；
- 4 路 Aria，1408×1408 RGB/left/right 和逐帧标定；
- 世界坐标 `poses3d`、最终 SMPL 6890 vertices/45 joints；
- 每相机 bbox、133 点 pose2d、稳定 `human_name/human_id`；
- COLMAP intrinsics/extrinsics，以及 Aria metric world 与 COLMAP 的 Sim(3)；
- 111 个同步 RGB/SMPL 帧，编号 00001–00111；
- 样例最终 GT 身份为 aria02、aria03、aria04。

### 2.3 流式 staging

由于 `/data` 当前可用空间约 338 GiB，禁止整体解压。固定流程为：

```text
外层 ZIP
  -> 流式提取一个内层 tar.gz 到 /data/.../EgoHuman_work_v19
  -> 校验 CRC/成员安全性
  -> 只展开当前 capture
  -> audit / manifest / GPU inference / GT-only evaluation
  -> 验证输出和指标完整
  -> 删除当前展开 capture 与内层 tar.gz
  -> 处理下一个 capture
```

所有临时文件写入 `/data/wangzheng/iJCV-CODE/data/EgoHuman_work_v19`，禁止写 `/tmp` 或系统根目录。后台总任务使用 `tmux`，所有阶段原子写 state/ledger，可断线恢复。

## 3. 与论文协议的关系

### 3.1 Multi-THuMBS 公开信息

Multi-THuMBS 使用 EgoHumans、EgoBody、Harmony4D 的同步多视角序列人工构造 shot boundary，报告：

- 人体：W-MPJPE、WA-MPJPE、MPJPE、MPVPE；
- 时序：Accel；
- 相机：ATE；
- 身份：IDs。

其 EgoHumans 文献数值为：

| Method | W ↓ | WA ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ | ATE ↓ | IDs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-THuMBS | 279.0 | 166.0 | 228.3 | 262.2 | 27.3 | 0.7 | 0.97 |

主文中的“150 帧、1920×1080、RTX 3090 约 10 分钟”是实现时间示例。公开 PDF 和 arXiv source 均未提供 EgoHumans 的具体 capture、camera pair、cut frame、可见性规则、完整 evaluator 或 supplement。因此上述数值只能作为 `literature reference / target scale`，不能声称官方同协议复现。

### 3.2 EgoHumans 原始数据依据

EgoHumans 原论文说明序列被切成平均约 30 秒、20 FPS 的短片。官方仓库说明 test tracking 使用 tagging、legoassemble、fencing，但 Multi-THuMBS 并未公开是否沿用这一原始 tracking split。此次上传归档包含七类动作，故采用独立、预注册的 `Movie3R-EgoHumans-CS100-v1`，同时在附录单列原 EgoHumans tracking-test 三动作结果。

## 4. 固定评测协议：EgoHumans-CS100-v1

### 4.1 长度与时间

- 每例固定 100 帧：50 pre + 50 post；
- FPS 固定为 20；
- 从 RGB、SMPL 和标定都可用的最长连续同步区间中取居中的 100 帧；
- pre 使用一个固定 exo camera，post 使用另一个固定 exo camera；
- 边界发生真实的一帧时间推进，不复制同一时间戳；
- 若 capture 不存在连续 100 帧，按 `structurally unavailable` 统一排除，不缩短该 case；
- 150 帧只在具备至少 150 连续帧的 capture 上作为附录长度消融，不能混入 CS100 主表。

采用 100 帧的理由是数据支持而非结果选择：首个未运行 Movie3R 的审计样例只有 111 帧，而 100 帧仍覆盖 5 秒在线轨迹，足以评价边界与 shot 内稳定性。

### 4.2 相机跨度

每个 capture 由 GT 标定、在推理前选择四个互不重复的 camera pair：

- small：全部相机对旋转跨度的低分位；
- medium：中低分位；
- large：中高分位；
- extreme：最大或接近最大跨度。

选择只读取相机标定和 GT 可见性，不读取任何模型输出或误差。方向按相机名和 capture 哈希确定，不能在结果出来后反转。

### 4.3 数据划分

为保证七类动作均有 development/holdout，同时降低首次 staging 风险，在任何模型指标产生前按每类动作的内层归档大小排序：

- development：每类动作最小的一个唯一 capture，共 7 个；
- holdout：每类动作第二小的一个唯一 capture，共 7 个；
- test：其余全部唯一 capture，预计 29 个；
- 重复 CRC 条目在划分前去重。

归档大小是结果无关的数据属性。development 可以调预测侧参数；holdout 只允许候选晋级/否决；test 只能在候选冻结后读取。

## 5. GT 隔离与坐标系

### 5.1 推理可见内容

Movie3R runtime 只可读取：

- 所选 exo RGB；
- 预注册的 known boundary，或独立评测 causal detector；
- Human3R/V9 自身预测、历史状态和 prediction-only diagnostics。

禁止 runtime 读取：SMPL、poses3d、bbox、pose2d、camera calibration、identity label、最终误差或动作标签。

### 5.2 evaluator 坐标

- EgoHumans 最终 SMPL 位于 Aria metric world；
- `colmap_from_aria_transforms.pkl["aria01"]` 提供 Aria→COLMAP Sim(3)；
- COLMAP exo 外参先求稳定 camera-to-COLMAP，再转换到 Aria metric world；
- 预测和 GT 统一在 Aria metric world 评测；
- GT 相机/身份只进入 evaluator。

### 5.3 可见性与匹配

- GT 身份来自 SMPL dict 的稳定 `ariaXX` key；
- 可见性由对应相机的 bbox/pose2d 有效性与 fisheye-aware 投影共同审计；
- 预测与 GT 使用 camera-coordinate root + root-centered body cost 的逐帧 Hungarian；
- 允许漏检和 false positive，不用 GT identity 修复 runtime track；
- identity 指标同时报告 IDs、IDF1、coverage、precision/recall。

## 6. 方法与消融

同一 RGB、同一 checkpoint policy、同一缓存比较：

1. Strict original Human3R；
2. clean reset；
3. no-V9 raw SE(3)；
4. B0 only；
5. B0 + identity；
6. B0 + identity + BRTC；
7. B0 + identity + BRTC + C1；
8. Movie3R-v15 oracle-boundary；
9. Movie3R-v15 causal detector；
10. v15 safe boundary permutation；
11. v17 parent；
12. frozen v17 MultiCue-Safe；
13. 最终 v19 EgoHumans 候选。

development 只探索小型、可解释、prediction-only 网格，优先围绕：

- boundary shared translation 的 blend/cap；
- 最少匹配人数与匹配比例；
- boundary residual trust gate；
- causal root stabilization 的 alpha/beta；
- 单人/多人可观测性 fallback；
- camera-human shared transform 是否提交。

不额外引入 VGGT、DA3、DROID-SLAM、Re-ID 网络或未来帧优化。

## 7. 指标定义与聚合

### 7.1 Multi-THuMBS 同名指标

- W-MPJPE：每条身份轨迹用开头最多两帧拟合一个 Sim(3)，之后固定对齐；
- WA-MPJPE：整条身份轨迹拟合一个 Sim(3)，只反映轨迹级对齐后的形状/运动；
- MPJPE：camera-local、pelvis-relative joints；
- MPVPE：camera-local、pelvis-relative SMPL vertices；
- Accel：预测与 GT 二阶差分误差，mm/frame²；
- ATE-Sim3 与 ATE-SE3：整段相机中心轨迹对齐后的误差，分别允许/不允许尺度；
- IDs：同一 GT 身份的 persistent ID 随时间改变次数。

由于 Multi-THuMBS 未公开 evaluator，上述命名在表中标记 `public-description reproduction`，并同时公开公式和代码。

### 7.2 Movie3R 特有指标

- camera translation/rotation、RPE；
- first-post camera/root/joint/vertex error；
- cut seam camera/root/joint/vertex excess；
- camera-human relative root gauge error（CHRGE）；
- pairwise human distance/vector layout；
- IDF1、ID continuity、Coverage、precision/recall；
- Jitter、Foot Sliding；
- static drift、moving motion retention；
- gate accept/fallback/harm rate；
- FPS、boundary overhead、peak VRAM、CPU geometry latency。

### 7.3 聚合

- 正文主值：case macro；
- 同时报告 action macro；
- 95% CI：action→capture→camera-pair 分层 bootstrap，10,000 次；
- 显著性：最高独立单元为 action，不把同一 capture 的四个相机对当独立样本；
- 所有 exact fallback 按方法契约记零差异；
- 每个 accepted case 报改善/恶化和最坏 harm。

## 8. 候选晋级门槛

### 8.1 development 候选进入 holdout

相对冻结 v17 parent：

- W-MPJPE、WA-MPJPE、Accel、ATE-SE3、Seam-root 五项至少三项改善；
- 综合几何均值相对改善至少 3%；
- MPJPE 和 MPVPE 各不得恶化超过 2%；
- Coverage 不得下降超过 1 个百分点；
- IDF1 不得下降超过 0.01；
- 任一 accepted case 的 W-MPJPE 灾难性恶化不得超过 20%；
- 失败时必须 exact fallback，不得丢 case。

### 8.2 holdout 冻结 v19

- 上述安全约束全部满足；
- W/WA/Accel/ATE-SE3/Seam-root 至少三项方向与 development 一致；
- 至少 5/7 动作不出现综合恶化；
- 相对 v17 parent 的核心综合指标改善至少 2%；
- 如果没有候选通过，则正式冻结 v17 MultiCue-Safe 作为 EgoHumans 方法，不按 holdout 重新调阈值。

### 8.3 test 论文结论

最终 test 需要回答：

1. 相对 Strict Human3R，Movie3R 是否在相机、身份、边界、时序中至少三类取得优势；
2. 相对 v15/v17，v19 是否在不破坏局部人体质量和 coverage 的情况下改善全局跨 shot 指标；
3. small/medium/large/extreme 是否都有稳定收益或安全 fallback；
4. low-visibility、人数变化、不同动作是否揭示明确 failure mode；
5. 与 Multi-THuMBS 文献量级相比，哪些同名指标达到或优于目标，哪些仍有差距。

## 9. 执行阶段

1. 完成 outer index、去重、帧数/相机/GT audit；
2. 写入并测试 stager、dataset loader、manifest builder、evaluator；
3. 在 `004_basketball` 跑 1 个 smoke case，核对坐标、GT 投影和指标有限性；
4. development 7 capture × 4 angle，运行冻结参考和有限候选；
5. 选择最多三个候选进入 holdout；
6. holdout 7 capture × 4 angle，冻结一个候选或回退 v17；
7. test 其余全部 capture × 4 angle；
8. 生成主表、附录、case CSV、LaTeX、统计和最终报告；
9. 校验 Git 状态、运行日志、磁盘回收与复现命令。

## 10. 允许的统一排除

只允许以下方法无关排除：

- ZIP/tar CRC 损坏；
- 路径穿越或归档安全失败；
- RGB/SMPL/标定没有 100 个连续同步帧；
- 相机外参无法形成合法旋转/变换；
- GT SMPL 非 6890 vertices 或身份集合自相矛盾；
- 所有方法都无法形成 evaluator 的初始匹配。

普通难例、低 coverage、错 ID、极端误差、gate reject 和某一方法推理失败都不能因结果不好而删除，必须记录并继续或修复。

## 11. 论文呈现

正文放：

- EgoHumans-CS100 主表；
- Strict Human3R / v15 / v17 / v19；
- W、WA、MPJPE、MPVPE、Accel、ATE-SE3、IDF1、IDs；
- 四个 angle strata；
- 关键模块消融；
- 在线运行成本；
- 与 Multi-THuMBS 的任务/资源/因果性对比。

附录放：

- ATE-Sim3、RPE、CHRGE、pair layout、Jitter、FS、Coverage；
- 全部逐 action/capture/case；
- 150 帧可用子集长度消融；
- detector known-boundary 与 automatic 两张表；
- 公开协议不完整说明；
- 统一排除和失败日志；
- gate harm/fallback 明细与可视化索引。

## 12. 结构排除后的 holdout gate 补充（2026-08-20）

冻结 holdout 中的 `fencing/005_fencing-001.tar.gz` 在方法推理前的数据审计阶段发现最长同步 RGB/SMPL 段为 0 帧，满足第 10 节的统一结构排除。该动作没有产生任何候选或 baseline 指标，也不以其他 capture 替换。

原实现把“至少 5/7 动作不恶化”误写成同时要求 `action_count == 7`，会使任一完整动作发生方法无关结构排除时所有候选必然失败。修正后保留预注册的绝对门槛：至少 5 个结构可评动作综合不恶化，并要求至少存在 5 个结构可评动作；候选自身报错、指标缺失或表现差不能缩小分母。该修正不改变候选、参数、指标或其他晋级门槛，并在 holdout 聚合/最终冻结前提交。
