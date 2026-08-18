# Movie3R-v15 × Harmony4D：ICLR 专项实验计划

日期：2026-08-18
状态：**规划已冻结，尚未解压、尚未运行 GPU、尚未进行数据集适配**
目标版本：Movie3R-v15-final；若产生方法改动，必须另建新版本，不能静默覆盖 v15
数据归档：

> /data/wangzheng/iJCV-CODE/data/Harmony4D.zip

论文依据：

> /data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf

当前 Git 基线：

> a4e6850 docs(v15): add ICLR method figure design brief

---

## 0. 本轮边界

用户当前要求先写规划，不开始执行。因此本轮明确不做：

- 不解压 Harmony4D.zip；
- 不提取任何嵌套序列 ZIP；
- 不启动 GPU 推理；
- 不生成 cut；
- 不计算数据集指标；
- 不为 Harmony4D 修改阈值、checkpoint 或方法；
- 不删除外层 Harmony4D.zip。

后续收到“开始执行”指令后，严格从本计划 Phase 0 开始。

---

## 1. 结论先行

Harmony4D 是 Movie3R 最重要的多人跨镜头测试之一，原因不是它背景一定低纹理，而是它包含：

- close human interaction；
- severe occlusion；
- body truncation；
- 多人身份易混淆；
- 快速真实运动；
- 二人/多人相互接触；
- 多个同步相机视角；
- camera cut 后人物外观、朝向和可见性突变。

它可以同时检验 Movie3R 的四个核心主张：

1. B0 是否把 post shot 带入可匹配的粗 world gauge；
2. B0 后的 permutation-aware association 是否保持跨 shot persistent identity；
3. BRTC-LC 是否在不破坏可信相机的情况下改善人体 root/depth/layout；
4. adaptive joint 是否只在几何可观测时联合修正 camera-human，并在遮挡或歧义时安全 abstain。

最终实验必须分成两个清楚的证据层：

### 层 A：可直接用于论文主结果

使用我们冻结并公开的：

> **Movie3R-Harmony4D-CrossShot-v1**

所有可运行方法在完全相同的 manifest、RGB、checkpoint policy、GT evaluator 和聚合方式下比较。该层可以直接判断：

- Full Movie3R 是否优于 strict Human3R；
- 是否优于 clean reset、no-V9、B0-only；
- ID、BRTC、C1、adaptive gate 分别贡献什么；
- 是否在线、是否安全、是否存在 false acceptance。

### 层 B：Multi-THuMBS 文献参考

Multi-THuMBS 论文公开了 Harmony4D 表格数值，但没有公开：

- sequence 列表；
- camera pair；
- cut frame；
- clip 数；
- visibility/miss/FP 规则；
- W/WA/ATE/IDs 的完整公式；
- supplementary；
- evaluator。

因此论文数值只能写为：

> literature reference / protocol-matched-as-far-as-public-information

除非作者之后公开官方协议，否则不能写：

> official reproduction / directly outperform Multi-THuMBS

这不是回避比较，而是保证 ICLR 论文实验不会因协议不严谨被审稿人直接质疑。

---

## 2. 当前数据与磁盘事实

### 2.1 外层归档

当前文件：

~~~text
/data/wangzheng/iJCV-CODE/data/Harmony4D.zip
size = 351,667,301,150 bytes
约 327.51 GiB
~~~

外层 ZIP 只有 27 个条目，核心内容是嵌套的 train/test 序列 ZIP，而不是直接展开的图片树。

当前 /data 可用空间：

~~~text
约 369 GiB
约 395.4 GB decimal
~~~

不能直接全量解压外层 ZIP，再同时展开所有内部序列。必须逐序列处理。

### 2.2 官方 test 嵌套包

| 顺序建议 | Test 序列 | 内层 ZIP 字节数 | 约 GiB |
|---:|---|---:|---:|
| 1 | 01_hugging | 1,695,682,656 | 1.58 |
| 2 | 15_mma4 | 2,344,076,740 | 2.18 |
| 3 | 05_sword2 | 8,835,201,165 | 8.23 |
| 4 | 08_ballroom2 | 10,133,815,342 | 9.44 |
| 5 | 16_mma5 | 11,068,053,080 | 10.31 |
| 6 | 03_grappling2 | 14,622,444,780 | 13.62 |
| 7 | 06_sword3 | 16,263,499,748 | 15.15 |

Test 嵌套 ZIP 合计：

~~~text
64,962,773,511 bytes
约 60.50 GiB
~~~

### 2.3 Train 嵌套包

官方 train 部分包含 15 个 ZIP，合计：

~~~text
286,662,680,609 bytes
约 266.98 GiB
~~~

训练数据不全量展开。只在以下情况按需提取少量 train 序列：

- 数据 schema 和坐标系适配；
- evaluator 开发；
- GPU smoke test；
- Harmony-specific calibration；
- 不接触 test 的方法选择。

### 2.4 Harmony4D 官方规模

Harmony4D 官方项目页公开：

- 1.66M images；
- 3.32M human instances；
- 超过 20 个同步相机；
- 208 video sequences；
- 24 unique subjects；
- human detection/tracking；
- 2D/3D pose；
- mesh recovery annotations；
- close interaction、occlusion、contact 与 in-the-wild 场景。

正式论文中应引用 Harmony4D 原论文，而不是只引用下载包。

---

## 3. Multi-THuMBS 如何使用 Harmony4D

根据 Multi-THuMBS 主文公开信息：

1. 从 Harmony4D 等 multi-view sequences 构造 multi-shot benchmark；
2. 在指定 frame 人为引入 shot boundary；
3. 使用 boundary 前后两帧形成 shared 3D space；
4. 对 boundary humans、root、orientation 和 camera 进行多阶段 alignment；
5. 将边界对齐传播到 shot 内其他帧；
6. 用 geometry、appearance 和 pose 进行 Re-ID；
7. 对全序列进行 temporal smoothing 与 cross-camera consistency optimization；
8. 在 Harmony4D 上报告人体、相机和身份指标。

其系统需要 VGGT、Grounded-SAM、ViTPose、4DHumans、DROID-SLAM 等多个模块，并进行离线优化。论文公开运行参考是：

~~~text
150 frames
1920×1080
single RTX 3090
approximately 10 minutes
~~~

Movie3R 的对比重点不是复制这套离线系统，而是证明：

~~~text
相同任务目标：
camera + human + identity consistency across cuts

更严格的执行约束：
first-frame causal
no future smoothing
no history rewrite
one same-model shadow proposal
explicit geometry verification
safe abstention
~~~

---

## 4. Multi-THuMBS 的 Harmony4D 文献参考线

### 4.1 人体指标

论文 Table 1：

| Method | W-MPJPE ↓ | WA-MPJPE ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ |
|---|---:|---:|---:|---:|---:|
| Multishot | 248.0 | 231.6 | 511.2 | 609.5 | 37.1 |
| GVHMR | 244.9 | 166.7 | 244.7 | 334.1 | 29.6 |
| PromptHMR | 1746.3 | 399.8 | 675.7 | 746.0 | 66.8 |
| HSfM† | 372.0 | 178.4 | 225.6 | **257.6** | 28.3 |
| Multi-THuMBS | **221.0** | **116.9** | **215.9** | 278.3 | **17.4** |

注意：

- Harmony4D 的 MPVPE 最佳不是 Multi-THuMBS，而是 HSfM† 的 257.6；
- 如果将来获得完全一致的官方协议，MPVPE 的 SOTA 目标应为小于 257.6，而不是只小于 278.3。

### 4.2 身份指标

论文 Table 2，Harmony4D IDs：

| Method | IDs ↓ |
|---|---:|
| PromptHMR | 8.00 |
| HSfM† | 1.58 |
| KPR | 1.19 |
| Pose2ID | 1.32 |
| Multi-THuMBS distance-only | 0.54 |
| Multi-THuMBS full | **0.46** |

### 4.3 相机指标

论文 Table 2，Harmony4D ATE：

| Method | ATE ↓ |
|---|---:|
| VGGT | 1.4 |
| PromptHMR | 2.3 |
| HSfM† | 3.2 |
| Multi-THuMBS | **0.7** |

### 4.4 使用限制

上述值只放在独立的 literature reference 表中，除非满足：

- 完全相同 sequence；
- 完全相同 camera pair；
- 完全相同 cut；
- 完全相同 clip；
- 完全相同 visibility/miss/FP；
- 完全相同 metric formulas；
- 完全相同 aggregation。

在未满足前，只能用作“目标量级”和“需要关注的短板”，不能正式判胜负。

---

## 5. 研究问题与预注册假设

### RQ1：状态—坐标事务是否必要

比较：

~~~text
strict Human3R state carry
clean reset
no-V9 raw SE(3)
B0
full Movie3R
~~~

假设：

- carry 会在大视角 cut 出现 state contamination；
- reset 会出现独立 gauge；
- B0 能显著改善 camera/world trajectory 与跨 shot ID 可观测性；
- full 方法进一步改善 human structure 和安全性。

### RQ2：B0 是否真正帮助跨 shot ID

比较：

~~~text
native Human3R detection index
raw/reset explicit association
B0 + association
B0 + association + geometry gate
~~~

假设：

- close interaction 和遮挡使 frame-local detection order 更不稳定；
- B0 后 root/torso/centered-body matcher 的 permutation margin 更大；
- persistent ID continuity 显著提升。

### RQ3：相机可信时，BRTC 是否只修人体而不破坏场景

假设：

- BRTC-LC 改善 fixed-world root/joint/vertex 和 pairwise layout；
- camera ATE/RPE 与 B0 bit-exact；
- rejected/unmatched track 与 B0 exact 相同。

### RQ4：C1 是否减少静止人物漂移且不抹除真实运动

假设：

- static subset 的 within-shot root speed、drift、Accel 下降；
- moving subset 的运动幅度和方向不产生显著衰减；
- severe interaction motion 会更多进入 fallback，而不是被强制平滑。

### RQ5：adaptive joint 在 Harmony4D 上是否安全

Harmony4D 的强遮挡和身体接触可能导致 shared Kabsch residual 假一致或错误一致。

假设：

- 可观测 cut 上联合 camera-human update 改善相机和人体；
- 歧义 cut 上 gate 应 abstain；
- false acceptance rate 与 gate harm 必须单独报告；
- 不能只报告 accepted case。

### RQ6：在线方法是否有实际系统优势

假设：

- Movie3R 每个 cut 只增加一次 same-model shadow forward 和 CPU geometry；
- 不依赖未来帧；
- 相比 Multi-THuMBS 的 150 帧约 10 分钟离线优化，Movie3R 应有明显更低的 boundary overhead；
- 必须实测 wall time、FPS、VRAM 和 CPU overhead，不能只写理论。

---

## 6. 数据划分与防泄漏规则

### 6.1 最终 test

只使用官方 outer archive 的：

~~~text
test/01_hugging.zip
test/03_grappling2.zip
test/05_sword2.zip
test/06_sword3.zip
test/08_ballroom2.zip
test/15_mma4.zip
test/16_mma5.zip
~~~

所有 test 序列的：

- 模型结果；
- gate decisions；
- failure cases；
- 每序列指标；
- 聚合指标；

都必须完整保留，不能只选好看的 clip。

### 6.2 Dev / calibration

只从 train ZIP 中选择开发序列。

建议最小开发顺序：

1. train/01_hugging.zip：schema、adapter、smoke test；
2. train/15_mma4.zip：高速、遮挡、contact；
3. 再选择一个 sword 或 grappling 序列：长武器/复杂肢体结构与快速朝向变化；
4. 如需 detector 校准，再增加 ballroom。

选择 dev 序列必须在读取 test 指标前冻结。

### 6.3 官方 train/test 仍需审计

部分 train/test ZIP 名称相同，如 01_hugging、03_grappling2、05_sword2、08_ballroom2、15_mma4。执行时必须确认它们是：

- 不同 camera views；
- 不同 frames；
- 不同 clips；
- 或官方其他划分方式。

如果 train/test 存在相同人物或相同时间段，论文必须准确写：

> official Harmony4D split generalization

而不能擅自写：

> unseen-subject generalization

### 6.4 严禁 test tuning

以下行为禁止：

- 看 test 指标后调整 gate threshold；
- 根据 test failure 选择 camera pair；
- 删除 detector 误报 case；
- 只报告 gate accepted case；
- 用 GT identity 进入推理；
- 用 GT camera 决定 runtime 分支；
- 用 test 训练 detector/V9/ID weight。

---

## 7. Cross-shot 协议设计

Multi-THuMBS 未公开 exact manifest，因此先冻结我们自己的可审计协议。

协议名称：

> **Movie3R-Harmony4D-CrossShot-v1**

### 7.1 主协议：H4D-CS150

目标：

- 与论文公开的 150-frame 系统规模相近；
- 足够计算 trajectory、Accel、within-shot drift；
- 仍保持 first-frame causal。

拟定：

~~~text
clip length = 150 frames
shot 1 = camera A, frames s ... b-1
shot 2 = camera B, frames b ... e
boundary decision uses only A@b-1 and B@b
no repeated future information
no post-shot lookahead
~~~

默认可先采用 75 + 75，但只有在 schema/fps/有效帧审计后正式冻结。

重要：

- clip 长度参考 Multi-THuMBS 公开 runtime 示例；
- 不声称其就是论文官方 evaluation clip length；
- 一旦 manifest 冻结，不能根据模型结果改 cut。

### 7.2 Boundary-synchronized 诊断协议

名称：

> H4D-BoundarySync

构造同步 camera A/B 的同一 timestamp boundary pair，用于隔离：

- camera gauge；
- identity permutation；
- human-camera relative geometry；
- B0 与 BRTC 的边界精度。

它适合 5 pre + 25 post 的快速诊断，但不作为主表 trajectory/Accel 唯一结果，因为重复 timestamp 会影响物理运动解释。

### 7.3 多 cut 压力协议

名称：

> H4D-CS150-3Shot

在 150 帧内使用三个同步 camera views，例如 50 + 50 + 50：

- 测试 transaction 累积；
- 测试 ID continuity；
- 测试 gauge drift；
- 作为 appendix，不替代 two-shot 主表。

### 7.4 Camera pair 分层

使用 GT calibration 仅用于 manifest 构造和离线分层，不能进入推理。

按 boundary camera rotation span 分层：

~~~text
small:       0°–30°
medium:     30°–60°
large:      60°–120°
extreme:   120°–180°
~~~

同时记录：

- camera-center baseline；
- overlap ratio；
- RGB texture score；
- person count；
- visible joint ratio；
- truncation ratio；
- occlusion level；
- motion magnitude；
- interaction type；
- body orientation difference。

### 7.5 Clip 选择

最终 clip 不能由 Movie3R 表现决定。

执行顺序：

1. 先根据 GT calibration/visibility 建立所有 eligible candidates；
2. 固定 eligibility；
3. 按 sequence 和角度分层；
4. 如果全量过大，使用固定 seed 做 deterministic balanced sampling；
5. 保存候选全集、排除理由、最终 manifest；
6. 对 manifest 计算 SHA256；
7. 冻结后才运行任何方法。

### 7.6 建议样本规模

先以数据审计后的 eligible 数量为准，不提前虚构官方 clip 数。

论文级最低要求：

- 7 个 test sequences 全覆盖；
- 每个 sequence 覆盖至少两个 viewpoint strata；
- 总 cut 数足以对 sequence/cut 做 bootstrap CI；
- 主表不能只用个位数 cuts；
- 如果资源允许，优先全量 deterministic manifest；
- 如果需要上限，预注册每 sequence/stratum 的固定样本数，而不是按结果停止。

---

## 8. 磁盘与逐序列解压策略

### 8.1 工作目录

计划使用：

~~~text
archive:
/data/wangzheng/iJCV-CODE/data/Harmony4D.zip

staging:
/data/wangzheng/iJCV-CODE/data/Harmony4D_work

temporary:
/data/wangzheng/iJCV-CODE/data/Harmony4D_work/tmp

compact outputs:
/data/wangzheng/iJCV-CODE/Movie3R/output/v15_harmony4d
~~~

所有 TMPDIR 必须指向 /data 下的显式目录，不使用根目录、/tmp 或用户 home 存储大文件。

### 8.2 单序列生命周期

每个 nested ZIP 严格执行：

~~~text
从外层 ZIP 提取一个 nested ZIP 到 .partial
→ 核对字节数
→ 按 .harmony4d_download_state.json 核对 SHA256/LFS OID
→ 原子改名
→ zipinfo 读取 inner uncompressed size
→ free-space gate
→ unzip -tq 做完整 CRC
→ 解压到 sequence-specific staging
→ schema audit / inference / evaluation
→ 保存 compact predictions + metrics + logs
→ 结果完整性检查
→ 删除该 sequence 的展开副本与临时 nested ZIP
→ 进入下一个 sequence
~~~

### 8.3 空间 gate

任何序列解压前要求：

~~~text
free_bytes >
inner_uncompressed_bytes
+ nested_zip_bytes
+ expected_output_bytes
+ safety_reserve
~~~

建议 safety reserve：

~~~text
至少 80 GiB
~~~

若不满足，停止，不做冒险解压。

### 8.4 处理顺序

先小后大：

~~~text
01_hugging
15_mma4
05_sword2
08_ballroom2
16_mma5
03_grappling2
06_sword3
~~~

最小序列先用于验证 adapter，不要一开始处理 16GB nested archive。

### 8.5 绝不删除的内容

在 Harmony4D 专项实验完成且用户明确同意前，不删除：

~~~text
/data/wangzheng/iJCV-CODE/data/Harmony4D.zip
~~~

只清理可重建的：

- extracted nested ZIP；
- expanded sequence staging；
- temporary image cache；
- temporary pointmaps。

删除前必须保证：

- source outer archive 仍存在；
- compact outputs 已校验；
- manifest/log/metrics 已落盘；
- selected qualitative payload 已保存。

---

## 9. Dataset Adapter 与坐标审计

在跑模型前必须先证明 GT 被正确读取。

### 9.1 必须识别的数据字段

- RGB camera names；
- synchronized timestamps；
- fps；
- image resolution；
- camera intrinsics；
- camera extrinsics；
- distortion model；
- world-to-camera 或 camera-to-world convention；
- coordinate handedness；
- length unit；
- stable person identity；
- 2D joints；
- 3D joints；
- SMPL/mesh parameters；
- mesh topology；
- visibility；
- bounding boxes；
- masks；
- truncation/occlusion metadata。

### 9.2 Camera convention audit

至少做：

1. GT 3D joint 投影到所有相机；
2. 与 GT 2D joint 比较；
3. 检查 x/y 轴、图像 origin、单位和 distortion；
4. 检查 camera center 与 extrinsic inverse；
5. 可视化至少 3 个序列、3 个相机、10 个随机帧；
6. 报告 reprojection median/P95；
7. 无系统性镜像、90°旋转、尺度或主点偏差后才能继续。

### 9.3 人体 topology audit

Harmony4D GT 与 Movie3R/Human3R 的人体模型可能不同：

- Harmony4D 可能提供 SMPL/HMR2.0 topology；
- Movie3R 输出 SMPL-X；
- 不能直接对不同 vertex index 计算 MPVPE。

必须冻结：

1. common 24-joint regressor；
2. SMPL-X → SMPL 或 common surface correspondence；
3. pelvis/root 定义；
4. global orientation convention；
5. body scale；
6. vertex subset 或 transfer matrix；
7. topology version/hash。

如果没有可靠的逐顶点 transfer：

- 主表 MPVPE 必须标明 common-surface protocol；
- 不能用错误 vertex index 得到看似很低的值；
- 同时报告 joint 与 root 指标。

### 9.4 Stable identity audit

确认：

- GT ID 是否跨 camera 一致；
- train/test 同名序列的 ID 语义；
- person enter/exit；
- invisible person 是否仍在 annotation；
- fully occluded frame；
- partially visible joints；
- interaction中人物接触导致的标注交换。

### 9.5 Adapter 单元测试

必须包含：

- camera inverse round-trip；
- world→camera→world round-trip；
- 3D→2D reprojection；
- identity persistence；
- mesh joint regression；
- units；
- frame synchronization；
- missing annotation；
- variable person count；
- visibility mask；
- deterministic manifest generation。

---

## 10. 方法比较矩阵

所有内部方法使用相同 RGB、相同 manifest、相同 evaluation topology、相同 aggregation。

### 10.1 必做内部消融

| ID | Method | 目的 |
|---|---|---|
| M0 | Strict original Human3R | 原始 state 跨 cut 的真实 baseline |
| M1 | Clean reset per shot | state 干净但 gauge 独立 |
| M2 | No-V9 raw camera SE(3) | 验证显式 raw camera bridge 是否足够 |
| M3 | B0 only | V9 learned coarse gauge 单独贡献 |
| M4 | B0 + anonymous identity | persistent cross-shot ID 单独贡献 |
| M5 | B0 + ID + BRTC-LC | camera-frozen human root/depth/layout |
| M6 | B0 + ID + BRTC-LC + C1 | shot 内静止人体稳定 |
| M7 | Full v15 + oracle boundary | 排除 detector 的方法上界 |
| M8 | Full v15 + causal GRU boundary | 默认部署结果 |
| M9 | Full + static/logistic detector | detector 消融 |

### 10.2 Checkpoint policy

Movie3R 主表：

- 使用 v15 primary_multihuman checkpoint；
- checkpoint SHA256 必须写入每个 case manifest；
- M1–M8 使用同一个当前 checkpoint 的对应 branch/cache；
- 不混合不同 checkpoint 的 camera/human 指标。

Strict Human3R：

- 使用官方 Human3R checkpoint；
- 单独标注为 external backbone baseline；
- 不能把它的某一指标拼入 Movie3R 行。

Cross-source checkpoint：

- 只用于 checkpoint sensitivity appendix；
- 不与 primary_multihuman 的模块消融混成同一行。

### 10.3 外部 baseline

优先级：

1. Multi-THuMBS literature reference；
2. Multishot literature/executable；
3. GVHMR；
4. PromptHMR；
5. HSfM†；
6. KPR/Pose2ID 仅用于 identity-only table；
7. VGGT 仅用于 camera-only table。

如果外部代码、split 或 evaluator不可得：

- 放在 literature-only 表；
- 明确 Required inputs / future frames / optimization / hardware；
- 不与本地数字混为“同协议直接比较”。

### 10.4 同-forward cache

Movie3R 内部消融尽量来自同一 GPU forward：

- raw/reset；
- shadow；
- B0；
- identity；
- BRTC；
- C1；
- adaptive。

这样可以避免：

- detector 差异；
- stochastic HMR 差异；
- checkpoint 差异；
- image preprocessing 差异；
- 重复 GPU 计算。

---

## 11. GPU 执行协议

### 11.1 当前 v15 的现实约束

当前：

- versions/v15/run_case.py 强制 CUDA_VISIBLE_DEVICES 为空；
- versions/v15/run_batch.py 也按 CPU release contract 运行。

用户本次要求 GPU，因此不能假装现有 wrapper 已支持 GPU。

后续必须新增 Harmony 专用、可审计的 GPU runner，或调用 v14 exporter 的 GPU 路径：

~~~text
versions/v15/harmony4d/run_harmony_case.py
versions/v15/harmony4d/run_harmony_batch.py
~~~

不能为了 GPU 静默改变模型输出契约。

### 11.2 GPU parity

先用 train smoke clip 比较：

- CPU reference；
- CUDA FP32；
- 可选 AMP。

至少核对：

- camera matrices；
- detected person count；
- SMPL-X root/joints；
- B0 transform；
- identity permutation；
- BRTC accept/reject；
- adaptive accept/reject；
- final metrics。

默认先用 CUDA FP32。只有 AMP 的数值差异和 gate decision 全部安全后，才可用于批量。

### 11.3 运行记录

每次运行记录：

- GPU model；
- driver；
- CUDA；
- cuDNN；
- PyTorch；
- Python env；
- checkpoint SHA256；
- git commit；
- seed；
- precision；
- image resolution；
- frames；
- wall time；
- GPU peak memory；
- CPU peak memory；
- detector latency；
- shadow-forward latency；
- explicit geometry latency。

### 11.4 GT 隔离

GPU inference 进程只读：

- RGB；
- frozen checkpoint；
- causal state；
- boundary event。

GT 只在独立 evaluator 中读取。

运行产物必须记录：

~~~text
gt_in_runtime = false
future_frames_at_boundary = 0
pre_cut_frames_mutated = false
~~~

---

## 12. Multi-THuMBS 命名指标

由于官方公式未公开，本地结果必须加后缀：

> Multi-THuMBS-named provisional protocol

### 12.1 W-MPJPE

本地冻结候选公式：

~~~text
对每个 stable GT identity
在最初两个可见 frames 的全部 common joints 上拟合一个 Sim(3)
将该 Sim(3) 应用于整条 predicted world trajectory
计算全轨迹 mean joint error
单位 mm
~~~

用途：

- trajectory/world consistency；
- 对跨 shot gauge 敏感。

同时额外报告 one-frame-fit 版本，避免“initial frame”含义不确定。

### 12.2 WA-MPJPE

~~~text
对每个 stable GT identity
在全部可见 trajectory frames 上拟合一个 Sim(3)
计算 aligned world joint error
单位 mm
~~~

用途：

- trajectory-level aligned body/shape accuracy；
- 比 W-MPJPE 更宽容。

### 12.3 MPJPE

~~~text
camera-coordinate common joints
pred/GT 分别 pelvis centered
matched visible person-frames
单位 mm
~~~

必须同时报告 coverage，因为 matched-only MPJPE 可能奖励漏检。

### 12.4 MPVPE

~~~text
camera-coordinate common mesh topology
pred/GT 分别 root/pelvis centered
单位 mm
~~~

必须先通过 topology audit。

### 12.5 Accel

同时保存两种，不冒充官方唯一公式：

~~~text
Accel-Δ²:
mean ||(X_pred[t-1]-2X_pred[t]+X_pred[t+1])
     -(X_gt[t-1]-2X_gt[t]+X_gt[t+1])||
mm/frame²

Accel-physical:
上式乘 fps²
m/s²
~~~

同时报告：

- world joint Accel；
- pelvis-centered Accel；
- static subset；
- moving subset；
- boundary-only；
- within-shot-only。

### 12.6 ATE

至少报告：

1. Sim(3)-aligned camera-center ATE RMSE；
2. SE(3)-aligned ATE；
3. metric no-scale ATE；
4. per-clip；
5. per-sequence macro；
6. all-frame micro。

Multi-THuMBS literature reference 0.7 只与其 unknown ATE 并列参考，不直接判胜。

### 12.7 IDs

Evaluator-side：

1. 用 GT 2D/3D association 将 prediction 对应到 stable GT identity；
2. 沿 GT track 统计 predicted persistent ID 改变；
3. 报告 total；
4. mean per clip；
5. per cut；
6. per GT identity；
7. re-entry；
8. unmatched gap；
9. coverage。

GT association只用于评价，不能进入 Movie3R runtime matcher。

---

## 13. Movie3R 专属指标

仅报告 Multi-THuMBS 命名指标不足以证明本项目创新。

### 13.1 Boundary Camera Error

在 pre-shot GT gauge 中报告第一 post frame：

- camera translation error，m；
- camera rotation geodesic error，degree；
- boundary RPE translation；
- boundary RPE rotation；
- post-shot mean/P90。

这是区分：

~~~text
camera 已正确但人体错
camera 与人体都错
~~~

的关键指标。

### 13.2 Fixed-world Human Error

不做 pelvis alignment：

- root error；
- world joint error；
- world vertex error；
- first-post；
- full post-shot；
- accepted/rejected split。

它能直接反映 BRTC 和 adaptive 的世界对齐能力。

### 13.3 Camera–Human Relative Gauge Error

定义：

~~~text
q_i = R_camera^T (root_i - camera_center)
CHRGE = mean ||q_pred - q_GT||
~~~

再报告 relative body orientation error。

意义：

- 判断 camera 和 human 是否虽然各自看似合理，但相对位置仍错；
- 直接对应项目核心问题。

### 13.4 Pairwise Layout Error

多人：

~~~text
pair distance error
= | ||r_i-r_j||_pred - ||r_i-r_j||_GT |

pair vector error
= ||(r_i-r_j)_pred - (r_i-r_j)_GT||
~~~

这是 BRTC-LC 和多人 layout 的核心指标。

### 13.5 Cut Seam Error

定义 boundary excess jump：

~~~text
E_seam =
|| (x_pred[b]-x_pred[b-1])
 - (x_GT[b]-x_GT[b-1]) ||
~~~

分别报告：

- camera seam；
- root seam；
- world joint seam；
- world vertex seam；
- camera-human relative seam。

### 13.6 Identity Quality

除 IDs 外：

- association accuracy；
- permutation all-correct rate；
- persistent-ID continuity；
- IDF1；
- GT identity coverage；
- unmatched rate；
- false-match rate；
- count-change handling；
- confidence margin calibration。

### 13.7 Within-shot Stability

静止 subset：

- root speed；
- root drift；
- vertex drift；
- camera-local drift；
- world Accel；
- C1 correction magnitude。

运动 subset：

- motion attenuation ratio；
- trajectory direction cosine；
- false-static rate；
- oversmoothing harm。

### 13.8 Gate Safety

对 adaptive gate 报告：

- acceptance rate；
- abstention rate；
- true beneficial accept；
- harmful accept；
- beneficial reject missed；
- gate harm rate；
- mean harm；
- P95 harm；
- catastrophic harm count；
- accept/reject 分别的 camera/human error。

父方法定义为：

> exact B0+BRTC+C1

Harm 不能只用单一指标，至少报告：

~~~text
camera translation
camera rotation
fixed-world root
world vertex
pair layout
ID
~~~

### 13.9 Detector

- Precision；
- Recall；
- F1；
- false-positive rate；
- false-negative rate；
- detection delay；
- calibration/Brier；
- downstream false-update rate；
- geometry gate 对 detector 误报的拦截率。

### 13.10 Online Cost

- FPS；
- ms/frame；
- boundary total overhead；
- extra shadow forward；
- CPU geometry overhead；
- peak VRAM；
- peak RAM；
- persistent-state size；
- future frames；
- per-video optimization iterations；
- total 150-frame runtime。

---

## 14. Coverage、漏检与遮挡

Harmony4D 强遮挡使 coverage 成为主指标，而不是附注。

每个方法必须报告：

~~~text
visible GT person-frames
matched person-frames
missed person-frames
false-positive detections
coverage
precision
recall
occlusion-stratified coverage
truncation-stratified coverage
~~~

人体指标输出两套：

1. matched-only error；
2. recall-aware summary。

recall-aware summary 至少包含：

- coverage 与误差并列；
- 不允许漏检 case 从平均误差中静默消失；
- failure table列出严重遮挡/截断的 miss。

按可见性分层：

~~~text
high visibility
partial occlusion
severe occlusion
truncated
person count changed
~~~

---

## 15. 聚合与统计

### 15.1 三种聚合

同时给：

1. person-frame weighted micro；
2. clip macro；
3. sequence macro。

论文主表优先 sequence/clip macro，防止长序列主导。

### 15.2 分布统计

每项至少：

- mean；
- median；
- P90；
- P95；
- standard deviation；
- 95% bootstrap CI。

### 15.3 配对显著性

Movie3R 内部消融是同一 clip 的 paired comparison：

- paired bootstrap；
- permutation test 或 Wilcoxon signed-rank；
- 以 clip/sequence 为 resampling unit；
- 不用 frame-level pseudo-replication。

### 15.4 分层分析

必须按：

- cut angle；
- camera baseline；
- texture；
- occlusion；
- truncation；
- person count；
- motion magnitude；
- interaction class；
- gate accept/reject；
- detector correct/incorrect；

生成表格或曲线。

---

## 16. 如果指标明显差：允许的调整策略

### 16.1 先诊断，不直接改方法

失败必须归类：

1. dataset adapter / coordinate bug；
2. camera convention；
3. SMPL/SMPL-X topology；
4. Human3R detection failure；
5. Human3R local pose failure；
6. V9/B0 gauge failure；
7. ID permutation failure；
8. BRTC observability reject；
9. BRTC wrong accept；
10. C1 false-static；
11. adaptive false accept；
12. detector miss/false alarm；
13. severe occlusion/unmatched；
14. scale/root convention。

数据或坐标 bug 修复不算方法调整，但必须记录。

### 16.2 所有方法调整只在 train/dev

可以探索：

- visibility-aware identity cost；
- torso/joint/mesh feature weighting；
- association dustbin/unmatched policy；
- BRTC joint visibility weights；
- ray gap/parallax/MAD calibration；
- layout consensus robustness；
- C1 static/moving threshold；
- adaptive normalized RMS/margin calibration；
- detector threshold；
- detector 在 Harmony train 上的 calibration；
- common topology mapping。

不优先引入：

- 新的大型预训练模型；
- 外部 Re-ID backbone；
- 离线 SLAM；
- future-frame smoothing；
- full-sequence bundle adjustment。

### 16.3 V9 是否需要 Harmony 适配

先跑 frozen v15。

只有满足以下证据才考虑：

- B0 对 Harmony train/dev 在多角度系统性失败；
- raw 与 shadow 的 error decomposition 明确；
- failure 不是坐标/topology/visibility；
- 后续 geometry 无法安全挽救；
- 有 pair-disjoint train/dev。

若适配 V9：

- 新建版本，不覆盖 v15；
- frozen v15 仍是主 baseline；
- 只用 train；
- test 只跑一次；
- 单独报告 +Harmony adaptation。

### 16.4 不允许静默改 v15

任何数据集特化变化必须成为明确行：

~~~text
Movie3R-v15 frozen
Movie3R + H4D visibility calibration
Movie3R + H4D detector calibration
Movie3R-vNext
~~~

不得将 tuned 结果仍标为原始 v15。

### 16.5 Promotion gate

一个调整只有在 train/dev 同时满足才晋升：

- mean 改善；
- P90/P95 不恶化；
- catastrophic count 不增加；
- coverage 不下降；
- ID 不恶化；
- camera 不被破坏；
- moving subset 不被过平滑；
- gate harm 不增加；
- 在线/因果约束保持；
- 不增加额外大型预训练模型。

最多保留 1 个主调整和不超过 2 个有解释力的消融，不让论文主线变成数据集特化模块堆叠。

---

## 17. 分阶段执行计划

### Phase 0：冻结实验合同

产物：

- protocol version；
- directory policy；
- checkpoint hashes；
- code commit；
- GPU policy；
- no-GT runtime contract；
- failure taxonomy；
- result schema。

通过条件：

- Git clean；
- v15 assets hash 通过；
- 输出目录固定；
- 磁盘 reserve 固定；
- 不解压全量数据。

### Phase 1：最小 train 序列 schema audit

只处理最小 train sequence。

任务：

- nested ZIP hash/CRC；
- inner size；
- RGB/GT tree；
- camera convention；
- identity；
- topology；
- fps；
- projection visualization。

通过条件：

- GT 3D→2D 正确；
- world/camera convention 明确；
- common joints/vertices 明确；
- person IDs 稳定；
- 适配测试全部通过。

### Phase 2：构造 dev protocol

任务：

- camera-pair candidates；
- visibility/occlusion；
- angle bins；
- 150-frame clips；
- boundary-sync diagnostics；
- train/dev disjoint；
- manifest freeze；
- SHA256。

通过条件：

- cut 不依赖模型结果；
- 每个 stratum 有样本；
- 所有 exclusion 有理由；
- manifest 可复现。

### Phase 3：Evaluator 闭环

复用并扩展：

~~~text
versions/v14/eval_multithumbs_protocol.py
versions/v14/eval_brtc_multithumbs_egohumans.py
~~~

新增：

- Harmony adapter；
- topology mapper；
- camera error；
- fixed-world errors；
- seam；
- pair layout；
- coverage；
- gate harm；
- detector；
- runtime。

通过条件：

- synthetic self-tests；
- perfect prediction 为 0 error；
- known transform test 正确；
- ID switch toy case 正确；
- miss/entry/exit toy case 正确；
- CPU evaluator deterministic。

### Phase 4：GPU runner 与 parity

任务：

- 不修改现有 CPU release contract；
- 新增 Harmony GPU entrypoint；
- train smoke clip；
- CUDA FP32 parity；
- optional AMP parity；
- runtime/VRAM logging；
- crash recovery。

通过条件：

- method decisions一致；
- numerical tolerance可解释；
- 无 root/tmp 大缓存；
- 断点续跑；
- case 原子完成。

### Phase 5：Frozen v15 dev 全消融

先跑：

~~~text
M0 strict Human3R
M1 clean reset
M2 no-V9
M3 B0
M4 B0+ID
M5 B0+ID+BRTC
M6 +C1
M7/M8 full
~~~

输出：

- 每 clip JSON；
- aggregate CSV；
- diagnostics；
- failure list；
- selected demo payload。

通过条件：

- 同-forward provenance；
- 所有 method rows 齐全；
- coverage 齐全；
- no hidden failed cases。

### Phase 6：差距诊断与可选适配

若 frozen v15 已达到明确优势：

- 不为了刷单一数字继续调；
- 直接进入 test。

若明显差：

1. 先修数据/evaluator；
2. 再做 failure taxonomy；
3. 只在 train/dev 探索；
4. 冻结最多一个主候选；
5. 新版本/新方法名；
6. 在 dev 过 promotion gate。

### Phase 7：冻结 Test Manifest

在 test forward 前完成：

- 7 sequences schema；
- eligible candidate list；
- deterministic sampling；
- H4D-CS150；
- H4D-BoundarySync；
- optional 3Shot；
- manifest hash；
- data version；
- evaluator version；
- topology hash。

一旦开始 test，不再修改。

### Phase 8：逐序列 GPU test

顺序：

~~~text
01_hugging
15_mma4
05_sword2
08_ballroom2
16_mma5
03_grappling2
06_sword3
~~~

每序列：

1. extract；
2. verify；
3. run all required methods/caches；
4. evaluate；
5. save compact artifacts；
6. integrity check；
7. cleanup staging；
8. update global ledger。

任何序列失败：

- 记录；
- 修复运行问题；
- 不删除该 case；
- 不能将其排除以改善结果。

### Phase 9：统计与论文表

生成：

- main human table；
- camera/ID table；
- Movie3R-specific boundary table；
- efficiency table；
- ablation；
- stratified plots；
- gate reliability；
- failure cases；
- 95% CI；
- LaTeX tables。

### Phase 10：定性可视化

至少保存：

- one easy textured pair；
- one large-angle pair；
- one severe occlusion；
- one truncation；
- one ID permutation fixed；
- one adaptive accept；
- one safe abstention；
- one true failure。

统一 demo.py 风格：

- RGB background；
- camera frustums；
- pointmap；
- color-consistent people；
- GT 只在独立 comparison viewer；
- 30-frame 和 150-frame 两种。

### Phase 11：收尾与清理

保留：

- outer Harmony4D.zip；
- manifests；
- code；
- compact forward cache；
- metrics；
- logs；
- selected demos；
- paper tables/plots。

删除：

- expanded raw sequence；
- temporary nested ZIP；
- redundant full pointmaps；
- failed partial archives；
- duplicate render frames。

删除行为在日志中记录路径与可恢复来源。

---

## 18. 论文表格设计

### Table A：Harmony4D Multi-shot Human Reconstruction

列：

~~~text
W-MPJPE
WA-MPJPE
MPJPE
MPVPE
Accel
Coverage
~~~

行：

~~~text
Literature reference block
Multishot
GVHMR
PromptHMR
HSfM†
Multi-THuMBS

Executable protocol block
Strict Human3R
Clean reset
No-V9
B0
B0+BRTC
B0+BRTC+C1
Full Movie3R
~~~

两个 block 必须视觉分隔，并标明不可直接比较。

### Table B：Camera and Identity

列：

~~~text
ATE Sim(3)
ATE SE(3)
RPE trans
RPE rot
IDs/clip
IDs/cut
IDF1
association accuracy
coverage
~~~

### Table C：Boundary Transaction

列：

~~~text
camera t/R at boundary
root
world joint
world vertex
camera-human relative
pair distance
pair vector
seam jump
gate harm
~~~

这张表是 Movie3R 的独特贡献表。

### Table D：Causal Efficiency

列：

~~~text
future frames
per-video optimization
extra pretrained models
FPS
150-frame runtime
boundary overhead
peak VRAM
peak RAM
gate acceptance
~~~

### Appendix

- oracle vs causal detector；
- checkpoint sensitivity；
- angle/occlusion/person-count bins；
- static vs moving C1；
- accepted vs rejected gate；
- topology mapping；
- coordinate audit；
- all sequence results；
- all failure cases。

---

## 19. 论文图设计

### Figure 1：Protocol construction

显示：

~~~text
synchronized Harmony4D cameras
→ deterministic camera switch
→ pre shot / post shot
→ no future frame at boundary
~~~

### Figure 2：Method qualitative

对比：

- strict Human3R；
- clean reset；
- B0；
- B0+BRTC；
- Full。

同时显示：

- camera；
- same-color person IDs；
- scene；
- boundary seam。

### Figure 3：Cut-angle / occlusion curves

x：

- cut angle；
- visibility；
- occlusion。

y：

- ATE；
- W-MPJPE；
- IDs；
- gate harm。

### Figure 4：Gate reliability

- residual confidence vs true improvement；
- acceptance/abstention；
- false accept；
- calibration；
- coverage。

### Figure 5：Runtime-quality tradeoff

Movie3R 与 literature methods：

- online/future；
- runtime；
- W/WA/ATE/IDs。

协议不同的 literature point 使用空心 marker。

---

## 20. 成功标准

### 20.1 最低论文完成标准

必须全部满足：

- 7 个 Harmony4D test sequences 全覆盖；
- manifest 冻结且有 SHA256；
- strict Human3R、reset、no-V9、B0、BRTC、C1、Full 全部跑完；
- oracle 与 causal detector 分开；
- W/WA/MPJPE/MPVPE/Accel/ATE/IDs 齐全；
- Movie3R-specific camera/root/layout/seam/gate 指标齐全；
- coverage 与 miss/FP 齐全；
- 95% CI；
- test 无调参；
- GPU/runtime/VRAM 齐全；
- 至少 7 类定性案例；
- 失败案例不隐藏；
- 结果可从 manifest 重跑；
- 磁盘临时数据清理完成。

### 20.2 方法有效标准

Full Movie3R 在同协议下应：

- 显著优于 strict Human3R；
- 显著优于 clean reset；
- 在 W/WA、ATE、IDs 和 seam 上优于 B0-only；
- BRTC 在相机不变时改善 fixed-world human/layout；
- C1 改善 static stability/Accel，moving harm 可控；
- adaptive accepted cuts 平均获益；
- harmful accept 稀少且透明；
- rejected cuts exact fallback；
- no-cut/可信多人相机不被破坏。

### 20.3 文献目标量级

在协议未公开前，仅作为目标：

~~~text
W-MPJPE       < 221.0 mm
WA-MPJPE      < 116.9 mm
MPJPE         < 215.9 mm
MPVPE         < 278.3 mm to beat Multi-THuMBS
MPVPE         < 257.6 mm to beat best Table-1 value
Accel         < 17.4
ATE           < 0.7
IDs           < 0.46
~~~

即使本地数值达到，也只能先写：

> under Movie3R-Harmony4D-CrossShot-v1

不能写：

> officially surpasses Multi-THuMBS

---

## 21. 风险与应对

| 风险 | 后果 | 应对 |
|---|---|---|
| 内层解压远大于 ZIP | 磁盘满 | zipinfo 预估、80GiB reserve、逐序列 |
| 外层 archive 被误删 | 数据不可恢复 | 全程只读保留 |
| v15 wrapper 强制 CPU | 不满足 GPU 要求 | 独立 GPU runner + parity |
| GT camera convention 错 | 所有指标伪结果 | reprojection audit |
| SMPL/SMPL-X topology 不同 | MPVPE 无效 | common topology freeze |
| train/test 同名序列泄漏 | 泛化 claim 错误 | 官方 split/subject audit |
| close interaction 导致 ID 错配 | IDs 与 geometry 崩溃 | visibility-aware association dev study |
| BRTC 射线因遮挡不可观测 | 大量 reject | 报 coverage；只在 train/dev改权重 |
| C1 平滑真实打斗/舞蹈 | Accel好看但运动错误 | moving subset harm |
| adaptive Kabsch 被接触姿态误导 | 相机灾难更新 | false-accept/harm gate |
| official protocol 缺失 | 无法直接判胜 | 两层表、发布自有协议 |
| pointmap/cache 太大 | 输出占满磁盘 | compact cache、在线评估、少量 demo |
| test 调参诱惑 | 论文不可信 | frozen manifest，test 单次 |

---

## 22. 最终产物

计划代码：

~~~text
versions/v15/harmony4d/
  README.md
  protocol.py
  dataset.py
  topology.py
  build_manifest.py
  run_harmony_case.py
  run_harmony_batch.py
  evaluate_harmony.py
  aggregate_harmony.py
  tests/
~~~

计划协议：

~~~text
versions/v15/harmony4d/protocols/
  h4d_cs150_dev.jsonl
  h4d_cs150_test.jsonl
  h4d_boundary_sync_test.jsonl
  h4d_cs150_3shot_test.jsonl
  protocol_spec.json
  checksums.sha256
~~~

计划输出：

~~~text
output/v15_harmony4d/
  archive_audit/
  schema_audit/
  manifests/
  predictions/
  metrics/
  aggregates/
  runtime/
  diagnostics/
  qualitative/
  paper_tables/
  paper_figures/
  logs/
  ledger.jsonl
~~~

最终文档：

~~~text
versions/v15/HARMONY4D_ICLR_EXPERIMENT_FINAL_REPORT.md
~~~

最终报告必须包含：

- protocol；
- data provenance；
- checkpoint/code hashes；
- all methods；
- all metrics；
- confidence intervals；
- failures；
- literature caveat；
- disk cleanup；
- ICLR claim ledger。

---

## 23. 执行后的停止条件

Harmony4D 专项任务只有满足以下条件才算完成：

1. 7 个 test sequence 全部处理；
2. 所有必做方法有结果或有不可恢复的明确证据；
3. 所有 paper-named 和 Movie3R-specific 指标计算；
4. 所有结果有 coverage；
5. 所有调整仅来自 train/dev；
6. final test manifest 未在看结果后改变；
7. GPU、runtime、VRAM 完整；
8. 主表、消融、分层、定性、失败齐全；
9. literature comparison 表述合规；
10. 最终报告与 LaTeX 表生成；
11. 展开数据和临时文件清理；
12. Git 工作区按功能提交。

在这些条件满足前，不因单个漂亮案例或某一项指标较好而提前结束。

---

## 24. 下一步入口

收到用户“开始执行”后，第一步不是全量解压，而是：

~~~text
1. 再检查 /data 空间
2. 验证 outer archive 与 download-state metadata
3. 只提取最小 train/01_hugging nested ZIP
4. 获取 inner uncompressed size
5. 通过 free-space gate
6. 完成 schema/camera/topology/identity audit
7. 写 adapter self-tests
8. 再决定第一个 GPU smoke clip
~~~

本计划之外的扩展，例如全量外部 baseline 复现、V9 重训或新预训练模型接入，需要独立记录，不得悄悄混入 Harmony4D-v15 主实验。
