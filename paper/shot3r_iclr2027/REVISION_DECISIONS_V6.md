# BRIDGE3R 投稿修订：决策记录

本文件记录正式投稿前由作者确认的范围、方法、协议和呈现选择。只有标记为“已确认”的选择才可进入后续实验与论文修改。

| 编号 | 决策 | 选项 | 作者决定 | 状态 | 影响 |
|---|---|---|---|---|---|
| D01 | 正式方法路线 | Track A（固定现有方法） / Track B（改变方法并全量重跑） | Track A：保持现有 BRIDGE3R，不引入 velocity compensation 等方法变更 | 已确认 | 仅补协议、验证、统计、runtime 与写作；不因本轮修订重跑三个数据集主表 |
| D02 | Harmony4D 的历史观察 | 不作额外披露 / 中性说明为同配置跨数据集结果 / 降级为诊断性结果 | 当前正式配置先在 EgoBody 与 EgoHumans 上固定，随后不作 Harmony-specific 改动地用于 Harmony4D；不作额外历史选择披露 | 已确认 | Harmony4D 作为同配置的第三个跨数据集结果呈现；需保留内部时间线以确保论文与实验记录一致，但不在论文中将其降级为 stress test |
| D03 | 固定混合系数 \(\lambda=0.5\) 的论证 | 仅陈述固定配置 / 补独立消融验证 | 补做小规模独立开发集消融：\(\lambda\) sweep，并比较 median 与 mean；不立即执行，不改变正式方法或当前三数据集主表 | 已确认（延后） | 为 \(\lambda=0.5\) 与 robust shared correction 提供证据；结果出来后再更新方法解释与消融表 |
| D04 | EgoHumans 的分母呈现 | 统一 full-manifest / 统一 shared-evaluable subset / 保留现状 | 主文内部比较统一限于 90-case shared-evaluable subset；TRACE/PromptHMR 的 116-case availability ledger 移至补充材料 | 已确认 | 主文不得在同一列直接比较 90-case 内部 IDF1/coverage 与 116-case 外部数值；W/WA 同样明确为 90-case 条件结果 |
| D05 | EgoBody 外部 baseline 的主文位置 | 保留独立主文表 / 移入补充材料 / 扩展为统一 leaderboard | 保留独立主文表，作为 same-input、interface-aware 的可执行公开方法参考 | 已确认 | 不与严格内部比较合并为统一 SOTA leaderboard；Abstract 不使用其 coverage/IDF1 作为核心 headline |
| D06 | scene 的论文范围与变换叙事 | Camera--Human 聚焦 / 补 scene 指标 / 扩展为 scene--human 方法 | 聚焦 Camera--Human；scene-aware backbone 只作为空间上下文，不主张或评估 dense scene / human--scene grounding 改善 | 已确认 | 共享平移只描述为 camera--human 更新；修改前核对 \(T_0\) 与 \(\Delta\) 的真实代码作用范围 |
| D07 | Harmony4D 结果的主文位置 | 主表第三数据集 / 补充材料诊断结果 | 保留在主文三数据集主表，完整报告 W 的变差及 WA、ATE-Sim3、IDF1 的改善 | 已确认 | 正文采用 mixed-result 表述；不声称全指标优于 HUMAN3R，也不选择性隐藏 W-MPJPE |
| D08 | 正式 online runtime 与 memory 评测 | 补标准化实测 / 仅使用现有组件诊断 / 不报告 | 补标准化实测，但延后执行；固定 GPU、分辨率、precision、batch size，报告 base/no-cut/cut/amortized latency 与 peak GPU memory | 已确认（延后） | 论文预留明确 `TODO-EXPERIMENT` 表格与引用位置，未实测前不填数值；该占位规则适用于所有已确认的延后实验 |
| D09 | Boundary association 的直接证据 | 补直接匹配评测 / 仅用 IDF1 间接证明 | 补直接评测，但延后执行：first post-cut correspondence accuracy、abstention rate、人数变化/遮挡/相似姿态分层及 oracle-identity upper bound | 已确认（延后） | 归入组件验证/消融：主文核心消融加入紧凑的 association accuracy 与 IDF1；完整分层、oracle 表放补充材料，并先留 `TODO-EXPERIMENT` |
| D10 | Cut detector 的 hard-negative 证据 | 补困难无切换/非硬切评测 / 仅保留现有 hard-cut 与 no-cut 控制 | 不补；将 detector 明确限定为 same-scene hard viewpoint-cut 的因果事件提议器 | 已确认 | 不主张通用 shot-boundary detection；保留现有 hard-cut 与 no-cut 控制，正文相应收窄措辞 |
| D11 | EgoBody 主结果的配对统计 | 补 paired test / 仅保留边际 bootstrap 区间 | 补 recording-level paired bootstrap / permutation test、paired 95\% CI、median improvement 和 win rate | 已确认（延后） | 不需重跑模型；结果前不使用 `statistically significant`，并预留主文紧凑统计行与补充材料完整报告 |
| D12 | Multi-cut 证据的规模与位置 | 扩大真实多切换评测 / 保留现有 4-capture 辅助控制 | 当前保留 4 captures / 8 boundaries 的辅助控制并放补充材料；核心闭环完成后再评估扩展至 20--50 captures | 已确认（分阶段） | 当前只主张有限的 same-scene repeated-transition evidence；未来扩展覆盖 3/5/10 cuts 与 return shots 后才可增强主张 |
| D13 | Backbone 普适性主张 | 补第二 backbone 验证 / 收窄为 HUMAN3R-derived 机制 | 本投稿不补第二 backbone；全文收窄为 HUMAN3R-derived stateful reconstructor 的 causal boundary mechanism | 已确认 | 禁止将方法写成适用于所有 stateful / streaming 3D reconstruction models 的通用框架 |
| D14 | 自然编辑视频证据 | 补无 GT 的自然编辑视频集 / 不纳入本投稿主线 | 后续补小规模自然编辑视频集，作为辅助真实性证据 | 已确认（延后） | 只报告 continuity、定性图与 failure taxonomy；不与三套具 3D GT 的主定量表混合，也不扩展至任意 cross-scene editing |
| D15 | 主文定性结果的形式 | 恢复真实 RGB/mesh/camera 可视化 / 保留轨迹图 | 作者后续制作真实重建可视化；当前论文保留明确图位与作图说明，不生成或伪造视觉结果 | 已确认（延后） | 主文图应含 pre-cut、first/later post-cut RGB、Strict HUMAN3R/Reset Only/BRIDGE3R mesh、camera frusta、persistent-ID colours、boundary zoom-in 和一个失败例 |
| D16 | 正式标题 | 保持当前长标题 / 改为强调 same-scene 的紧凑标题 | `BRIDGE3R: Causal Camera--Human Bridging Across Same-Scene Viewpoint Cuts` | 已确认 | 标题锁定任务边界；不暗示一般 online multi-person reconstruction 或 dense scene reconstruction |
| D17 | Abstract 的结果叙事 | 内部主证据为核心 / 以外部 baseline 覆盖率为核心 | 以严格内部证据为核心，按历届 ICLR 的问题--洞见--方法--证据结构重写；不以 TRACE/PromptHMR coverage 作为 headline | 已确认 | EgoBody 是最强证据；EgoHumans 简洁说明迁移结果；Harmony4D 如实说明 mixed result；精确数字仅在最终证据确认后按需要择一加入 |
| D18 | Related Work 与不可复现相关工作的比较 | 属性/协议比较 / 直接数值比较 / 弱化相关工作 | 新增任务属性与协议对比表；不进行直接数值比较 | 已确认 | 明确承认 HumanMM/Multi-THuMBS 的任务重叠，突出 causal state transition、无未来帧与无离线全局优化的差异 |
| D19 | 9 页主文的证据取舍 | 紧凑主证据 / 完整结果全部放主文 | 主文保留任务图、严格三数据集内部表、EgoBody 外部公开 baseline 表、核心消融、定性图和精简 runtime/统计；完整 ledger、分层和辅助实验入 supplement | 已确认 | 以 ICLR 9 页主文为硬约束，避免以不可读长表或过小字号堆砌结果 |
| D20 | 正式术语与工程残留 | 全面学术化重命名 / 保留现有 transaction、locked 等词 | 全面学术化重命名，清除工程/审计式表达与历史模块残留 | 已确认 | 正式方法仅保留 clean reset、read-only shadow gauge、prediction-only association、shared camera--human translation；删除 root/filter/ray 等历史描述 |
| D21 | 方法可复现细节的深度 | Supplement 给出完整定义 / 维持当前高层描述 | 主文保留核心公式与因果约束；Supplement 给出 detector、assignment、坐标变换、fallback、多 cut composition 的可实现定义及 method-to-code map | 已确认 | 减少不透明 wrapper 印象；匿名代码发布时可以逐项对应论文定义 |
| D22 | 可 replay 的数据与实验链路 | 恢复必要数据并重建 replay 包 / 仅保留当前聚合 artifacts | 正式投稿前恢复必要数据、per-case prediction 与 replay 包；作者后续重新上传此前删除的数据 | 已确认（依赖数据恢复） | 建立 manifests、configs、checkpoint/evaluator 清单、表格生成脚本与 replay report；恢复后补齐已确认的延后实验 |
| D23 | 相机指标的解释深度 | 补 Sim(3)/SE(3) 分解与说明 / 仅保留 ATE-Sim3 | 暂按补 Sim(3)/SE(3) 分解、translation/rotation/scale 说明执行；恢复数据并核验结果后，允许改为主文仅保留 ATE-Sim3 | 暂定 | 先预留补充材料完整表与主文解释位置；不在没有完整核验前夸大相机改善 |
| D24 | AI Use Statement 的披露范围 | 如实完整披露 / 保留现有窄披露 | 如实披露语言编辑、文献整理、实验规划、结果分析、论文结构建议和代码支架；作者独立核验全部科学内容 | 已确认 | 明确 AI 未生成或伪造实验数字、图像、数据标注或引用，作者承担最终责任 |
| D25 | 延后实验的论文占位 | PDF 显式 TODO-EXPERIMENT / 仅源文件注释 | 当前工作版 PDF 显式显示 `TODO-EXPERIMENT`、待填表格与图位；最终投稿前统一清除 | 已确认 | 最终打包启用 fail-closed 检查，禁止 PDF、源文件与 release archive 遗留 TODO；当前仅作为版本迭代辅助 |
| D26 | 匿名发布与最终打包 | 论文+匿名代码可复现包 / 仅 Overleaf 论文包 | 当前不处理代码包；论文与实验正式定稿后再单独构建匿名可复现包 | 已确认（延后） | 当前只维护论文工程；最终阶段再执行环境、脚本、checkpoint 说明与匿名泄露扫描 |
| D27 | Harmony4D 的 88/100 分母呈现 | 主文统一 88-case 内部子集 / 恢复后重评 100 cases / 保留现有混合表述 | 与 EgoHumans 一致：内部比较统一使用共同可评的 88-case 交集；不作为特殊缺陷强调 | 已确认 | 主表仅简洁标注 \(N=88\)，不反复突出 88/100；完整 manifest 与可用性细节留在补充协议。数据恢复后可选择重建 evaluator 升级为全量结果 |
| D28 | Oracle controls 的呈现 | 保留在消融 / 删除 oracle rows / 混入 external baseline | 保留在组件消融，明确标注 `oracle boundary`；不进入 external baseline 表 | 已确认 | 用于分解 clean reset、coarse gauge 与 shared translation 的作用；不是可部署比较方法 |
| D29 | 论文的核心创新叙事 | causal state/evidence separation 为核心 / post-hoc output alignment 为核心 | 以 causal state/evidence separation 为核心；粗对齐与精对齐是该因果设计下的两阶段机制 | 已确认 | Introduction、Method 和 Conclusion 应证明历史只读证据与 clean future recurrence 的分离，而非把 BRIDGE3R 写成普通后处理 |
| D30 | 正式方法是否纳入训练的 correction-token 粗对齐分支 | 纳入并审计结果绑定 / 维持纯 array-level wrapper 叙事 | 纳入正式 BRIDGE3R：训练的 event-conditioned correction-token branch 是粗对齐核心，随后以显式 association/shared translation 精对齐，并由 clean reset 独占未来 recurrence | 已确认 | 审计 EgoBody、EgoHumans、Harmony4D 都绑定同一冻结 checkpoint 与 causal detector event route；论文不得使用历史 `oracle` 命名或把方法写成纯 array-level wrapper |
| D31 | correction-token 粗对齐的消融 | 补模块消融 / 仅叙述其作用 | 补完整两阶段模块消融：Strict streaming → Clean reset → 无 correction-token shadow → correction-token coarse gauge → + identity association → + shared translation（Full BRIDGE3R） | 已确认（延后） | 主文呈现连续模块增益；camera-only/human-only、oracle boundary 与 token-route 细分放补充材料，并先留 `TODO-EXPERIMENT` |
| D32 | 训练式粗对齐模块的训练披露 | 完整训练/数据审计 / 作为黑箱 backbone 不展开 | 完整披露训练数据、cut 构造、监督、冻结/训练参数、checkpoint 与 test-overlap audit；原始 V8/V9/V14 记录已定位，作为可复核来源 | 已确认 | 将“未在三个 benchmark 上训练或调参”与“预训练后冻结的 coarse module”明确区分，避免不准确的无可训练参数表述 |
| D33 | EgoHumans 的 cut 触发实现 | 以已知切点运行 / 消费 causal detector 输出且重跑 / 消费 detector 输出并复用等价冻结缓存 | 使用 causal GRU 的 first-positive 作为 runtime trigger；现有 116 个 frozen case 已验证逐例等于 evaluator boundary，故只更新代码并 fail-closed 校验，不重跑数值 | 已确认 | 文中可将该协议表述为 RGB-only causal detector trigger；若将来出现不相等的 detector proposal，runner 必须重新执行 RGB backbone，不能复用已有缓存 |

## D32 evidence trace (2026-08-25)

The historical records support a reproducible, non-black-box description of the
learned coarse module.  They will be distilled into the supplement rather than
copied verbatim into the paper.

- **V8 origin and parameter policy.** `Movie3R/docs/movie3r/archive_v8/START_HERE.md`
  records the initial decoder-in correction prompt and the use of raw calibrated
  camera targets. `v8_7_head_lora_finetune_plan.md` specifies that the image
  encoder, decoder, scene head, and base pose/human heads are frozen; the
  correction branch and low-rank pose/human-head updates are optimized.
- **V9 frozen learned module.** `Movie3R/versions/v9/manifest.json` binds the
  V9 model to original HUMAN3R initialization, its resolved configuration, git
  commit, and a SHA-256-addressed final checkpoint. `v9_implicit_human_pose_token.md`
  and `MODEL_ARCHITECTURE_DETAILS.md` define the relation correction tokens and
  latent camera/human corrections. The V9 formal plan documents the supervised
  camera, human-translation, residual, and no-op objectives, together with the
  held-out source protocol.
- **Publication checkpoint.** The actual publication checkpoint is not the
  historical V9 checkpoint: it is the V14.1 P0 multi-person checkpoint
  (`de2430ed...828265`), initialized from the formal V9 model. Its exact
  five-source, 480-event/epoch, six-epoch mix is fixed in
  `config/train_v14_1_cut_first_cross_source_multihuman_p0.yaml`. The training
  contract in `V14_1_ONE_SHOT_SHADOW_CORRECTION_TRAINING.md` specifies the
  causal `A(t-1), A(t), B(t)` event, event-only supervision, frozen base
  components, and the trainable correction/LoRA components.
- **Required before submission.** Verify the exact trainable-parameter count
  from the publication checkpoint/config and perform a case-level overlap audit
  against EgoBody, EgoHumans, and Harmony4D. Do not claim zero benchmark
  overlap until the latter audit has completed. Also verify that each formal
  runner supplies event routing from the causal detector rather than a GT
  shot label.
