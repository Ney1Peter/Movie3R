# Shot3R Method 重构草案（2026-09-05）

## 1. 重构目标

本草案只重写论文方法叙事、符号、术语、方法图和主文图注，不改变模型执行流程、训练配置、实验协议或已有结果。当前锁定版本 `v031_20260902_read_reset_register_and_baseline_plan` 保持不变；本文件和配套方法图均为独立候选稿。

方法部分采用以下论证顺序：

1. 定义包含镜头切换的单目流式多人体重建任务；
2. 介绍有状态在线重建的基础表示；
3. 从两类信息具有不同生命周期出发，提出对齐与状态传播解耦；
4. 通过人物关联和共享空间变换维持相机—人体一致性；
5. 说明镜头间对齐如何利用同步观测学习。

## 2. 章节结构

### 3. Method

- 3.1 Overview
- 3.2 Stateful Online Human Reconstruction
- 3.3 Decoupling Alignment from State Propagation
- 3.4 Camera--Human Consistency across Shots
- 3.5 Learning Alignment at Shot Transitions

对应中文：

- 3.1 方法概述
- 3.2 有状态在线人体重建
- 3.3 对齐与状态传播的解耦
- 3.4 镜头间的相机—人体一致性
- 3.5 镜头间对齐的学习

## 3. 术语统一

| 旧术语 | 统一后的表述 | 修改原因 |
|---|---|---|
| `BRIDGE3R` | `Shot3R` | 与定稿标题和任务定位一致 |
| `Read--Reset--Register` | 不再作为方法标题 | 避免以工程步骤定义方法 |
| `gauge` / `coarse gauge` | `world-frame alignment` / `cross-shot transform` | 直接说明是世界坐标系对齐 |
| `read-only shadow evaluation` | `history-conditioned alignment path` | 强调它提取对齐信息的作用 |
| `clean post-cut evaluation` | `reinitialized reconstruction path` | 强调状态从初始值重新建立 |
| `clean state` / `future-state owner` | `propagated recurrent state` | 避免拟人化和内部工程称呼 |
| `causal boundary input` | `online shot transition` | 用可直接理解的输入条件替代抽象术语 |
| `causal contract` / `causal invariants` | `online information-flow constraint` | 保留不访问未来帧的含义，删除审计式表述 |
| `prediction-only association` | `person association from model predictions` | 在首次出现时直接解释信息来源 |
| `shared translation` | `person-guided shared translation` | 说明该变换由多人对应关系共同支持 |

## 4. 符号统一

| 符号 | 含义 |
|---|---|
| $I_t$ | 时刻 $t$ 的单目 RGB 图像 |
| $S_t$ | 处理至时刻 $t$ 的递归状态 |
| $C_t$ | 相机到世界坐标系的变换 |
| $\mathcal H_t$ | 当前帧的多人体重建结果 |
| $F_t,p_0,H_t$ | 图像 token、初始相机 token 和人体提示 |
| $F'_t,p_t,H'_t$ | 经过递归解码器更新的图像、相机和人体表示 |
| $(\widetilde C_b,\widetilde S_b)$ | 历史条件路径在边界 $b$ 的相机预测和临时状态 |
| $(C_t^r,\mathcal H_t^r,S_t^r)$ | 从初始状态建立的新镜头重建结果和递归状态 |
| $T_b^{\mathrm{align}}$ | 由两条路径的边界相机预测得到的镜头间对齐 |
| $\mathcal M_b$ | 镜头边界处的有效人物匹配对集合 |
| $\pi_b$ | 将新镜头匿名槽位映射到持续人物身份的对应关系 |
| $\Delta_b$ | 由匹配人物共同估计的平移修正 |
| $T_b=T_b^\Delta T_b^{\mathrm{align}}$ | 应用于新镜头相机与人体的最终共享变换 |

不再使用同一个 $\mathcal P$ 同时表示匹配集合和身份置换；不再用缺少边界下标的 $T_0$ 和 $\Delta$ 表示所有镜头切换。

## 5. Method 中文整合稿

### 3.1 方法概述

给定一段包含镜头切换的单目视频流 $\{I_t\}_{t=1}^{T}$，Shot3R 按照时间顺序处理每一帧，在时刻 $t$ 预测世界坐标系下的相机姿态 $C_t$ 和多个人体重建结果 $\mathcal H_t$，同时维护递归状态 $S_t$ 以积累历史几何与运动信息。在连续镜头内，这一状态为相机与人体提供稳定的时空参照；然而，当镜头切换引起视角、场景内容和可见人物的突变时，继续传播原有状态会使旧镜头中的视觉信息干扰新镜头，而直接重置状态又会丢失不同镜头之间的空间联系。

Shot3R 的核心观察是：镜头切换时真正需要继承的是此前重建所建立的世界坐标参照，而不是包含旧镜头视觉内容的完整递归状态。基于这一观察，我们将镜头间对齐与递归状态传播解耦。在检测到镜头切换后，新镜头的第一帧分别在历史状态和初始状态下进行重建：前者提供已有世界坐标参照下的边界预测，仅用于估计镜头之间的空间变换；后者产生新镜头的重建结果，并提供后续帧唯一使用的递归状态。随后，模型根据对齐后的人体预测建立跨镜头人物对应关系，并对相机与人体施加一致的空间变换。由此，Shot3R 能够在不访问未来帧、也不进行逐视频全局优化的情况下，保持相机姿态、人体运动和人物身份的跨镜头一致性。

### 3.2 有状态在线人体重建

Shot3R 采用有状态在线重建作为基础表示。在时刻 $t$，输入图像 $I_t$ 被编码为图像 token $F_t$，当前帧中检测到的每个人体由对应的人体提示 $H_t$ 表示。图像 token、初始相机 token $p_0$ 和人体提示共同与上一时刻的递归状态 $S_{t-1}$ 交互：

$$
[F'_t,p_t,H'_t],S_t
=
D_\theta([F_t,p_0,H_t],S_{t-1}),
$$

其中，$D_\theta$ 表示递归解码器。更新后的图像、相机和人体表示分别用于预测场景几何 $X_t$、相机到世界坐标系的变换 $C_t$，以及当前帧中所有人物的 SMPL-X 参数。由此得到

$$
\mathcal H_t=\{(J_t^i,V_t^i,q_t^i)\}_{i=1}^{N_t},
$$

其中，$J_t^i$、$V_t^i$ 和 $q_t^i$ 分别表示三维关节、人体网格顶点和模型分配的匿名人物槽位。

记 $y_t=(C_t,\mathcal H_t)$，并将从图像与上一状态到当前输出和新状态的完整映射简记为

$$
(y_t,S_t)=f_\theta(I_t,S_{t-1}).
$$

具体实现采用 Human3R 作为上述在线重建器。Human3R 将相机、场景和多个人体表示统一到同一个递归状态中，使模型能够从历史状态中恢复时间连续的几何与人体信息。在单个连续镜头内，Shot3R 保留这一逐帧递归过程。该过程同时隐含一个前提：当前图像 $I_t$ 应当是状态 $S_{t-1}$ 所描述视频流的连续观测。镜头切换破坏了这一前提，使世界坐标参照和与旧镜头相关的视觉记忆不能再以相同方式传播。

### 3.3 对齐与状态传播的解耦

设新镜头从第 $b$ 帧开始。Shot3R 使用同一帧 $I_b$ 构造两条具有不同信息生命周期的路径：

$$
(\widetilde y_b,\widetilde S_b)
=f_{\theta,\phi}(I_b,S_{b-1}),
\qquad
(y_b^r,S_b^r)
=f_\theta(I_b,S_{\varnothing}),
$$

其中，第一条路径由历史状态提供条件，并引入可学习的镜头间对齐表示；第二条路径从初始状态 $S_{\varnothing}$ 独立重建新镜头。历史条件路径产生临时状态 $\widetilde S_b$，但该状态不会传播。新镜头中的后续帧只按照

$$
(y_t^r,S_t^r)=f_\theta(I_t,S_{t-1}^r),\qquad t>b
$$

继续更新。因此，$S_{b-1}$ 可以参与边界处的显式空间对齐，却不存在进入 $S_t^r$ 的递归路径。这一信息流约束是 Shot3R 解耦对齐与状态传播的核心。

历史条件路径包含一组仅在镜头边界激活的对齐 token，用于汇集当前图像—人体语义、相机姿态变化以及递归记忆中的时间上下文。将它们记为

$$
Z_b^{\mathrm{align}}
=\Phi_\phi(u_b,\bar u_b,p_b,p_{b-1},p_b-p_{b-1},m_b,h_{b-1}),
$$

其中，$u_b$ 和 $\bar u_b$ 概括当前观测与历史状态，$p_b$ 和 $p_{b-1}$ 表示当前与先前的相机特征，$m_b$ 是递归记忆上下文，$h_{b-1}$ 汇集此前的校正和可靠性信息。对齐 token 与当前图像、相机和人体表示共同经过递归解码器，并通过相机残差得到历史坐标参照下的边界相机预测 $\widetilde C_b$。三类 token 的具体构造保留在补充材料中。

从初始状态建立的路径对同一帧产生相机预测 $C_b^r$。两条路径观察相同的 RGB 图像，但使用不同的历史条件，因此它们的相对相机变换给出新镜头到已有世界坐标系的对齐：

$$
T_b^{\mathrm{align}}
=\widetilde C_b(C_b^r)^{-1}.
$$

得到 $T_b^{\mathrm{align}}$ 后，历史条件路径的状态、人体和场景预测均被丢弃；只有重置路径产生的 $S_b^r$ 继续处理新镜头。后续出现新的镜头切换时，同一转换独立重复。

### 3.4 镜头间的相机—人体一致性

相机变换 $T_b^{\mathrm{align}}$ 将重置路径的预测放入此前建立的世界坐标系：

$$
\bar C_t=T_b^{\mathrm{align}}C_t^r,
\qquad
\bar J_t^i=T_b^{\mathrm{align}}J_t^{r,i},
\qquad
\bar V_t^i=T_b^{\mathrm{align}}V_t^{r,i}.
$$

相机对齐本身不能确定镜头切换前后的人物对应关系，而分别移动相机和人体又会破坏二者的相对几何。Shot3R 因此使用模型预测建立一对一人物对应，并由所有匹配人物共同估计一个作用于相机和人体的共享变换。

对于旧镜头中的人物 $i$ 和新镜头中的人物 $j$，匹配代价由骨盆位置、躯干朝向和去除根节点平移后的关节结构组成：

$$
d_{ij}
=
\widehat d_{\mathrm{root}}(i,j)
+\widehat d_{\mathrm{torso}}(i,j)
+\widehat d_{\mathrm{joint}}(i,j).
$$

三个候选代价矩阵分别除以其有限元素的中位数，以减小量纲差异。匈牙利算法给出匹配集合 $\mathcal M_b$ 和相应的身份映射 $\pi_b$。匹配仅使用模型预测，不依赖真实人物身份、相机标定或未来帧。

对于匹配对 $(i,j)\in\mathcal M_b$，令 $r_{b-1}^i$ 和 $\bar r_b^j$ 分别表示镜头切换前后的人体骨盆位置。所有有效匹配共同确定平移修正

$$
\Delta_b
=\lambda\underset{(i,j)\in\mathcal M_b}{\operatorname{median}}
\left(r_{b-1}^i-\bar r_b^j\right),
\qquad \lambda=0.5.
$$

中位数聚合减小单个人体预测误差的影响；固定的 $\lambda$ 在保留相机对齐和采用完整人体位移之间进行收缩。令 $T_b^\Delta=[I_3\mid\Delta_b]$，最终共享变换为 $T_b=T_b^\Delta T_b^{\mathrm{align}}$。对于从 $b$ 开始、到下一次镜头切换之前的所有帧，

$$
C'_t=T_bC_t^r,
\quad
J_t^{\prime i}=T_bJ_t^{r,i},
\quad
V_t^{\prime i}=T_bV_t^{r,i},
\quad
q_t^{\prime i}=\pi_b(q_t^{r,i}),
\quad
S'_t=S_t^r.
$$

相机和人体共享同一个世界空间变换，从而保持相对几何；递归状态不接受空间变换，也不从旧镜头复制。若没有可靠人物匹配，模型保留 $T_b^{\mathrm{align}}$，并为新镜头人物建立新的匿名身份。

### 3.5 镜头间对齐的学习

真实镜头切换通常同时包含视角变化、人体运动和遮挡变化。为了单独学习镜头之间的空间关系，我们使用同步相机 $A$ 和 $B$ 构造训练序列

$$
A(t-1),\qquad A(t),\qquad B(t),
$$

并将 $B(t)$ 作为新镜头的第一帧。由于 $A(t)$ 和 $B(t)$ 观察同一时刻，二者的差异主要来自相机视角，而不是人体运动。训练只在 $B(t)$ 上监督边界对齐，不使用其后的图像。

训练目标按作用归纳为

$$
\mathcal L_{\mathrm{boundary}}
=\mathcal L_{\mathrm{cam}}
+\lambda_h\mathcal L_{\mathrm{human}}
+\lambda_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+\lambda_{\mathrm{keep}}\mathcal L_{\mathrm{keep}}.
$$

$\mathcal L_{\mathrm{cam}}$ 使用平移损失和旋转测地损失监督边界相机；$\mathcal L_{\mathrm{human}}$ 使用可见人物的根节点位置提供辅助空间监督；$\mathcal L_{\mathrm{res}}$ 约束潜在校正幅度；$\mathcal L_{\mathrm{keep}}$ 保持原有点图以及人体姿态、形状和表情表示的稳定。辅助人体预测只作为训练信号，不进入最终重建结果。精确损失项、权重和监督范围在补充材料中给出。

具体实现冻结预训练在线重建器的图像编码器、递归解码器、点图预测头、基础相机和人体预测头以及 Multi-HMR 主干，只学习镜头边界处的对齐 token、相机与辅助人体残差，以及相机和人体预测头的低秩更新。所有主要数据集使用同一个模型，不进行逐数据集或逐视频调整。推理时，镜头边界由仅使用当前和历史 RGB 的在线检测器提供；没有检测到切换时，模型保持原有递归过程。

## 6. 公式整理原则

主文保留六类公式：基础递归表示、边界双路径、后续状态传播、相机对齐、人物匹配与共享变换、训练目标。以下内容移至补充材料：

- 三个对齐 token 的逐项构造；
- 八个实际损失项及其精确权重；
- 参数数量与 LoRA rank；
- 多次镜头切换的展开式；
- 检测器回退和无匹配回退的完整伪代码；
- 训练—测试数据交集审计。

主文不再使用信息集合 $\mathcal F_t$、可测性和 boxed causal contract。在线约束直接由边界双路径公式体现：历史状态只进入 $\widetilde y_b$，未来状态只由 $S_b^r$ 递归产生。

## 7. 方法图重构原则

新方法图使用四个视觉区域，但不再画成编号式工程步骤：

1. 单目视频流与镜头切换；
2. 临时对齐路径和持续重建路径；
3. 世界坐标系下的相机—人体一致性；
4. 新镜头的流式输出。

颜色仅表示信息生命周期：

- 橙色虚线：只在边界使用的临时对齐信息；
- 绿色实线：新镜头中持续传播的递归状态；
- 蓝色实线：相机与人体的显式输出变换；
- 紫色：人物对应关系。

图内不再出现 `BRIDGE3R`、`gauge`、`read-only shadow`、`clean reset`、`causal invariants`、参数占比和长句解释。

图内主要英文标签及其中文含义如下：

| 英文标签 | 中文含义 |
|---|---|
| `Monocular stream` | 单目视频流 |
| `Alignment--state decoupling` | 对齐与状态解耦 |
| `World-frame consistency` | 世界坐标系一致性 |
| `Streaming output` | 流式输出 |
| `Alignment path / temporary` | 对齐路径／临时使用 |
| `Reconstruction path / propagated` | 重建路径／持续传播 |
| `Camera alignment` | 相机对齐 |
| `Person correspondence` | 人物对应关系 |
| `Shared world transform` | 共享世界空间变换 |
| `temporary alignment information` | 临时对齐信息 |
| `propagated recurrent state` | 持续传播的递归状态 |
| `camera--human outputs` | 相机—人体输出 |

## 8. 主文图片短图注候选

### Figure 1：定性示例

英文候选：

> An EgoHumans shot transition with a $176.7^\circ$ camera rotation. Shot3R preserves coherent camera placement and person identities under the large viewpoint change; TRACE and PromptHMR provide mesh-only outputs through their released interfaces.

对应中文：

> EgoHumans 上具有 $176.7^\circ$ 相机旋转的镜头切换示例。Shot3R 在大幅视角变化下保持了相机位置与人物身份的一致性；TRACE 和 PromptHMR 按其公开接口仅提供人体网格结果。

### Figure 2：方法概览

英文候选：

> Overview of Shot3R. A history-conditioned path estimates cross-shot alignment, while a reinitialized path alone propagates recurrent state. Prediction-based person association and a shared camera--human transform place new-shot outputs in the existing world frame.

对应中文：

> Shot3R 方法概览。历史条件路径估计镜头间对齐，重新初始化的路径则独立传播递归状态。基于模型预测的人物关联与共享相机—人体变换将新镜头输出统一到已有世界坐标系。

### Figure 3：视角分层结果

英文候选：

> Reconstruction across EgoHumans viewpoint-change strata. Curves show case means with 95\% capture-cluster bootstrap intervals; the largest gains occur in the two widest camera-rotation strata.

对应中文：

> EgoHumans 不同视角变化区间下的重建结果。曲线表示样本均值和以采集序列为聚类单位的 $95\%$ bootstrap 区间；两个相机旋转范围最大的区间取得了最明显的改进。

定性样例的筛选规则、方法输出范围差异和统计细节应在正文或实验设置中说明，不再全部塞入图注。
