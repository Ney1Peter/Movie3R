# V14 自适应联合边界修正冻结总结

日期：2026-08-05

## 1. 最终采用的流程

当前 baseline 仍然是冻结的 B0 + BRTC-LC + C1-EMA25。新模块只在检测到候选 shot 边界时工作，不改变连续 shot 内原有输出。

```text
输入 RGB 流
  ↓
因果 RGB shot detector（只看 I[t-1], I[t]）
  ↓  给出候选边界
B0+BRTC+C1 正常重建
  ↓
人体几何可信度 gate
  ├─ 不可信/残差小：exact baseline fallback
  └─ 可信：执行一次 post→pre 联合相机-人体修正
                  ↓
        后续 post 帧持续复用该修正
```

候选边界处使用最后一个 pre 帧和第一个 post 帧的预测 SMPL-X 网格：

1. 枚举多人排列，用一个共享 Kabsch SE(3) 估计 post 到 pre 的人体方向残差，同时解决匿名人物匹配。
2. 检查旋转幅度、顶点 RMS、身体尺度归一化 RMS 和多人排列 margin。
3. 通过 gate 后，使用 B0 人体残差确定旋转；同 checkpoint 的 raw Human3R shadow 分支提供另一条 root ray，用于联合求解相机平移。
4. 人体围绕当前 BRTC root 旋转，保持 BRTC 已经得到的根位置；相机使用同一旋转，并由 B0/raw root ray 平均值求平移。
5. 世界背景点云跟随实际相机世界变换；pre 帧完全不动。修正只在 boundary 之后因果保持。

当没有 raw shadow 分支时，模块退化为纯共享 SE(3) 版本：相机、背景和人体使用同一个世界变换。这一版本用于离线 payload 审计；demo 的 adaptive 模式默认保留 shadow 分支，因此优先使用数值更稳定的联合相机-人体版本。

## 2. 代码入口

- 核心几何和 gate：[src/dust3r/adaptive_joint.py](/data/wangzheng/iJCV-CODE/Movie3R/src/dust3r/adaptive_joint.py)
- demo 在线接入：[demo.py](/data/wangzheng/iJCV-CODE/Movie3R/demo.py)
- 后置 gate payload 工具：[versions/v14/adaptive_post_human_boundary.py](/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/adaptive_post_human_boundary.py)
- 前置 detector + gate 工具：[versions/v14/streaming_detector_joint_boundary.py](/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/streaming_detector_joint_boundary.py)
- 单元测试：[versions/v14/tests/test_adaptive_joint.py](/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/tests/test_adaptive_joint.py)

demo 新增参数：

```bash
--adaptive_joint_mode detector   # 推荐主线
--adaptive_joint_mode post       # 显式 cut_indices 的后置 gate 对照
--cut_indices 5                 # post 模式必需；detector 模式可省略
```

detector 模式会自动运行同 checkpoint 的 event-off shadow 分支，因此不需要额外预训练模型；只增加一次同模型推理。默认 gate：

```text
最小人体旋转：20°
最大顶点 RMS：0.20 m
最大尺度归一化 RMS：0.20
最小多人排列 margin：0.01 m
```

## 3. 实验结果

所有实验均为 CPU 后处理，运行时不读取 GT。GT 只在独立 evaluator 中用于结果检查。

### 3.1 单人 AvatarReX，低纹理，约 60°跳变，30 帧

数据：`avatarrex_t1836_c22070935_c22053912_pre5_post25`，5 帧 pre + 25 帧 post。

人体 gate 的 B0 残差：`66.27° / 0.0448 m RMS`，通过；RGB detector 预测边界 `[5]`，无额外误报。raw shadow root-ray 联合版本的 GT 检查为：

| 指标 | B0+BRTC+C1 | 自适应联合修正 |
|---|---:|---:|
| 首 post 相机平移误差 | 1.697 m | 0.011 m |
| 首 post 相机旋转误差 | 66.51° | 0.40° |
| 首 post root 误差 | 0.066 m | 0.066 m |
| 首 post MPVPE | 0.281 m | 0.100 m |
| 25 帧平均相机平移误差 | 1.703 m | 0.054 m |
| 25 帧平均相机旋转误差 | 66.56° | 0.44° |
| 25 帧平均 MPVPE | 0.247 m | 0.123 m |

### 3.2 多人高纹理，3 人，30 帧

测试了已有的四个序列：

```text
three_t1100_c1_c2_pre5_post25
three_t1100_c1_c4_pre5_post25
three_t1100_c2_c5_pre5_post25
three_t1100_c4_c5_pre5_post25
```

RGB detector 在四个序列均只预测第 5 帧为 cut。几何 gate 的结果全部为 baseline fallback：

| 序列 | 共享人体旋转残差 | 顶点 RMS | 结果 |
|---|---:|---:|---|
| c1→c2 | 4.03° | 0.117 m | 拒绝修正 |
| c1→c4 | 10.25° | 0.131 m | 拒绝修正 |
| c2→c5 | 8.16° | 0.083 m | 拒绝修正 |
| c4→c5 | 3.12° | 0.093 m | 拒绝修正 |

因此新模块不会破坏当前多人 B0+BRTC+C1 结果，也不会把多人中的正常姿态变化误认为需要大范围人体旋转。

### 3.3 单人约 180°案例 sanity check

AvatarReX `22053903→22139907` 的 4 帧 raw payload 上，人体共享残差为 `179.41° / 0.127 m RMS`，post gate 和 detector gate 都正确触发第 2 帧。这是极端大视角跳变的几何稳定性检查；该 payload 是原版 Human3R 两段输出，不作为 B0+BRTC+C1 的正式 30 帧主指标。

### 3.4 Detector 本身

已有 image-only detector 的历史四源训练/验证表中，使用 RGB/灰度/光流/ORB 全部图像特征的 logistic 版本约为：

```text
accuracy 0.9828，F1 0.9838，false-positive rate 0.0208
```

本文新 gate 的意义是：即使 detector 偶尔误报，只要人体几何残差小、匹配不可靠或尺度残差过大，就保持 baseline，不执行危险的世界变换。

## 4. 两种接入位置的比较与冻结决策

### 前置 detector 版本

优点是事件条件在模型 forward 前就已知，能够自然描述为“event-conditioned streaming reconstruction”，并能让 ShotToken/shot reset 与后续几何交易使用同一个事件信号，论文叙事更完整、更有方法感。

风险是 detector 存在少量误报。当前通过后置人体 gate 做第二道保险后，测试序列没有错误更新。

### 后置 gate 版本

优点是实现最稳，直接观察 B0 输出再决定是否修正；在已有显式 boundary 的离线评测中最容易复现。缺点是事件已经发生后才修正，论文中更像安全校正器，而不是完整的流式事件处理系统。

### 冻结结论

推荐把“前置 causal shot detector + 后置人体几何 gate + root-ray 联合相机人体更新”作为论文主线；后置 gate 作为安全 fallback 和 ablation。二者在当前单人/多人样例上给出相同的接受/拒绝决策，主线版本在多人上 exact fallback，在单人低纹理上执行修正。

## 5. 当前限制和下一步

1. detector 的 2.08% FPR 来自已有探针数据，尚未在完整 AvatarReX/多人全量序列上重新统计；下一步应做全量事件 precision/recall 和误报后的 3D 影响。
2. 当前 20°阈值是保守部署阈值，需在更多小角度 cut、人物自身快速转身和多人遮挡案例上扫描 ROC，并冻结 dev/held-out 分割。
3. raw shadow 分支会增加一次同 checkpoint forward；若最终论文强调低延迟，需要测试纯共享 SE(3) fallback 与 root-ray 分支的时间/精度折中。
4. 之后再进行大规模 Multi-THUMBS 指标和跨 AvatarReX、THuman、MVHuman 泛化实验；所有实验都应同时报告 detector 触发率、gate 接受率、相机误差、MPVPE、root/joint 误差、ID continuity、seam jump 和额外延迟。

