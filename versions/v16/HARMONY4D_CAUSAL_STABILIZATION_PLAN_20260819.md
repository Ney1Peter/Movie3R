# Movie3R-v16：Harmony4D 因果联合稳定专项计划

日期：2026-08-19

## 1. 目标与边界

v15 的 Harmony4D 正式结果、28 个 test case 和 M15 结论永久冻结。v16 不覆盖 v15，也不使用已读 test 指标选择参数。

v16 的目标是解释并缩小“demo 主观连续，但 W-MPJPE、Accel 和 seam-root 偏高”的差距，得到一个无需额外预训练模型、可逐帧在线部署、对 Harmony4D 明确有效的版本。

重点改进：

```text
W-MPJPE
WA-MPJPE
Accel
first-post root / seam-root
metric camera trajectory
```

保护指标：

```text
MPJPE / MPVPE
ATE-Sim3
IDF1 / IDs
coverage
camera-human relative geometry
```

## 2. 核心假设

Harmony4D 的 exo camera 在单个 shot 内是静态标定相机，但 Human3R 会预测非物理相机漂移。由于人体世界坐标由相机坐标和预测相机共同决定，虚假相机运动会同时造成：

1. 静止人体在世界坐标中移动；
2. 世界轨迹 W-MPJPE 增大；
3. 二阶差分 Accel 增大；
4. shot 边界虽主观看起来合理，fixed-world seam 仍偏大。

因此首先测试 **Causal Shot-Local Gauge Stabilization (CSGS)**：

```text
当前帧预测相机 C_t
        ↓ 因果 shot-local SE(3) 滤波
稳定相机 C'_t
        ↓ G_t = C'_t C_t^{-1}
将同一个 G_t 同时作用于当前帧相机、人体 joints 和 vertices
        ↓
保持 camera-human 相对几何严格不变，去除共同世界 gauge 漂移
```

在 cut 处再测试 **Human-Anchored Coupled Boundary Registration (HCBR)**：根据 boundary 前后已匹配人体的 pelvis/torso，估计一个共同 SE(3)，只作用于整个 post shot 的相机和所有人体。它不单独拉动某个人，因此不会破坏相机—人体相对位置。

最后只在必要时测试因果人体 root 滤波；该分支不得改变身体姿态和相机，并必须通过保护指标 gate。

## 3. 数据拆分

### 3.1 探索集

```text
Harmony4D train/01_hugging
4 cases：small / medium / large / extreme
```

只用于候选形式和参数网格。

### 3.2 验证集

```text
Harmony4D train/03_grappling2
Harmony4D train/15_mma4 capture 005
```

用于选择是否晋升，不允许针对单个 case 手调。

### 3.3 开发留出集

```text
Harmony4D train/15_mma4 capture 014
```

仅在候选冻结后读取新方法结果。

### 3.4 最终未见留出集

在提取和读取任何 v16 指标前预注册：

```text
primary: Harmony4D train/09_karate
selection: first coordinate-valid capture in frozen structural SHA order
cases: small / medium / large / extreme
frames: 75 pre + 75 post
```

如果整个 nested ZIP 无 coordinate-valid capture，记为 dataset-unavailable，并按相同规则使用 `train/02_grappling` 作为预注册备选；不能根据方法结果更换 capture 或 camera pair。

## 4. 跳过、回退与诚实报告

允许跳过的只有评测不可定义特例：

1. 官方标定或 GT 文件损坏；
2. 冻结 evaluator 无法建立 shared initial fit；
3. GPU/runtime cache 损坏且重跑仍失败。

所有跳过均报告分母和原因。

运行时“难例”不从表中删除，而是由可观测性 gate 回退到冻结 M15。gate 只能读取当前及历史预测，不读取 GT、未来帧或最终指标。

## 5. 候选消融

从冻结 M15/B0 geometry 出发，保持 Boundary-Permutation ID：

| ID | 候选 | 说明 |
|---|---|---|
| V16-0 | M15 | 冻结基线 |
| V16-1 | CSGS-freeze | 每个 shot 将共同 gauge 固定在首帧 |
| V16-2 | CSGS-EMA | 相机 SE(3) 因果 EMA |
| V16-3 | HCBR-T | boundary matched roots 的稳健共同平移 |
| V16-4 | HCBR-SE3 | matched pelvis/torso 的稳健共同 SE(3) |
| V16-5 | CSGS + HCBR | shot 内稳定与 boundary 联合修正 |
| V16-6 | V16-5 + root filter | 可选因果 root alpha-beta 滤波 |
| V16-7 | gated V16-5/6 | 不可观测时 exact M15 fallback |

参数只在探索集形成有限网格；验证后一次冻结。

## 6. 成功标准

### 6.1 探索晋升

相对 M15，4-case macro 至少满足：

```text
W-MPJPE       改善 >= 10%
Accel         改善 >= 15%
seam-root     改善 >= 15%
MPJPE/MPVPE   变差 <= 5%
coverage      不下降
```

### 6.2 验证晋升

在 `train03 + train15/capture005` 上：

1. W、Accel、seam-root 三项中至少两项改善；
2. 至少 60% cases 的 W 不变差；
3. 任一 case 不得出现相对 M15 超过 20% 的灾难性 W/MPJPE 恶化；
4. ATE-Sim3、IDF1 和 coverage 不得形成系统性退化。

### 6.3 最终完成

冻结候选在 `train09` 未见留出集相对 M15：

```text
W-MPJPE       改善 >= 10%
Accel         改善 >= 15%
seam-root     改善 >= 15%
MPJPE/MPVPE   变差 <= 5%
ATE-Sim3      绝对恶化 <= 0.005 m
IDF1          绝对下降 <= 0.01
coverage      完全不变
```

若没有候选达到最终标准，则以完整负结果、失败机理和下一条可证伪路线结束，不在已读 test 上继续调参。

## 7. 执行与产物

长任务统一运行于：

```text
tmux session: movie3r_h4d_v16
output root: output/v16_harmony4d
temporary data: /data/wangzheng/iJCV-CODE/data/Harmony4D_work_v16
```

最终必须包含：

1. 候选实现和单元测试；
2. train01 网格与诊断；
3. train03/train15 验证表；
4. train09 冻结留出结果和统计；
5. M15 与 v16 的标准 demo.py 可视化；
6. 失败/跳过清单；
7. 最终方法文档、可复现命令和 Git 提交。
