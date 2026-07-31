# V14 BRTC-LC v1 冻结基线

日期：2026-08-01

状态：冻结，作为后续所有精对齐实验必须超过的 baseline

冻结编号：`BRTC_LC_V1_20260801`

## 1. 冻结结论

BRTC-LC v1 是当前第一个同时通过新确认集 root、人体世界坐标和多人布局门槛的在线精对齐方法。
从本次冻结起，B0 不再参与方法搜索：它固定负责相机与场景的粗对齐；BRTC-LC 在 B0 的结果上，
只修改切镜后各人的整体三维位置。

完整链路为：

```text
Human3R 原生 pre 段 + hard-reset post 段
-> 学习得到的 B0 camera/scene Boundary
-> 冻结 B0 相机、场景与共同世界坐标系
-> root + torso + centred-joints 匿名 Hungarian 匹配
-> 最后一帧 pre 与第一帧 post 的五个核心关节射线
-> closest-ray 三角化，得到每个人沿 post pelvis ray 的 signed depth residual
-> ray gap / joint MAD / parallax 三项显式可靠性 gate
-> 同一 cut 内 accepted 位移的 group median
-> 用 pre 多人布局从固定 lambda 集合选择个体 residual 强度
-> accepted 人体只做刚性平移
-> rejected 或 unmatched 人体精确回退 B0
```

它严格不修改 B0 camera，也不引入 DA3 或任何新的预训练模型，不读取未来帧。它解决的是
`root/depth + multi-person layout`，不是内部关节姿态、体型或全局朝向的完整修复。

## 2. 冻结资产

### 2.1 B0 权重

持久化位置：

```text
checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth
```

校验信息：

```text
size    = 5,087,696,045 bytes
sha256  = 8379243216775adbc886d00e6f93b6492f7d8f1bd67adb4e8ad6fbdd84e47123
```

该文件从易失的 `/dev/shm/movie3r_b0_seed456_ep150_full_b192/checkpoint-best.pth` 复制，冻结时
逐字节比较一致。权重和训练日志位于 `/data` 下并被 gitignore；Git 提交冻结的是路径、hash、
代码、参数和结果清单，而不是把 4.8 GiB 权重写入 Git。

同目录还保留：

```text
forward_path_dual.json
forward_path_shadow.json
log.txt
metrics_epoch.jsonl
train_steps.jsonl
```

### 2.2 精对齐 runtime

```text
path    = versions/v14/b0_person_triangulation.py
sha256  = 98b839f4ae2ff130b0c6ecbc4e0e634ba626d2433f148bee3e55ac169aab3327
API     = refine_matched_people(...)
```

机器可读清单：

```text
versions/v14/frozen/BRTC_LC_V1_20260801.json
```

## 3. 输入、模块和输出契约

### 3.1 在线输入

切镜时只允许使用已经到达的内容：

1. 最后一帧 pre-cut 的 B0 对齐相机和人体；
2. 第一帧 post-cut 的 B0 对齐相机和人体；
3. 匿名关联器输出的 `(pre_index, post_index)` 一对一索引；
4. 每个人 pelvis、左右 hip、左右 shoulder 的世界坐标。

不允许使用 GT camera、GT identity、GT depth、GT root、未来 post 帧或离线全序列优化。

### 3.2 匿名人物关联

B0 先把 post 人体放进 pre 世界坐标。匹配器把 root 距离、torso 朝向和去中心化关节形状组合成
代价矩阵，再做 Hungarian 一对一分配。身份模块只产生数组索引，不把数据集身份标签传给
BRTC-LC。确认集的 `125/125` 匹配正确率来自这条匿名路径。

### 3.3 五关节射线三角化

对匹配后的每个人、每个核心关节，构造：

```text
pre ray  = pre camera centre  -> pre predicted joint
post ray = post camera centre -> post predicted joint
```

求两条空间射线的 closest points，中点作为该关节的显式三角化位置。再减去 post mesh 内
“关节相对 pelvis”的向量，得到 pelvis 候选。候选与当前 post pelvis 的差只投影到 post pelvis
观察射线上，因此最终不会任意横向拉动人。五个关节候选用 median 汇聚。

### 3.4 显式可靠性 gate

冻结参数如下：

| 参数 | 冻结值 |
|---|---:|
| joint ids | `0, 1, 2, 16, 17` |
| minimum valid joints | `1` |
| median ray gap | `<= 0.20 m` |
| residual MAD | `<= 0.40 m` |
| median parallax sine | `>= 0.025` |
| residual magnitude cap | `2.0 m` |
| lambda grid | `0, 0.25, 0.5, 0.75, 1` |

gate 完全由预测相机与预测人体计算。任何条件失败都输出零修正，而不是猜测一个修正。

### 3.5 多人布局共识

先取一个 cut 内所有 accepted individual shift 的逐坐标 median `g`，再对第 `i` 人使用：

```text
shift_i = g + lambda * (individual_shift_i - g)
```

`lambda` 只根据修正后 post 的 pairwise root vectors 与最后一帧 pre 预测布局的一致性，从固定
网格选择。它不查看 GT。若只有一人，布局目标退化为零，固定选择网格中的最小值；此时 group
shift 等于该人的 individual shift，修正仍保持完整。

### 3.6 输出

输出保证：

- camera 与 scene：和 B0 数值完全相同；
- accepted person：root、joints、vertices 加同一个三维刚性 shift；
- pose、shape、global orientation：完全不变；
- rejected/unmatched person：和 B0 bit-exact；
- debug：输出每人的 gap、MAD、parallax、individual shift、group shift、lambda 和接受状态。

该刚性修正应因果传播到当前 post shot 后续帧，直到下一个 cut 建立新的 Boundary；传播本身不
重新读取 pre 之后的未来信息。

## 4. 冻结确认结果

确认报告：

```text
output/v14/fine_alignment_research/b0_two_view_person_triangulation/confirm_three_offset1.json
sha256 = 0ef83e82274549457319054d2505b26b08f0fc4a3cfe4d550d1a2531915f7481
```

该确认在开发规则冻结后进行，使用 `MultiHuman three offset1` 的 42 个 cut、125 人，且走自动
匿名匹配。均值结果为：

| 指标 | B0 | B0+BRTC-LC v1 | 相对改善 |
|---|---:|---:|---:|
| Root | 0.3779 m | **0.2314 m** | 38.8% |
| World joint | 0.4117 m | **0.2745 m** | 33.3% |
| World vertex | 0.3891 m | **0.2525 m** | 35.1% |
| Pairwise distance | 0.1341 m | **0.0984 m** | 26.7% |
| Pairwise vector | 0.3297 m | **0.2588 m** | 21.5% |

安全性结果：

- coverage：`88.0%`；
- root improve rate：`67.2%`；
- root 恶化超过 5 cm：`7.2%`；
- accepted residual 符号正确率：`87.3%`；
- 自动关联准确率：`100%`；
- camera 最大数值改动：`0.0`。

冻结门槛是 root gain `>=8%`、layout-vector gain `>=5%`、harm `<=10%`、coverage `>=20%`、
camera change `<=1e-12`，本版本通过全部门槛。

## 5. 基线边界与后续比较规则

后续候选只有同时满足以下条件，才可声明优于 BRTC-LC v1：

1. 不修改 B0 camera/scene；
2. 严格在线，不读取未来帧；
3. 不额外引入预训练视觉模型；
4. rejected/unmatched 可精确回退 B0；
5. 先在开发集确定公式、阈值与参数，再在未用于调参的确认集运行；
6. 确认集 root、world joint、world vertex 和 pairwise layout 相对本冻结版本改善；
7. coverage 不能通过大幅少做案例伪造提升，harm 必须受控；
8. 在 EgoHumans 连续小样本上的 W/WA、Accel、ATE、IDs 不得出现不可接受退化。

`raw Human3R`、`B0 only` 与 `B0+BRTC-LC` 是三条不同结果。原版 Human3R 的 Multi-THuMBS
风格指标只能作为诊断下界，不能代表本冻结版本。真正的本地近似对榜必须把 B0 Boundary 和
BRTC-LC person shift 应用到同一批 EgoHumans 帧与身份后再计算。

## 6. 已知限制

- 刚性平移不会修复错误的内部 articulated pose、body shape 或 global orientation；
- 当前确认集是受控 MultiHuman，不是 Multi-THuMBS 官方 EgoHumans split；
- 人数变化、长时间遮挡、误检和 association dustbin 尚未完整解决；
- 第一张 post 图 parallax 太小或射线噪声过大时会回退 B0，因此 coverage 不是 100%；
- 当前权重体积很大，冻结只保证复现性，不代表已经完成部署压缩；
- Multi-THuMBS 未公开的精确 split、可见性与 aggregation 规则仍不能从主文完全复现。

因此本冻结版本的准确定位是：

> 已验证、可复现、严格在线的 root/depth 与多人布局精对齐 baseline；不是完整人体重建问题的最终答案。
