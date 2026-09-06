# EgoBody 三视角、两次跳变原始帧

- 原始数据：`EgoBody.zip`
- Recording：`recording_20210907_S03_S04_01`
- 场景：`seminar_g110`
- 原始帧率：30 FPS
- 人数：2
- 相机顺序：`kinect_12 -> kinect_13 -> kinect_11`
- 两次相邻视角跳变：49.70°、91.48°

## 选择依据

正式 EgoBody 测试结果中，单次大角度切镜案例
`egobody_recording_20210907_S03_S04_01_extreme_kinect_13_kinect_11_b01801`
是 BRIDGE3R 在完整身份关联（IDF1=1、Coverage=1）的 extreme 案例中
W-MPJPE 最低的一例。其结果为 W-MPJPE 143.03 mm、WA-MPJPE 124.40 mm、
PA-MPJPE 61.07 mm、ATE-Sim3 0.00699 m。

九帧素材保留这个表现最好的正式切镜 `kinect_13 -> kinect_11`，并在其前面
加入同一 recording 的 `kinect_12`，从而形成三个真实标定视角、两次相机跳变。
`kinect_13` 的最后一帧 01800 与 `kinect_11` 的第一帧 01801 正好对应正式
实验的相邻时间边界。

## 帧序列

| 顺序 | 原始帧 | 相机 | 与前一相机的旋转跨度 | 文件 |
|---:|---:|---|---:|---|
| 1 | 01750 | kinect_12 | - | `01_view1_kinect12_frame01750.jpg` |
| 2 | 01760 | kinect_12 | - | `02_view1_kinect12_frame01760.jpg` |
| 3 | 01770 | kinect_12 | - | `03_view1_kinect12_frame01770.jpg` |
| 4 | 01780 | kinect_13 | 49.70° | `04_view2_kinect13_frame01780.jpg` |
| 5 | 01790 | kinect_13 | - | `05_view2_kinect13_frame01790.jpg` |
| 6 | 01800 | kinect_13 | - | `06_view2_kinect13_frame01800.jpg` |
| 7 | 01801 | kinect_11 | 91.48° | `07_view3_kinect11_frame01801.jpg` |
| 8 | 01811 | kinect_11 | - | `08_view3_kinect11_frame01811.jpg` |
| 9 | 01821 | kinect_11 | - | `09_view3_kinect11_frame01821.jpg` |

这些帧按原始采集时间严格递增，构成单目多镜头时间流，而不是同步多视角输入。
旋转跨度由官方外参计算，只用于素材说明和评估，不是模型输入。

需要注意：上述“效果最好”仅指已经完成并量化的 `kinect_13 -> kinect_11`
单次切镜。加入 `kinect_12` 后的九帧序列是两次切镜展示素材，不把原单切镜指标
冒充为新三视角序列的量化结果。
