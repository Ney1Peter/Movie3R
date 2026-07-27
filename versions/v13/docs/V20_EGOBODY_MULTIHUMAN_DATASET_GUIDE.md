# V20 多人 Multi-Shot 调试数据指南

## EgoHumans `001_legoassemble` 与 MultiHuman 的数据特点、限制和推荐用途

> 命名勘误（2026-07-27）：`001_legoassemble` 实际属于 EgoHumans。本文文件名中的
> `EGOBODY` 和本地父目录 `data/EgoBody/` 是历史命名，为保持旧引用和实验路径可复现而保留。
> 独立的 EgoBody release 位于本地 `data/EgoHuman/`，详见
> `V13_EGOHUMANS_EGOBODY_DATASET_AUDIT_20260727.md`。

## 1. 文档目的

本文档面向需要继续设计 Movie3R/V20 多人 multi-shot 方法的研究者或 AI，系统说明以下两套本地数据：

```text
/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble
/data/wangzheng/iJCV-CODE/data/MultiHuman
```

目标不是将它们立即定义为正式 benchmark，而是判断：

- 哪套数据适合调试多人检测、tracking 和跨 shot Re-ID；
- 哪套数据适合验证多人物共同估计一个 Boundary；
- 哪套数据可以评价 camera、human、SMPL-X vertices 或 scene；
- 哪些 GT 只能用于评价和 Oracle，不能进入可部署 candidate；
- 两套数据如何互补，避免重复或错误使用。

当前 Movie3R 的多人目标是：

```text
Token / body feature answers WHO.
Explicit geometry answers WHERE.
All accepted humans agree on ONE shared shot Boundary.
```

即人物表示只负责身份关联，最终 camera、scene 和所有人仍然共用一个 shot-level 变换，不能为每个人单独估计世界坐标系。

---

## 2. 总体对比

| 属性 | EgoHumans `001_legoassemble` | MultiHuman 静态部分 | MultiHuman Real-World-Capture |
|---|---|---|---|
| 数据性质 | 真实同步多相机连续序列 | 静态扫描/拟合 mesh | 真实同步多相机连续视频 |
| 人数 | 固定 3 人 | 1/2/3 人 | `box` 2 人、`dance` 2 人、`three` 3 人 |
| 固定外部相机 | 8 路 | 无 | 6 路 |
| 第一人称相机 | 3 套 Aria | 无 | 无 |
| 连续时间 | 601 帧 | 无 | 264/645/1177 个有效标注帧 |
| 原始全画面 | 3840x2160 JPG | 无 | 2048x2048 MP4 |
| 相机模型 | 明确为 `OPENCV_FISHEYE` | 无 | `K/R/T` pinhole 形式，无畸变参数 |
| 稳定 GT identity | `aria01/02/03` | 文件夹编号 | `person0/1/2` |
| 2D bbox | 有 | 无 | 无现成 bbox，可由 mesh 投影生成 |
| 2D keypoints | 有，133 点 | 无 | 无 |
| 3D joints | 有，17 点及拟合版本 | 无显式 joints | 无现成 joints，可由 SMPL-X vertices 回归 |
| 人体模型 | **SMPL** | SMPL-X OBJ | **逐帧 SMPL-X OBJ** |
| 人体 vertices | 6890 | 10475 | 10475 |
| 完整 body parameters | 有 SMPL pose/betas/transl | 主要为最终 OBJ | 主要为最终 OBJ，没有完整参数向量 |
| 人物 mask | 没有独立 GT segmentation mask | 无 | `box/dance/three` 没有现成逐人 mask |
| GT scene depth/mesh | 无 | 无 | 无 |
| 最适合用途 | 检测、2D关联、鱼眼、视角缺人、GT-ID几何 | mesh/变换单元测试 | 动态多人交互、遮挡、SMPL-X vertex、真实 multi-cut |
| 是否已运行 Human3R 多人实验 | **是，已完成首轮 smoke test** | 否，不具备 RGB | **尚未运行，只完成数据和投影审计** |

最重要的分工是：

- **EgoHumans** 更适合验证多人 detection、2D身份标注、鱼眼输入、不同视角人数变化和相机标定。
- **MultiHuman Real-World-Capture** 更适合验证真实动态交互、多人遮挡、逐帧 SMPL-X vertex 和 recurrent multi-cut。
- **MultiHuman 静态部分** 只能做几何和 mesh 单元测试，不能直接测试 RGB Human3R。

---

# 第一部分：EgoHumans `001_legoassemble`

## 3. EgoHumans 数据概况

根目录：

```text
/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble
```

该序列是三个人共同组装积木的真实同步多相机场景，包含：

| 项目 | 数量/格式 |
|---|---|
| 同步时间帧 | 601，编号 `00001-00601` |
| 外部固定相机 | `cam01-cam08`，共 8 路 |
| Aria 相机 | `aria01-aria03`，共 3 套 |
| 稳定人物身份 | `aria01`、`aria02`、`aria03` |
| exo 原图 | 3840x2160 JPEG |
| 外部相机模型 | `OPENCV_FISHEYE` |
| 最终人体 GT | SMPL，6890 vertices，45 joints |

八个 exo 相机每个都有完整 601 帧：

```text
exo/cam01/images/00001.jpg ... 00601.jpg
exo/cam02/images/00001.jpg ... 00601.jpg
...
exo/cam08/images/00001.jpg ... 00601.jpg
```

相同编号表示同步时刻，因此可以直接构造：

```text
cam01/frame300 -> cam06/frame300
```

这种人体完全不动、只改变相机的严格 camera-cut 几何测试。

## 4. EgoHumans 目录说明

### 4.1 `exo/`

外部固定相机的完整第三人称 RGB 图像，是当前 Human3R 的实际输入来源。

```text
exo/
  cam01/images/
  ...
  cam08/images/
```

当前输入规则：

```text
3840x2160 -> 512x288
```

只缩放，不裁剪，不改变 16:9 宽高比。

### 4.2 `ego/`

三个人佩戴的 Aria 第一人称相机数据：

```text
ego/aria01/
ego/aria02/
ego/aria03/
```

每个 Aria 目录包含：

```text
calib/          每帧 Aria 标定
images/rgb/     第一人称 RGB
images/left/    左侧相机
images/right/   右侧相机
```

当前 V20 首轮测试没有使用 Aria 图像作为 Human3R 输入。`aria01/02/03` 主要作为稳定 GT identity 和世界坐标来源。

### 4.3 `processed_data/bboxes/`

每帧、每个相机、每个人的 bbox 标注。每条记录包含：

```text
bbox
human_name
human_id
color
```

用途：

- 将 Human3R detection 对应到 GT identity；
- 评价 detection recall；
- 构造跨 shot Re-ID ground truth；
- 处理头部或骨盆点因弯腰偏离检测中心的情况。

GT bbox 不能进入可部署 Re-ID candidate，只能用于 probe、Oracle 和评价。

### 4.4 `processed_data/poses2d/`

每个人在每个相机中的 133 个 2D keypoints，包含置信度和有效标记。

用途：

- 评价 Human3R 2D joints/mesh reprojection；
- 生成 torso visibility 和遮挡分组；
- 检查某个人是否适合作为 Boundary anchor；
- 调试 Keypoint R-CNN 或 V11.4 cue。

当前首轮身份分配使用 bbox 为主、骨盆点为 tie-breaker。原因是弯腰时 Human3R detection center 和 GT pelvis 可能相差 `50-130 px`，只用 pelvis threshold 会错误拒绝正确人物。

### 4.5 `processed_data/poses3d/`

每帧按 `aria01/02/03` 保存 17x4 的 3D joints。

相关中间目录：

```text
poses3d/
fit_poses3d/
refine_poses3d/
```

它们分别对应初始、拟合和进一步优化的 3D joint 中间结果。最终绝对人体评价优先结合最终 `smpl/` 和 camera calibration。

### 4.6 `processed_data/init_smpl/`

SMPL 优化的初始化结果，包含：

- betas；
- pose/rotmat；
- 初始平移和 global orientation；
- vertices 和 joints；
- bbox、focal length 和 best view。

这是中间量，不应当作最终 GT。

### 4.7 `processed_data/smpl/`

每帧三个人最终拟合的 SMPL。每个人包含：

```text
global_orient
transl
body_pose
betas
vertices
joints
epoch_loss
```

注意：这是 **SMPL，不是 SMPL-X**。

可直接比较：

- world root；
- pelvis 和公共身体 joints；
- global orientation；
- body shape/scale 趋势。

不能不经转换就直接比较：

- Human3R SMPL-X 的完整 10475 vertices；
- SMPL-X hands/face；
- SMPL 与 SMPL-X 的非公共关节。

如果需要 full vertex metric，必须建立 SMPL-to-SMPL-X correspondence 或只评价公共 surface/joints。

### 4.8 `colmap/workplace/`

关键文件：

```text
cameras.txt
images.txt
points3D.txt
colmap_from_aria_transforms.pkl
aria_from_colmap_transforms.pkl
temp.db
project.ini
```

其中：

- `cameras.txt`：11 个相机的 intrinsics 和 fisheye distortion；
- `images.txt`：图像和外参记录；
- `points3D.txt`：COLMAP 稀疏点；
- 两个 PKL：Aria 与 COLMAP world 的双向 Sim(3)。

最终 SMPL 位于 `aria01` metric world。转换到 exo COLMAP world 需要：

```text
colmap_from_aria_transforms.pkl["aria01"]
```

该 Sim(3) 的 scale 为：

```text
1.7142045310549123
```

这只能用于 GT evaluation/gauge conversion，不能进入可部署 Boundary candidate。

## 5. EgoHumans 相机和投影审计

相机标定明确使用：

```text
OPENCV_FISHEYE
```

当前 Human3R smoke test 直接读取了原始鱼眼 JPG，没有先去畸变。由于 exo 图像是 16:9，输入阶段只缩放到 512x288，没有裁剪。

对后续模块的影响：

| 模块 | 是否可直接使用原图 |
|---|---|
| Human3R 多人 detection/token smoke test | 可以 |
| GT-ID root/orientation 几何 probe | 可以，但边缘人物需谨慎 |
| 精确 2D reprojection | 应使用 fisheye-aware projection 或先去畸变 |
| Keypoint R-CNN shared scale | 建议统一去畸变 |
| DA3 metric depth | 建议统一去畸变并同步 intrinsics |
| mesh bbox/width/height | 必须使用正确鱼眼模型 |

GT SMPL vertices 经 `aria01 -> COLMAP -> fisheye camera` 投影后，frame 300 的平均 bbox IoU：

| Camera | Mean bbox IoU |
|---|---:|
| cam01 | 0.846 |
| cam02 | 0.765 |
| cam03 | 0.808 |
| cam04 | 0.465 |
| cam05 | 0.715 |
| cam06 | 0.822 |
| cam07 | 0.782 |
| cam08 | 0.695 |

这证明 GT body、Aria/COLMAP gauge 和 exo 标定总体一致。`cam04` 是明显困难视角，适合测试遮挡和 fallback，不适合单独作为最精确的 reprojection 样本。

## 6. EgoHumans 已完成的多人实验

这一部分是**已经运行过的结果**。

测试了三条三相机链，共 45 张图、6 个 camera cuts：

```text
cam01 296-300 -> cam06 300-304 -> cam07 304-308
cam02 176-180 -> cam05 180-184 -> cam08 184-188
cam03 416-420 -> cam04 420-424 -> cam01 424-428
```

Boundary 两侧重复同一个同步时间戳，以排除人体运动歧义。

关闭的模块：

```text
DA3
Keypoint R-CNN shared scale
V11.4 shared scale
VGGT
scene refinement
```

### 6.1 多人 detection

- 第一条链每帧检测到 3 人，45 个 detection-to-GT assignment 全部成功；
- 第二条链 `cam02` 前四帧只检测到 2 人，之后恢复 3 人；
- 第三条链 `cam04` 每帧只检测到 1 人。

这使该序列天然适合测试：

- 所有人可见；
- 某个人漏检；
- 遮挡严重只剩一人；
- multi-human -> single-human fallback。

### 6.2 Human3R 原生 track ID

原生 track ID 在 wide-view camera cut 后不稳定。例如：

```text
aria01: 1 -> 1 -> 1
aria02: 0 -> 2 -> 0
aria03: 2 -> 0 -> 3
```

因此：

- 输出数组顺序不能当作稳定身份；
- 原生 `smpl_id` 不能直接当作跨 shot identity；
- scene/camera reset 后需要独立的 external tentative identity bank。

### 6.3 跨 shot feature probe

在 6 个 cuts、14 个可评价 assignment 上，normalized L2 + Hungarian 的结果：

| Feature | Correct | Accuracy |
|---|---:|---:|
| refined human token `H'` | 4/14 | 28.6% |
| fused human prompt | 6/14 | 42.9% |
| Multi-HMR head token | 6/14 | 42.9% |
| CUT3R head token | 8/14 | 57.1% |
| SMPL beta | 10/14 | 71.4% |
| root-centered local pose | 13/14 | 92.9% |

`local pose` 的高准确率不能解释成真正 identity 能力，因为 cut 两侧是同一时刻，姿态几乎相同。它目前只能视为同步匹配 cue。

当前结论：raw `H'` 不足以直接部署为跨 shot Re-ID embedding。

### 6.4 GT-ID 多人物几何 smoke test

在 4 个三人都可用的 cuts 上：

| 方法 | Camera T mean | Rotation mean |
|---|---:|---:|
| Oracle best single | 0.816 m | 13.48 deg |
| Three-human mean | 0.664 m | 12.68 deg |
| Confidence weighted | **0.575 m** | **9.96 deg** |

三个人的几何共识通常优于最佳单人，但 `cam05 -> cam08` 明显退化。说明：

- 多人具有真实冗余收益；
- 不能简单平均所有人物；
- head confidence 不足以识别几何异常人物；
- 必须加入 pairwise layout、translation dispersion、Huber/trimmed consensus 和 reject-then-resolve。

该结果目前仍是 full-body orientation/root smoke test，尚不是正式的：

```text
Fixed Explicit coarse
+ per-human V16 torso-motion residual
+ shared 20 deg bound
```

不能将上述数字写成最终 V20 结果。

## 7. EgoHumans 有什么、没有什么

### 有

- 8 路同步完整 RGB；
- 3 个稳定身份；
- bbox；
- 133 点 2D pose；
- 17 点 3D pose；
- 最终 SMPL parameters、joints、vertices；
- 鱼眼 intrinsics/distortion；
- exo camera extrinsics；
- Aria/COLMAP world conversion；
- 第一人称和第三人称多视角；
- 真实遮挡、弯腰、多人接近和视角漏检。

### 没有或不足

- 没有最终 SMPL-X GT；
- 没有独立逐像素人物 segmentation mask；
- 没有 dense GT scene depth；
- 没有完整 GT scene mesh；
- 当前输入没有去鱼眼畸变；
- GT body 是优化拟合，不是独立 mocap ground truth；
- 只有一个 capture/三个人，不能单独证明跨人物泛化。

## 8. EgoHumans 推荐用途

最适合：

- Human3R 多人 detection/within-shot tracking；
- wide-view camera-cut Re-ID；
- token/shape/pose feature 对比；
- bbox 和 2D keypoint identity supervision；
- GT-ID multi-human Boundary Oracle；
- 遮挡过多时排除人物；
- 只剩一人时 fallback；
- 鱼眼和边缘人物鲁棒性；
- camera translation/rotation 评价；
- 公共 joints/root 的绝对评价。

不适合单独承担：

- SMPL-X full vertex benchmark；
- dense scene pointmap benchmark；
- 长期跨人物泛化结论；
- 不经 fisheye-aware conversion 的精确 pinhole reprojection。

---

# 第二部分：MultiHuman

## 9. MultiHuman 解压位置和总体结构

原始目录：

```text
/data/wangzheng/iJCV-CODE/data/MultiHuman
```

两个原始压缩包保留不变：

```text
MultiHuman.zip
Real-World-Capture.zip
```

解压后：

```text
MultiHuman/MultiHuman/
MultiHuman/Real-World-Capture/
MultiHuman/Real-World-Capture/extracted/
```

当前总占用约 34 GB。

`Real-World-Capture.zip` 外层 ZIP64 header 对旧版 `unzip/7z` 不兼容，但 Python ZIP64 能完整读取；内部 7 个子包均已成功解压并通过读取/CRC 过程，没有修改原压缩文件。

## 10. MultiHuman 静态部分

目录：

```text
/data/wangzheng/iJCV-CODE/data/MultiHuman/MultiHuman
```

分类：

| 类别 | 场景数 | 每场人数 | individual SMPL-X 数量 |
|---|---:|---:|---:|
| `single` | 30 | 1 | 30 |
| `single_occluded` | 18 | 1 | 18 |
| `two_closely_inter` | 30 | 2 | 60 |
| `two_naturally_interactive` | 46 | 2 | 92 |
| `three` | 26 | 3 | 78 |

目录通常包含：

```text
obj/       每个单人的扫描 mesh
obj_all/   同一场景合并后的多人 mesh
smplx/     每个人拟合后的 SMPL-X OBJ
```

SMPL-X topology：

```text
10475 vertices
20908 faces
```

### 静态部分适合做什么

- 验证读取和保存标准 SMPL-X topology；
- 测试同一个 Sim(3)/SE(3) 是否正确作用于所有人；
- 测试 individual mesh 与 `obj_all` 的多人布局；
- 测试 pairwise human layout residual；
- 生成几何单元测试和合成 camera projection；
- 研究紧密接触、人体互相穿插和遮挡时的 mesh relation。

### 静态部分不能做什么

- 没有 RGB，不能运行 Human3R；
- 没有时间序列，不能测试 tracking 或 Re-ID；
- 没有真实 camera cut；
- 没有 camera calibration；
- 不能评价 recurrent multi-shot drift。

因此静态部分只是几何辅助数据，不是 V20 主调试流。

## 11. MultiHuman Real-World-Capture

完整解压目录：

```text
/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted
```

包含：

```text
box/
box_original_video/
dance/
dance_original_video/
three/
three_original_video/
zyx_single/
```

核心动态多人序列：

| 序列 | 人数 | 有效 RGB/parameter/SMPL-X 公共帧 | 人物身份 | 相机 |
|---|---:|---:|---|---:|
| `box` | 2 | 264，frame 434-699，缺 435/436 | person0/1 | 6 |
| `dance` | 2 | 645，frame 109-753 | person0/1 | 6 |
| `three` | 3 | 1177，frame 379-1555 | person0/1/2 | 6 |

`three` 是首选调试序列，因为：

- 3 人完整存在；
- 1177 个连续标注帧无缺失；
- 6 路原始视频都是 1677 帧；
- 三人距离近、交互和遮挡明显；
- 每帧每人都有 SMPL-X。

## 12. Real-World-Capture 原始全画面视频

以 `three` 为例：

```text
three_original_video/three_new/
```

6 个视频和 camera index 的映射：

| Camera index | Video |
|---:|---|
| 0 | `SaveToAvi-MJPG-18181923-0000.mp4` |
| 1 | `SaveToAvi-MJPG-18181924-0000.mp4` |
| 2 | `SaveToAvi-MJPG-18307701-0000.mp4` |
| 3 | `SaveToAvi-MJPG-18307863-0000.mp4` |
| 4 | `SaveToAvi-MJPG-18307864-0000.mp4` |
| 5 | `SaveToAvi-MJPG-18307870-0000.mp4` |

视频格式：

```text
2048x2048
30 FPS
three: 每路 1677 帧
```

Movie3R/Human3R 应使用这些**完整全画面**，输入规则：

```text
2048x2048 -> 512x512
```

只缩放，不裁剪。

## 13. 不要把逐人物裁剪图当作多人输入

例如：

```text
three/three/person0/img/1000/0.jpg
three/three/person1/img/1000/0.jpg
three/three/person2/img/1000/0.jpg
```

这些虽然都是 512x512，但围绕不同人物分别裁剪，内容和 intrinsics 不同。它们适合：

- 检查单人 GT；
- 验证人物对应关系；
- 外观/身份 probe；
- SMPL-X crop reprojection。

它们不适合：

- 作为全场景多人 Human3R 输入；
- 估计 shared shot Boundary；
- 评价人物之间的完整图像布局；
- 评价背景 pointmap。

如果错误地把 person0、person1 的 crop 当成两个 camera shot，会把“人物裁剪变化”误认为 camera cut。

## 14. Real-World-Capture 人体 GT

每个有效 frame、每个人都有：

```text
personX/smplx/<frame>/smplx.obj
```

每个 OBJ：

```text
10475 vertices
20908 faces
```

优点：

- 与 Human3R 输出同为 SMPL-X topology；
- 可以直接做对应 vertex comparison；
- 可以得到每个人的世界 trajectory；
- 可以从标准 SMPL-X joint regressor 得到 joints；
- 可以计算人物间相对布局、遮挡和 penetration。

限制：

- 主要提供最终 OBJ，不提供完整 betas/body_pose/global_orient 参数文件；
- root/global orientation 需要从 vertices/joint regressor 或配准定义；
- mesh 仍然是多视角拟合结果，不是独立 mocap 真值。

## 15. Real-World-Capture 相机标定

原始视频标定：

```text
*_original_video/calibration_new.json
```

每个 camera 包含：

```text
K: 3x3
R: 3x3
T: 3
```

没有 distortion coefficients。与 EgoHumans 不同，这里不应直接标记为 `OPENCV_FISHEYE`。当前应按数据给出的 pinhole K/R/T 使用；若后续发现边缘系统误差，再单独审计图像是否已经去畸变。

逐人物裁剪参数：

```text
personX/parameter/<frame>/<camera>_intrinsic.npy
personX/parameter/<frame>/<camera>_extrinsic.npy
```

性质：

- extrinsic 为 3x4；
- 同一 frame/camera 的 extrinsic 在所有 person 文件夹中完全一致；
- crop intrinsic 因人物裁剪位置不同而不同；
- 原始 JSON `R` 与 SMPL-X mesh gauge 的 extrinsic rotation 之间存在一个固定公共轴旋转；
- 该公共旋转对所有 6 个相机一致，最大数值离散约 `8e-7`；
- 没有发现人物相关 scale 或独立 world transform。

建议的评测实现：

- 完整 2048x2048 图像使用 `calibration_new.json` 的 full-image `K`；
- SMPL-X world 到 camera 使用逐帧 `*_extrinsic.npy`，或显式将 JSON rotation 转到相同 SMPL-X gauge；
- 不要直接混用 JSON `R` 和 SMPL-X vertices，而忽略固定轴转换；
- camera GT 只能用于 evaluator，不能用于可部署 Boundary generation。

## 16. MultiHuman 投影闭环审计

已在 `three` frame 1000 上完成：

- 从 6 路原始 MP4 读取同一 frame；
- 读取 person0/1/2 的 SMPL-X vertices；
- 使用 full-image K 和 SMPL-X-gauge extrinsic 投影；
- 三个人的 projected bbox 与图像人物位置一致；
- crop K 投影结果也落在对应 512x512 crop 内。

例如 camera 0：

```text
person0 full bbox: [817, 324, 1378, 1494]
person1 full bbox: [728, 290, 960, 1576]
person2 full bbox: [25, 377, 536, 1692]
```

可视化：

```text
output/v20_multihuman_dataset_audit/three_frame1000_contact.jpg
output/v20_multihuman_dataset_audit/three_frame1000/projection_overlay_cam0.jpg
```

这证明 full video、camera calibration、identity folder 和 SMPL-X mesh 可以组成同一个明确 world gauge。

## 17. MultiHuman 有什么、没有什么

### 有

- 6 路同步完整 RGB 视频；
- 2 人和 3 人真实交互；
- 稳定 `person0/1/2` identity；
- 连续 264/645/1177 帧；
- 每帧每人的标准 SMPL-X vertices；
- full-image K/R/T；
- crop-adjusted intrinsics 和统一 extrinsics；
- 近距离遮挡、交叉、前后遮挡和动作变化；
- 原始 full video，可构造真实 recurrent camera cuts。

### 没有或不足

- 没有现成 2D keypoints；
- 没有现成 per-person bbox，但可由 GT mesh 投影；
- `box/dance/three` 没有现成逐人 segmentation mask；
- 没有 GT scene depth/pointmap/scene mesh；
- 没有完整 SMPL-X parameter vector；
- 没有 source/capture 多样性，只有少量真实 capture；
- `box/dance` 各相机原始视频 frame count 有 1-2 帧差异，使用前要按公共有效 frame 对齐；
- 当前尚未运行 Human3R 多人 probe，只有数据完整性和 GT 投影审计。

## 18. MultiHuman 推荐用途

最适合：

- 无 cut 多人 detection 和 within-shot track ID；
- 两人/三人遮挡、交叉和接触；
- native token 跨 view Re-ID；
- shape/pose/token/appearance feature 对比；
- GT-ID multi-human Boundary Oracle；
- per-human V16 candidate 和 robust SO(3) consensus；
- translation candidate dispersion；
- pairwise human-layout residual；
- 错误 ID 的 geometry verification；
- 1/2/4/8 cut 真实 recurrent rollout；
- per-person SMPL-X world vertex error；
- relative human layout 和 identity trajectory；
- 人物进入/离开/被完全遮挡时的 tracklet TTL 调试。

不适合单独承担：

- scene pointmap accuracy benchmark；
- background scale GT；
- 大规模跨 capture/跨 subject 泛化；
- 依赖现成 2D keypoints 或 mask 的实验，除非先生成这些标注。

---

# 第三部分：两套数据如何联合使用

## 19. 推荐职责分工

| 研究问题 | 首选数据 | 原因 |
|---|---|---|
| Human3R 是否检测到所有人 | EgoHumans + MultiHuman | 一个有 bbox，一个有更强遮挡 |
| 输出数组 index 是否稳定 ID | 两者 | 都有稳定 GT identity |
| Native `H'` 是否支持跨 shot Re-ID | 两者 | EgoHumans 已有负结果，MultiHuman 可验证动态动作 |
| shape 是否比 token 更稳定 | 两者 | 需要避免只在三个固定人物上过拟合 |
| local pose 是否只是同步 cue | MultiHuman | 可测试不同时间戳和明显动作变化 |
| 多人是否改善 Boundary | MultiHuman `three` 优先 | 3 人动态连续、SMPL-X GT 完整 |
| 遮挡人物是否应剔除 | MultiHuman | 人体距离近、遮挡明显 |
| 只剩一个人时 fallback | EgoHumans `cam04` | 已观察到只检测到 1 人 |
| 鱼眼边缘鲁棒性 | EgoHumans | 明确 fisheye 标定 |
| SMPL-X full vertex error | MultiHuman | GT 与 Human3R topology 一致 |
| 2D keypoint reprojection | EgoHumans | 有 133 点 GT |
| scene pointmap accuracy | 两者都不足 | 缺 dense scene GT |
| 长期泛化或 benchmark | 两者都不足 | capture/subject 数量有限 |

## 20. 推荐的实验顺序

### 阶段 A：无 cut 多人基础链路

在 MultiHuman `three` 的单个 camera 上连续运行：

```text
full 2048x2048 frame
-> resize 512x512, no crop
-> Human3R max_humans > 1
```

记录：

- detected human count；
- native `smpl_id`；
- refined `H'`；
- head/CUT3R/MHMR tokens；
- predicted SMPL-X；
- 与 GT projected bbox/vertices 的对应关系。

先证明无 cut 时 detection 和 ID 基础链路可用。

### 阶段 B：GT-ID 多人物几何硬门槛

优先使用 MultiHuman `three`：

```text
Hard Reset
+ Fixed Explicit
+ per-person V16 torso-motion residual
+ one shared 20 deg bound
+ robust translation consensus
```

比较：

- first/largest/highest-confidence single；
- Oracle best single；
- mean/weighted mean；
- Huber/trimmed/RANSAC；
- rotation medoid；
- pairwise-layout outlier rejection。

如果 GT-ID 多人鲁棒共识都不优于单人，应停止“多人改善 alignment”的主张。

### 阶段 C：跨 shot identity probe

两种 cut 都要测试：

1. 同一 timestamp 不同 camera：隔离 view change；
2. 不同 timestamp 不同 camera：同时包含 view change 和动作变化。

比较：

- refined `H'`；
- fused prompt；
- CUT3R head token；
- Multi-HMR token；
- beta/shape；
- local pose；
- token + shape + pose；
- zero/shuffle/wrong-person controls。

EgoHumans 用于有 bbox/2D GT 的受控 probe；MultiHuman 用于动态动作和 SMPL-X GT。

### 阶段 D：可部署 Match-Then-Align

只有 GT-ID geometry 和 identity probe 都有正结果后，才实现：

```text
tentative association
-> all matches produce geometry candidates
-> robust shared Boundary solve
-> reject identity/geometry outliers
-> at most one re-solve
-> Align-Then-Commit
```

Fallback：

```text
>=2 reliable humans -> multi-human consensus
1 reliable human    -> existing single-human V14.7 path
0 reliable humans   -> Fixed Explicit / scene fallback
```

### 阶段 E：真实 recurrent multi-cut

使用 MultiHuman full video 构造：

```text
camera 0 -> 1 -> 2 -> 3 -> 0 ...
1/2/4/8 cuts
```

每次 cut 后都使用上一次预测 world，不能在每个 cut 重新恢复 GT gauge。

## 21. 推荐的第一条 MultiHuman 流

数据：

```text
Real-World-Capture/extracted/three_original_video/three_new
```

建议先选 frame 980-1019 附近，因为三个人在多个视角中都清晰可见，同时存在部分互相遮挡。

第一轮可构造同步 Boundary：

```text
camera0: frame 996-1000
camera1: frame 1000-1004
camera2: frame 1004-1008
```

Boundary timestamp 重复，用于隔离相机变化。

第二轮构造真实连续时间：

```text
camera0: frame 996-1000
camera1: frame 1001-1005
camera2: frame 1006-1010
```

该版本同时测试 motion prediction。

输入必须来自原始 full MP4，不使用 person crop。

## 22. GT 使用和信息泄漏规则

两套数据都应遵守：

| 变量 | Candidate generation | Evaluation/Oracle |
|---|---:|---:|
| RGB | 是 | 是 |
| GT cut index | 可作为触发信号 | 是 |
| GT identity | 否 | 是 |
| GT bbox/keypoints | 默认否 | 是 |
| GT camera | 否 | 是 |
| GT SMPL/SMPL-X | 否 | 是 |
| GT scene/scale | 否 | 是 |
| source/camera ID | 不进入 selector | 用于分组报告 |

受控 feature probe 可以用 GT ID 构造正确 prototype，以单独测量 feature separability，但必须明确标记为 probe，不得把该准确率当成可部署 Re-ID。

## 23. 当前最合理的数据策略

短期调试建议：

1. 用 **MultiHuman `three`** 完成正式 GT-ID multi-human Fixed Explicit/V16 几何门槛。
2. 用 **EgoHumans** 验证 bbox/2D keypoint identity、鱼眼视角和只剩一人的 fallback。
3. 用两者共同测试 native token 是否跨 camera、跨动作稳定。
4. MultiHuman geometry 成立后，再实现 external identity bank、dustbin、TTL 和 geometry verification。
5. DA3、V11.4 shared scale 和 VGGT 在 GT-ID Lite 多人收益成立前保持关闭。
6. 两套数据都只定位为开发/调试数据，最终论文仍需要 capture-disjoint、subject-disjoint 的更大 holdout。

## 24. 最终结论

### EgoHumans

是一套标注丰富、坐标明确、包含鱼眼和视角漏人的三人同步多相机数据。它对 detection、2D identity、camera evaluation 和 fallback 非常有价值，但人体 GT 是 SMPL，不适合直接做 Human3R SMPL-X full vertex 评价。

### MultiHuman 静态部分

是一套有价值的多人 mesh/SMPL-X 几何库，但没有 RGB、时间和相机，只能作为几何单元测试与合成诊断。

### MultiHuman Real-World-Capture

是当前更适合调试动态多人 V20 的数据：有 6 路同步 full video、稳定 person ID、连续两人/三人交互和逐帧标准 SMPL-X。它最适合验证多人是否真的改善 Boundary、遮挡人物是否能被剔除，以及 multi-cut 后 per-person trajectory 是否稳定。

两套数据不是替代关系，而是互补关系：

```text
EgoHumans:
  richer 2D labels + fisheye + camera/dropout diagnostics

MultiHuman:
  dynamic close interaction + direct SMPL-X vertices + recurrent multi-cut
```

最合理的下一步不是立即训练 Re-ID adapter，而是先用 MultiHuman `three` 完成 GT-ID robust multi-human geometry gate，再回到两套数据共同验证身份模块。
