# 原生输入兼容性失败：混合分辨率

以下样例的原始 RGB 在第 50 帧切换分辨率。JOSH 的 SAM3 预测已传播到第 100 帧，但其原生输出处理假设掩码与当前图像等大，在 `preprocess/run_sam3.py:88` 写出掩码时抛出 `IndexError`，未进入 TRAM、DECO 和 JOSH 联合优化。

| 样例目录 | 视频片段 | 前 50 帧 | 后 50 帧 |
| --- | --- | --- | --- |
| `cases/egohumans/line083` | `ego_test_volleyball_001_volleyball_extreme_cam02_cam09_b00301` | 1920×1080 | 3840×2160 |
| `cases/egohumans/line085` | `ego_test_volleyball_002_volleyball_extreme_cam09_cam02_b00301` | 3840×2160 | 1920×1080 |
| `cases/egohumans/line088` | `ego_test_volleyball_004_volleyball_extreme_cam02_cam09_b00301` | 1920×1080 | 3840×2160 |

验证依据：逐帧读取 100 个原始 JPEG 的尺寸；各例 `sam3.log` 第 27–29 行为原生异常栈。三个案例的输入尺寸均只在第 50 帧变化。

英文错误原文：

> IndexError: boolean index did not match indexed array along dimension 0

中文：布尔掩码的高度与被索引图像的高度不一致。其中 `line083` / `line088` 为 1080 对 2160，`line085` 为 2160 对 1080。

## 本次处理

- 保留原始输入、原生日志和失败状态，不更改分辨率、掩码、人物 ID 或冻结配置，也不为这几例选择性重跑。
- 按现有失败评测规则保留在完整 90 例分母中：IDF1 和覆盖率为零，几何与相机误差不可用，而不是误差为零。
- 这是**原生发布流程的输入兼容性失败**，不是显存不足，也不能将其单独解释为 JOSH 联合优化无法处理镜头切换的实验证据。
- 若后续另行评测统一分辨率或输出兼容修复，应标注为新的实验设置，并与本轮冻结结果分开保存。当前没有进行该扩展。

## 后续论文中的备选说明

英文：

> Three sequences fail in the released SAM3 preprocessing wrapper because their image resolution changes at the shot boundary; these failures are retained in the full-sequence accounting.

中文：

> 三个片段因镜头边界处的图像分辨率变化，在公开实现的 SAM3 预处理封装中失败；统计完整测试集时仍保留这些失败片段。

使用这句话时，应同时报告完整运行最终的推理成功数和逐指标支持数，不能仅引用当前已结束样例的统计。
