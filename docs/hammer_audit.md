# HAMMER 数据审计（2026-10-10/11）

证据：`results/hammer_alignment_check_v1/`（审计脚本、JSON、15 张对齐图、代码快照哈希、环境）。审计在 Claude 工作区用当时的代码快照（`code_snapshot_sha256.txt`）运行，所用 `hammer.py` 与随后提交版本一致；图中第 2、4 格标题仍是 NYU 文案（“precise mirror contour”“background inpainted”），标题已在提交版本中改为按数据集显示。

## 下载完整性
- `E:\TrustPrompt_data\hammer_subset_v1`：选中 65,070 个文件全部记录在 receipt，大小与 ZIP 中央目录一致，提取时逐文件 CRC 校验通过；归档身份（大小/ETag/Last-Modified）与索引一致；411/411 个批次成功，下载 17.03 GB。
- 官方归档中有 1 个异常文件 `scene2_traj2_1/polarization/depth_d435/{06d}.png`，加载器忽略并记录。

## 格式（核实）
- polarization 视角 832×1088；RGB 8 位；`depth_d435` / `depth_l515` / `depth_tof` 与 `_gt` 为 16 位 PNG，单位毫米，0 = 无测量；`_instance` 8 位实例 ID（0–220，步长 20）；`intrinsics.txt` 3×3；`_pose/*.txt` 4×4。
- 每个序列中 rgb、GT、三种深度帧集合完全一致。帧数 28–354/序列；train（scene 2–10）5,253 帧，val（scene 11）540 帧，test（scene 12–14）1,414 帧。
- **没有材质或类别标签**：归档只有 PNG 与 TXT，论文也未给出逐实例材质表（只在图注中描述类别：透明玻璃/塑料、反光餐具、部分反光罐头等）。material 统一记为 0（未知）。

## GT 深度约定
- 用 naked GT 反投影后做 RANSAC 平面拟合（20,000 点，阈值 3 mm）：按轴向 Z 解释的内点数 6,683–11,215，按射线距离解释只有 1,447–2,203，残差也更大。GT 为相机轴向 Z。`strict_rgb_z_qualified=True`（GT 在 polarization 相机渲染）。
- 传感器深度由作者投到该视角；在无物体像素上与 GT 的中位相对差：L515 −0.6%～+0.6%，Helios −1.4%～+0.8%，**D435 +2.0%～+3.9%（系统性偏远）**。

## naked 配对
- 5 帧全部配对成功：位置差 ≤ 0.05 mm，旋转差 ≤ 0.011°。`scene6_traj2` 物体序列 309 帧、naked 316 帧，因此配对按逐帧位姿检查（阈值 1 mm / 0.1°），不按序号假定。
- 物体像素（GT 与 naked GT 差 > 1 mm）占图像 7.7%–15.8%。物体上的有效提示中，18%–51% 更接近 naked（背后）深度而不是物体表面，即穿透/反射现象确实存在。

## 对评测口径的影响（写入 plan_revision_v2）
1. 全图 C 区域混入大量非材质误差：D435 背景偏差使其全图 C 占 8.6%–32%；Helios 在黑色台面上大面积偏远；Helios 视场只覆盖图像中部（空洞约 58%）。
2. 因此 HAMMER 主口径改为 **objects_only**：物体像素（GT vs naked GT > 1 mm）内的 C/Cfar/band，与 NYU 的 mirror_only 对应；全图结果照常报告并注明。
3. 按材质（透明/金属）分解需要逐实例材质表。HAMMER 不提供，需人工标注：每个场景约 6–10 个实例，共约 100 个，由团队按 RGB 标注后作为 `configs/hammer_materials_v1.csv` 版本化；标注前不报告分材质结果。
