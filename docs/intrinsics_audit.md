# NYU / Mirror3D 内参与像素坐标审计

本次审计只读核对官方 NYU 工具箱与 Mirror3D 官方源码；未改变 GLARE、原始数据或已发布的像素。下列矩阵均使用 TrustPrompt 的整数像素中心、0-based 索引；轴为 x 向右、y 向下、z 向前。**已确认数值与源码行为，不等于已确认本地 MAT 生成时所用工具箱版本。**

## 结论与使用建议

1. NYU `rawDepths` 是投影到 RGB 图像网格、未补洞的米制深度。不能为它直接使用原始红外深度相机约 582.6 的焦距。RGB 焦距约为 518.86 / 519.47。[NYU 官方数据说明](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
2. Mirror3D 的 456×608 图像来自 480×640 图像裁去上/下 12 行、左/右 16 列，**没有缩放**。因此主点仅减去 `(16, 12)`，焦距保持不变。[官方导出脚本](https://github.com/3dlg-hcvc/mirror3d/blob/main/mirror3d/utils/export_mat_image.py)
3. 官方 NYU 工具箱内部存在可逐行证明的 1 像素约定差异：RGB 去畸变使用 0-based 射线；深度写回投影图时把投影坐标直接当作 MATLAB 的 1-based 数组索引。故不能笼统声称“MATLAB 代码中的主点全部减 1 就得到唯一正确内参”。原包没有 `undistort_image.m`，实际文件名是 `undistort.m`。[官方工具箱 ZIP](https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip)
4. 像素级 `raw_prompt` 验证可以继续采用发布数组原有对应关系，无须使用 K。本项目当前选择共同的 `REGISTERED_K`，采用下文的 depthOut 索引分支主点 `(308.58244941119034,240.73616633400465)`，并设置 `K_rgb=K_prompt=REGISTERED_K`。这是已记录的发布网格几何近似，**不是物理 RGB 去畸变分支的精确 K**。不要为消除本次发现而私自移动 RGB、提示、GT 或掩码。对于需要 K 的几何模块，必须标注选用的约定，不能标为“精确标定与 RGB-Z 已完全核验”。
5. 用于重现 Mirror3D 平面标注的 `K_annotation = [[519,0,304],[0,519,228],[0,0,1]]` 应单独保存，不能替代 NYU 的物理 RGB 标定。[官方可视化说明](https://github.com/3dlg-hcvc/mirror3d/blob/main/docs/Mirror3D_dataset.md)；[平面与点云工具](https://github.com/3dlg-hcvc/mirror3d/blob/main/mirror3d/utils/plane_pcd_utils.py)

## 数值与推导

`camera_params.m` 的 RGB 标定常数为：

| 参数 | 原文件数值 |
| --- | ---: |
| fx_rgb | 518.8579011745019 |
| fy_rgb | 519.4696111212749 |
| cx_rgb | 325.58244941119034 |
| cy_rgb | 253.73616633400465 |

此处“原文件数值”特意不预先贴 0-based 或 1-based 标签：不同调用分支对它的解释不一致。以下两个矩阵分别由各分支行为推导，不能把差异当作浮点精度误差。

### RGB 去畸变分支：物理 RGB 网格的候选 K

`project_depth_map.m` 按 RGB 标定调用 `undistort.m`；后者保持同一焦距和主点建立输出内参。其内部 MATLAB 网格从 1 开始，但反投影射线显式使用 `px-1` 和 `py-1`，插值的 0-based 坐标转回数组索引时再加 1。因此此分支输出 RGB 的 Python 图像坐标对应原数值，不应再额外减 1。

由此推导的完整 480×640 去畸变 RGB 内参：

```text
K_rgb_full_0based =
[[518.8579011745019,   0.0,               325.58244941119034],
 [  0.0,             519.4696111212749,   253.73616633400465],
 [  0.0,               0.0,                 1.0]]
```

Mirror3D 裁剪后 456×608：

```text
K_rgb =
[[518.8579011745019,   0.0,               309.58244941119034],
 [  0.0,             519.4696111212749,   241.73616633400465],
 [  0.0,               0.0,                 1.0]]
```

这是**按当前官方工具箱 RGB 去畸变分支推导**的物理 RGB 候选内参；本地 `images` 是否由完全同版流程生成尚未做原始 RGB 重投影复核。本项目当前共同 `REGISTERED_K` 的选择见下节，不能把两者混称为已核验的同一个矩阵。来源：[官方工具箱中的 camera_params.m、project_depth_map.m、undistort.m](https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip)。

### 深度投影分支：K_prompt 的一像素歧义

`rgb_world2rgb_plane.m` 用原始 RGB 标定将 3D 点投影为 `(u,v)`。`project_depth_map.m` 对它取整后直接作为 MATLAB `depthOut` 的列、行索引，没有加 1。转为 Python 后，实际数组位置是 `(u-1,v-1)`。`rgb_plane2rgb_world.m` 也以 1-based 网格直接减原始主点，与这一深度数组解释一致。

若本地 `rawDepths` 确由这一未修改分支生成，则应推荐：

```text
K_prompt_full_0based =
[[518.8579011745019,   0.0,               324.58244941119034],
 [  0.0,             519.4696111212749,   252.73616633400465],
 [  0.0,               0.0,                 1.0]]

# 再裁去左16、上12；并非把分辨率缩为0.95倍。
K_prompt_project_depth_map_branch =
[[518.8579011745019,   0.0,               308.58244941119034],
 [  0.0,             519.4696111212749,   240.73616633400465],
 [  0.0,               0.0,                 1.0]]
```

**这不证明应该立即把 sample 的 K_prompt 改成与 K_rgb 不同的矩阵。** 公开说明将 `images` 与 `rawDepths` 描述为对齐数据，但未给出本地 MAT 生成脚本版本；工具箱本身有上述分支冲突。当前采用以下可追溯的临时选择：

- 保留发布数据的共同像素网格，不移动任何数组；设置 `K_prompt=K_rgb=K_prompt_project_depth_map_branch`，在实现中命名为 `REGISTERED_K`。这采用 depthOut 数组的索引约定近似描述共同发布网格，并未消除物理 RGB 的 1 px 分支冲突。
- 元数据明确记录该共同 K 是近似、存在 1 px caveat，设置 `strict_rgb_z_qualified=False`。它适合当前不使用 K 的像素深度指标闭环；它不是精确 RGB 光心标定或精确配准认证。上节物理 RGB 候选 K 保留作后续核查证据，不能无记录覆盖当前 K。
- 在需要精确射线的后续几何实验前，取得本地 MAT 的生成流程，或用与 MAT 帧对应的原始 RGB/IR 帧重新运行注册并比对；确认实际分支后，再决定共同网格重采样或分别设置 K，并显式处理 RGB 与提示的网格关系。

不能用测试集误差挑选主点分支，也不能为获得更好结果默默给某一个方法平移输入。来源：[官方工具箱中的 rgb_world2rgb_plane.m、rgb_plane2rgb_world.m、project_depth_map.m](https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip)。

### Mirror3D 标注网格

官方 Mirror3D 可视化脚本为 NYU 使用焦距 519。`plane_pcd_utils.py` 的相关 Python 循环使用 0-based 像素，并按图像半宽、半高定义主点。因此在 456×608 上：

```text
K_annotation = [[519, 0, 304], [0, 519, 228], [0, 0, 1]]
```

这里是 `W/2,H/2`，不是 `(W-1)/2,(H-1)/2`。该矩阵是复现官方平面/点云工具的约定；主点和 RGB 标定相差数个像素，不能称为同一个标定矩阵。重建 Mirror3D 平面 GT 时应使用标注约定及经核验的平面坐标轴转换；直接使用已发布 delta/refined depth 时无需重新渲染平面。[官方可视化代码](https://github.com/3dlg-hcvc/mirror3d/blob/main/mirror3d/visualization/check_sample_info.py)；[官方平面/点云工具](https://github.com/3dlg-hcvc/mirror3d/blob/main/mirror3d/utils/plane_pcd_utils.py)

此外，NYU 工具箱 `crop_image.m` 调用的是另一种投影有效区裁剪，保留 MATLAB 行45至471、列41至601，输出427×561。它与 Mirror3D 的 5% 中心裁剪无关，不能混用。

## 深度轴定义的额外限制

官方说明证实 `rawDepths` 已除去 Kinect 非线性、单位为米，且注册到 RGB 图像平面；这足以排除把它视为 disparity 或径向 range 的实现。[NYU 官方数据说明](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

但审计到的 `project_depth_map.m` 用经外参变换的点计算投影位置，Z-buffer 写入值仍来自变换前的 `depth2`；并未直接写入 RGB 坐标系的第三坐标。故严格表述应为“米制传感器轴向深度，按 NYU 官方流程注册到 RGB 网格”，而非已证明是精确 RGB 相机轴向 Z。NYU 发布网格的深度误差评测可以采用该发布约定；后续需要真实 RGB 相机点云的几何实验必须明确这一近似，保留源码版本并核验变换，不能自行把原数据当作欧氏 range 转换一次。

本次没有改写深度数值，没有推定误差大小，没有声称完成精确 RGB-Z 重建。平面修正 GT 又引入 Mirror3D 的标注相机约定，故首阶段结果应标明为 NYU/Mirror3D 发布深度口径的工程验证。来源：[官方 project_depth_map.m 和 depth_world2rgb_world.m](https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip)。

## rawDepths 的 10 米有效范围

`camera_params.m` 设置 `maxDepth=10`，注释说明单位为米。`project_depth_map.m` 在输出前将超过这一上限的值截顶到 10，将负值及 NaN 置为 0。因此当前源提示有效性采用 `isfinite(rawDepths) & (rawDepths > 0) & (rawDepths <= 10)` 有工具箱输出范围依据。建议配置显式保存 `sensor_valid_max_m=10.0` 和上界包含关系，而非仅把阈值藏在代码中。[官方工具箱](https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip)

10 米是截顶值，不等于确认真实表面恰好位于 10 米处。可另计 10 米饱和像素数量作为数据审计，不应按某方法的预测结果选择是否剔除。此传感器范围约束也不能自动替代镜面平面 GT 的有效范围；GT 范围与主评测最大深度应单独声明并保持方法无关。

## 来源校验和与可复核性

官方 ZIP 本次读取大小为 30,990 字节，未复制数据或权重。以下 SHA256 基于下载响应内的原始文件字节，包含原始换行；不是网页渲染后文本的散列。

| 来源 | SHA256 |
| --- | --- |
| toolbox_nyu_depth_v2.zip | `6211b20f78a3b90e111b4e9f8d549acf3cfc298fdcb25164222edb9a994f3e36` |
| camera_params.m | `03ad3755b605d1a4d2943c8cd56d722f1adec11c1ca7c761d5cf20dad5a1f660` |
| project_depth_map.m | `78485d3a43482e3dd53279e8c86db6e966ec0bb398c524464376b0142bfc4286` |
| undistort.m | `8041a527d66b0ed378d2b0e016c775eaab0660a67628f151b53da3ad5c28ffda` |
| rgb_world2rgb_plane.m | `7dfcc4335ce88e2681053b4f3b56373e847a5a3c8795d671879c88c672e89ece` |
| rgb_plane2rgb_world.m | `7115fe2edc7a4c2f3ce01dfa1ff9c6365fc24e9b02532718ec930cb79179517d` |
| depth_world2rgb_world.m | `555d779e66cd76c18f3558a137af6fa866a5cfcdc889e02617eb9720f924bfdd` |
| crop_image.m | `03c913f2077ac645c6bdfb6e5d9f9d6bc93c7986f917a0bba2422f72ee89918c` |
| get_projection_mask.m | `8b7c2f36bcb1e4e9d747c9246f1793a2a21eb1fadeb897195cbfa49ca5a051c9` |
| Mirror3D mirror3d/utils/export_mat_image.py | `2fd20041bff2dc656dcedd03fa0f63f5d2a4ddfe2938bc80b216c387e86658ae` |
| Mirror3D mirror3d/visualization/check_sample_info.py | `f24173da1383efc06eca959cbb6fcc8caab6ad31128b8df063f9ddc60edb4961` |
| Mirror3D mirror3d/utils/plane_pcd_utils.py | `9c7fcbe17573cb387b6d98f990bb0a522e3cc1889cf7d3efae07e9afa2c55925` |

Mirror3D 的原始文件 URL 为官方仓库 `main` 分支；此处使用内容 SHA256 固定本次证据，不把可变分支名当作永久版本。
