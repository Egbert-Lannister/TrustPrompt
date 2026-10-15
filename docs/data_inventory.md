# 数据清单（2026-10-15）

项目不再使用 E 盘。以下为项目目录内现有的数据，以及已删除数据的重新获取方式。

## 现有（都在 D:\Python_Script\TrustPrompt 下，不进 git）

| 内容 | 位置 | 用途 |
|---|---|---|
| NYU 开发子集（train 76 / val 19 帧）原始 h5 | `data/source_cache/nyu_dev_v1.h5` | 重新处理 NYU 开发集 |
| NYU 处理后样本 | `data/processed/mirror3d_dev_v1` | 所有 NYU 评测和训练 |
| HAMMER 处理后样本（train 816 / val 327 张） | `data/processed/hammer_dev_v1` | 所有 HAMMER 评测和训练 |
| HAMMER 远程索引 | `data/source_cache/hammer_index_v1.json` | 按需重新下载 HAMMER 原始文件 |
| UniDepth、PromptDA（NYU 全部；HAMMER 仅 2026-10-14 之后的修复缓存）、T 的预测与可信度缓存 | `cache/` | 评测、训练 |
| PromptDA 检查点（三个，SHA256 已核对） | `checkpoints/promptda` | PromptDA 推理 |
| PromptDA 官方源码（提交 9161547，哈希已核对） | `external/PromptDA` | PromptDA 推理 |
| 所有评测结果、运行记录和预测哈希 | `results/` | 表格、配对检验 |

## 已删除（用户在清理 E 盘时删除，2026-10-14 确认）

| 内容 | 影响 | 需要时如何重新获取 |
|---|---|---|
| E 盘上的 HAMMER PromptDA 缓存（raw、S、S-Transparent、focus、oracle、检测器两种、Gate、T v0、T v1） | 已完成的评测结果和指标都在 `results/`，不受影响；只是不能再直接拿这些预测做新组合（例如以 PromptDA 输出为底图的填充）。 | 用对应的 `configs/promptda/hammer_*.yaml` 重新生成（GPU，每份约 1 小时），并可与 results 里记录的预测哈希比对。 |
| HAMMER 原始子集 `hammer_subset_v1`（约 17 GB） | 开发集评测用的是处理后样本，不受影响。最终测试需要的场景 12–14 本来就没有下载。 | `scripts/windows/hammer_fetch.ps1`（按索引对官方压缩包做分段下载，目标已改为 `data/source_cache/hammer_subset_v1`），再用 `scripts/prepare_hammer.py` 处理。 |
| Mirror3D 的 `nyu.csv` 和 `nyu.zip`（原在 GLARE 数据目录） | 开发集已处理完，不受影响；准备 NYU 测试集和重新处理时需要。 | 从 Mirror3D 官方发布处重新下载，放到 `data/source_cache/mirror3d/`（路径已写在 `configs/paths.dev.yaml`）。 |
