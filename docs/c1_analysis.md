# C1 开发诊断：PromptDA 是否继承错误提示（2026-10-11）

**状态：小规模验证通过的开发诊断，不是正式结果。** 只用 train/val；测试集未读取。无尺度对齐；逐图→逐场景→跨场景等权。数据表：`docs/dev_results.md`（NYU）、`docs/hammer_dev_results.md`（HAMMER）；配对 bootstrap 见下列 results 目录。

## 结论摘要

1. **大面积、连贯的错误提示（NYU 镜面）会被 PromptDA 大量继承。** 镜面污染区内，PromptDA-L 的 AbsRel 比免训练的全局鲁棒对齐高 0.49（train，49 个场景，95% CI [0.38, 0.60]）和 0.33（val，16 个场景，CI [0.10, 0.57]）；继承率高 0.56 / 0.34，危险高估率高 0.48 / 0.27。C1 在镜面上成立且显著。
2. **小而分散的错误提示（HAMMER 桌面透明/反光物体）基本没有被继承。** 物体像素上 PromptDA-L 明显优于鲁棒对齐：AbsRel 低 0.067（train，9 个场景，CI [−0.080, −0.055]），继承率低 0.18，危险高估率低 0.26。val 上 PromptDA-L 的污染区 AbsRel 0.043、继承率 0.15，好于所有对齐基线。
3. **继承程度随污染面积上升**：NYU 镜面每张图的污染面积比例与 PromptDA-L 继承率的 Spearman ρ = 0.49（p = 5×10⁻⁶，78 张）；面积 < 0.5% 时中位继承率 0.03，5%–20% 时 0.93。HAMMER 的污染分散在多个小物体上，面积多在 2%–5%，继承率中位 0.06–0.12，与面积无关（ρ = 0.03）。（`results/c1_inheritance_vs_area_v1.json`）
4. **神谕上界**（用真值掩码删除错误提示，再用全局对齐先验 D_g 填充）：
   - NYU：在镜面上把 PromptDA-L 从 0.467 降到 0.096（val），但与“直接输出 D_g”几乎相同（train 差 +0.003，CI 跨 0）。填充后 PromptDA 在镜面里只是复现 D_g；**提升的瓶颈在填充质量，而不在 PromptDA 本身。** 镜面是平面，计划中的平面候选 k3 正好针对这一点。
   - HAMMER：完美可信度 + k0 填充把 PromptDA-L 从 0.043 降到 0.032（−26%），继承率从 0.155 降到 0.115，危险高估率略升 0.008。去掉全部物体像素（关注区掩码）反而略差（0.045），因为删掉了物体上本来正确的提示。仅有 1 个场景，只作描述。
5. PromptDA-S-Transparent（作者在 HAMMER 上微调过）在 HAMMER val 上不比 PromptDA-L 好（0.051 vs 0.043），在 NYU 镜面上更差（0.646 vs 0.467）。

## 对计划的影响（已写入 plan_revision_v2 R13）

- C1 的表述需要限定：继承在**大面积、连贯**的污染（镜子、大块玻璃）上严重，在桌面级小物体上较轻。论文主线应围绕这一规律展开（原计划 F1 图正是“继承率 vs 污染面积”）。
- 数据优先级随之调整：需要更多大面积污染的真实数据——镜子与玻璃门/窗（自采 iPhone、MP3D/ScanNet 镜面子集）。HAMMER 仍用于留一传感器实验和小物体场景，但不是展示 C1 的主力。
- 方法上：NYU 结果说明**低自由度填充（模块 C）是镜面上的主要收益来源**，应在训练 T 之前，先用神谕掩码评估 k0–k3 候选的上界（CPU 即可完成）。HAMMER 上 PromptDA-L 已经较强，剩余空间主要在可信度判断与 PromptDA 改造（模块 D）。

## 数据与运行

- NYU：train 76 帧 / 55 场景，val 19 帧 / 18 场景（镜面污染非空：train 49 个场景，val 16 个场景）。
- HAMMER：train 816 个样本 / 9 场景，val 327 个样本 / 1 场景（scene 11）；3 种传感器合并，按传感器分解见表。
- 缓存：`cache/promptda_*_nyu_dev_v1`、`E:\TrustPrompt_data\cache\promptda_*_hammer_dev_v1`，全部 `real_dev_verified`，权重安全模式加载。HAMMER 的 L/S-T/神谕缓存用 0.2 mm 定点存储，其中 PromptDA-L 有 718 个像素超过 13.107 m 被截断（占 1,143 张图像素总数的 0.0001%）。
- 配对比较：`results/mirror3d_dev_paired_promptda_{val,train}_v1`、`results/hammer_dev_paired_promptda_train_v1`、`results/hammer_dev_paired_promptda_oracle_val_v1`。
