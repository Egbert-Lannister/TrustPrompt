# 对比方法（depth completion baselines）与 Holm 校正（2026-10-16）

状态：**代码已实现，小规模验证通过**（云端单元测试 + CPU 随机权重结构冒烟）。**Windows GPU 尚未运行，没有任何对比结果。**
目前只有 PromptDA-L 一行有结果，来自已有缓存，见第 5 节。

## 1. 目的

对比方法同时承担两件事：

1. **对比**：和我们的方法 v1 放在同一张表里比较。方法 v1 = T v1 + conformal 阈值 + D_g 填充 + PromptDA-L。
2. **检验“T 可插拔”**：把 T 接到每个模型前面，看是否都能降低误差。这是论文第 2 条贡献唯一缺的证据（`docs/t_v2_results.md` §4）。
   - 每个模型各跑三种输入：
     - **raw**：原始传感器提示；
     - **tmask**：T v1 标为不可信的像素置为 0，交给模型自己补全；
     - **oracle_mask**：所有污染像素（|P−GT|/GT > 0.05）置为 0，作为上界 [oracle]。
   - tmask 的阈值是方法 v1 的 conformal 阈值：α = 0.1，危险定义为 P > 1.2·GT；NYU λ̂ = 0.72，HAMMER λ̂ = 0.46。
   - 补全类模型能直接处理空洞，所以被屏蔽的像素不再用 D_g 填充。PromptDA-L 仍按方法 v1 用 D_g 填充。

## 2. 方法清单

| 模型 | 来源（固定 commit） | 权重（固定 HF revision） | 许可 | 状态 |
|---|---|---|---|---|
| PriorDA-B v1.1 | SpatialVision/Prior-Depth-Anything @ 8c029cb | Rain729/Prior-Depth-Anything @ 25b3bfc（DAv2-B + prior_depth_anything_vitb_1_1） | 代码 Apache-2.0；DAv2-B 主干 CC-BY-NC-4.0 | 代码已实现 |
| OMNI-DC v1.1 | princeton-vl/OMNI-DC @ a5ecd35 | zuoym15/OMNI-DC @ 9ec01f0；DAv2-L 取自 Rain729 同一 revision | BSD-3；DAv2-L CC-BY-NC-4.0 | 代码已实现；CPU 结构冒烟到前向（前向内部写死 cuda） |
| CDM（按相机，ViT-L） | ByteDance-Seed/manip-as-in-sim-suite @ 757b2ea（cdm/） | depth-anything/camera-depth-model-{d435, l515, kinect} | 代码 Apache-2.0；权重 CC-BY-NC-4.0 | 代码已实现；CPU 随机权重冒烟通过 |
| LingBot-Depth v0.5 | Robbyant/lingbot-depth @ f3a237e | robbyant/lingbot-depth-pretrain-vitl-14-v0.5 @ 79204ed | Apache-2.0 | 代码已实现；只核对了导入和构造参数 |
| Any2Full-L | zhiyuandaily/Any2Full @ 780edf6 | zhiyuandaily/Any2Full @ 5c8271e（checkpoints/Any2Full_vitl.pth.tar） | Apache-2.0 | 代码已实现；CPU 随机权重冒烟通过 |
| Marigold-DC（可选） | prs-eth/Marigold-DC @ c6d756b | prs-eth/marigold-depth-v1-0 @ f4fc453（fp32 safetensors） | Apache-2.0 | 代码已实现；每张约 30–60 s，默认不跑 |

- 所有仓库都是公开的，不需要登录或同意协议，我们没有替你接受任何协议。许可记录在 `checkpoints/baselines/setup_receipt.json` 和每个缓存的 receipt 里。
- CC-BY-NC 权重只用于研究。
- 权重的大小和 SHA256 是 2026-10-16 从 Hugging Face API 读到的 LFS 值，写在 `configs/baselines/registry.json`，setup 和每次运行都会核对。
- 这些哈希值是经由网页工具抄录的。如果出现“大小一致、哈希不一致”，先当作抄录错误报告给我，不要直接改。

**暂未纳入，以及原因：**

- **LingBot-Depth 2.0**：2026-07 发布了新闻稿，但官方仓库和 HF 合集目前只列出 v0.5、v0.1 和 DC 权重，没有找到 2.0 的公开权重。
- **Mirror3DNet**：需要 Mirror3D 的镜面掩码和平面网络（基于 detectron2 / PlaneRCNN），而且只适用于 NYU/Matterport。适合放到租用的 Linux GPU 上做，列为下一步。
- **GlassRecon 式 patch-RANSAC 全局对齐**：免训练，可以自己实现，作为 D_g 的替代对齐基线，列为下一步。
- **DEPTHOR++ 检测器**：已在 R14 复现（`tp/models/depthor_det.py`），不重复。

## 3. 实现与相对官方的偏差

全部偏差都写进每个缓存的 `receipt.model_version.deviations`。代码在 `baselines/completion.py`。

- **通用**：输入是 RGB 网格上的米制深度，0 表示缺失；RGB 为 uint8。输出按官方脚本还原到原分辨率。不做任何尺度对齐，指标为 Z，与主结果一致。
  - torch ≥ 2.6 把 `torch.load` 的默认值改成了 weights_only=True。PriorDA、OMNI-DC、CDM 的官方代码写在这之前，调用时不传这个参数。对这三个模型，我们设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 恢复旧默认值，只读取已经核对过 SHA256 的文件。
- **PriorDA**：官方依赖 `torch_cluster.knn`，但我们用的 torch 2.11 没有对应的 wheel。改用精确 KD-tree k-NN，并复现 CUDA 核的并列规则（距离相同时取索引小的）。与暴力 k-NN 逐元素一致，见单元测试。其余参数为官方 v1.1 默认值：vitb/vitb、K=5、pattern=None。
- **OMNI-DC**：
  - 参数取自 `demo.sh` / `test_v1.1.sh`；RGB 做 ImageNet 归一化，按 demo.py 补零到 16 的倍数。
  - `pvt.py` 导入 mmcv/mmseg 只是为了加载一个 ImageNet 初始化，v1.1 权重会把它覆盖，所以这两个包用空模块代替。
  - ResNet 初始化文件在 Google Drive 上，没有下载，改用随机初始化。同样会被 v1.1 权重完全覆盖：网络里任何一个参数在 safetensors 中缺失，程序就会报错停止。
  - DAv2-L 从我们固定的副本读取，不走官方的相对路径。
  - 832×1088 全分辨率在 8 GB 显存上能否跑通，要看 preflight 结果。
- **CDM**：
  - 预处理完全照 `cdm/infer.py`：丢弃 > 25 m 的深度，输入逆深度，输出取 1/预测；RGB 数组的传法也和 infer.py 一样，不做改动。
  - **相机不匹配**：CDM 只有 D405/D435/L515/ZED 2i/Azure Kinect 的模型。HAMMER 的 Helios（ToF）和 NYU 的 Kinect v1（结构光）都用 Azure Kinect 模型，属于相机不匹配，表中要注明。
- **LingBot-Depth**：
  - 内参按官方方式归一化（fx/W 等）。
  - 官方代码在把每个样本的 token 序列以列表形式送进 DINOv2 块时依赖 xFormers，而 `.venvs/baselines` 里没有装 xFormers（2026-10-17 第一次 preflight 因此失败）。**更正**：我当时说“torch 2.11 没有可用的 xFormers wheel”，这一点没有核实，而且是错的。PyPI 上的 xFormers 0.0.35 有 Windows wheel，要求 torch ≥ 2.10（2026-10-17 查）。它对应的 CUDA 版本是否与本机的 cu126 兼容，还没有验证。已生成的 6 个 LingBot 缓存，receipt 里记录的仍是旧的理由。核对办法：`scripts/windows/check_lingbot_xformers.ps1` 把 xFormers 0.0.35 装进 `.venvs/baselines`（`--no-deps`，不动 torch），在 30 张 NYU 和 30 张 HAMMER 上比较 xFormers 路径与缓存。判据在运行前写定：每张图主范围 C AbsRel 之差的最大值 < 1e-3，且像素相对差的合并 p99 < 0.5%，就认为可以忽略，保留现有缓存；否则用 xFormers 重做 6 个 LingBot 缓存。装上 xFormers 以后，CDM 和 Any2Full 的 DINOv2 注意力也会改走 xFormers；它们的缓存已经做完，以后如果重跑，需要注意这一点。改为把每个序列单独送进标准块：推理时这与 xFormers 的块对角注意力完全等价，只是注意力由 torch SDPA 计算，而不是 memory_efficient_attention。等价性由 `tests/test_lingbot_nested.py` 检验。
  - 官方默认 `apply_mask=True`，会把低置信像素设成 inf。我们设为 False，保留稠密输出，让所有方法在同一组像素上评分；mask 的占比记录在每张图的 report 里。
- **Any2Full**：与 `run_any2full.py` 一致（init_scailing、max_depth 1000、不开 denoise）。
- **Marigold-DC**：与 `python -m marigold_dc` 默认一致（bf16、DDIM trailing、50 步、集成 1 次、768 px、seed 2024）。

## 4. 评测人群与统计

- **人群**：val（样本外）和训练集留出场景（T 未拟合；HAMMER 为 scene10，NYU 为 crc32(scene)%5==0 的场景）。训练集其余场景不跑。缓存脚本和评测（`train_scope: holdout`）都只读这些样本。测试集不读。
- **指标**：与所有已有表格一致，包括主范围 C AbsRel、全图 C、干净区域 AbsRel（用来暴露过度屏蔽）、IR 和 DOR。预测无效的像素照实记为 NA，不丢弃。
- **预注册检验族**（`configs/baselines/holm_family_v1.yaml`，2026-10-16 在任何对比结果出来之前写定，decisions §18）：
  - 问题：同一模型加 T v1 掩码后，主范围 C AbsRel 是否下降。
  - 主族：val 上 6 个模型 × 2 个数据集 = 12 个检验，Holm 校正，α = 0.05。
  - 次族：训练留出场景，单独做 Holm 校正。
  - oracle_mask 与 raw 的比较只做描述，不做检验。Marigold-DC 不进这两个族。
  - 缺失或无效的比较按 p = 1 留在族里，族的大小不会因为看到结果而缩小。
- **重抽样单位**（`tp/eval/family.py`）：
  - NYU 有多个场景，用成对场景 bootstrap。
  - HAMMER 的 val 和留出各只有 1 个场景，场景 bootstrap 无法给出 p 值。这里改用场景内按轨迹的循环块 bootstrap：块长 8 帧，同一帧的三个传感器一起抽。
  - **它推断的只是这个场景里的帧，不能外推到新场景。**要做场景级推断，需要对 HAMMER 做留场景交叉拟合：T 按场景分折重训，让每个场景都在样本外。见第 7 节。

## 5. 已有结果：只有 PromptDA-L 一行（interim）

`python -m scripts.holm_family` 已用现有的 PromptDA-L 缓存跑过一次（`results/holm_plugin_v1_*`，`docs/holm_plugin_v1.md`）。其余 10 个比较还没有结果，按规则以 p = 1 计入族内。

| PromptDA-L，主范围 C AbsRel | raw | + T v1 掩码 | 差值 [95% CI] | p（Holm，12 个检验） |
|---|---:|---:|---|---:|
| NYU val（18 个场景） | 0.4669 | 0.0959 | −0.371 [−0.589, −0.173] | 0.011 |
| HAMMER val（1 个场景，帧块） | 0.0434 | 0.0340 | −0.0095 [−0.0139, −0.0051] | 0.0012 |
| NYU 留出（7 个场景，次族） | 0.7920 | 0.1650 | −0.627 [−1.044, −0.210] | 0.029 |
| HAMMER 留出（1 个场景，次族） | 0.0376 | 0.0292 | −0.0084 [−0.0121, −0.0046] | 0.0012 |

同时要看到一个代价，即过度屏蔽。检验族输出里的 clean 检查只做描述，不进检验族；所看的是主范围内本来正确的提示像素：

| PromptDA-L，干净像素 AbsRel | raw | + T v1 掩码 | 差值 [95% CI] | 按 decisions §18 判定 |
|---|---:|---:|---|---|
| NYU val | 0.0574 | 0.0838 | +0.026 [+0.0003, +0.061]（+46%） | 明显变差 |
| NYU 留出 | 0.0261 | 0.1455 | +0.119 [+0.006, +0.322]（+458%） | 明显变差 |
| HAMMER val | 0.0194 | 0.0153 | −0.004 [−0.008, −0.0004]（−21%） | 未变差 |
| HAMMER 留出 | 0.0115 | 0.0121 | +0.0007 [−0.0001, +0.0014]（+6%） | 未变差 |

按 decisions §18 的判据，PromptDA-L 在 NYU 上属于“污染区改善、干净区明显变差”，不能计为干净的成功。镜面上的干净像素很少，所以区间较宽。注意：这条判据写定时我已经看过 PromptDA-L 这一行的 clean 数字，所以对 PromptDA-L 来说它不算预注册；对其余 5 个模型仍然是事前写定的。这一点必须写进论文，不能只报 C 区域。

## 6. 运行方式（Windows，需要你执行）

1. `powershell -ExecutionPolicy Bypass -File scripts\windows\setup_baselines.ps1`
   - 新建 `.venvs\baselines`：使用 Anaconda 里的 torch，另装几个固定版本的包。
   - 克隆源码并检出固定 commit；下载权重约 30 GB，加 `-WithMarigold` 约 36 GB；逐个核对大小和 SHA256。
   - 可选 `-HfMirror https://hf-mirror.com`。
2. `powershell -ExecutionPolicy Bypass -File scripts\windows\step10_baselines.ps1`
   - GPU 部分：5 个模型 × 2 个数据集 × 3 种输入，每组先跑 3 张图的 preflight，再跑 val 和留出场景（每组约 450 张）。
   - CPU 部分：评测、汇总表、Holm 检验族。
   - 预计 GPU 时间 3–6 小时（未实测）。CDM 每个相机的模型首次加载约 1 分钟，每次运行都要核对约 16 GB 权重的哈希。
   - Marigold-DC 用 `-WithMarigold`，另需约 10 小时，建议放到租用的 GPU 上跑。
3. 失败的处理：某个模型失败（例如 OMNI-DC 在全分辨率下显存不够）时，把对应文件夹改名为 `_failed` 后告诉我，我再调整（例如给 OMNI-DC 加最大边长）。如果只想先跑一部分模型，用 `-Models priorda,cdm`。

## 7. 下一步

- 拿到结果后：
  - 汇总表和 Holm 结果更新到本文件和 progress。
  - 判定标准（预注册）：主族中至少 3 个“模型 × 数据集”组合在 Holm 校正后显著改善，并且干净区域没有明显变差，才算支持“T 可插拔”。
- HAMMER 留场景交叉拟合 T：5 折，每折约 30 分钟 GPU。完成后 HAMMER 的 10 个场景都在样本外，可以做场景级 bootstrap。这一步需要你同意后再跑。
- Mirror3DNet、GlassRecon 式对齐、LingBot-Depth 2.0（等有公开权重）。

## 来源

- PriorDA：https://github.com/SpatialVision/Prior-Depth-Anything ；https://huggingface.co/Rain729/Prior-Depth-Anything
- OMNI-DC：https://github.com/princeton-vl/OMNI-DC ；https://huggingface.co/zuoym15/OMNI-DC
- CDM：https://github.com/ByteDance-Seed/manip-as-in-sim-suite ；https://huggingface.co/depth-anything/camera-depth-model-d435
- LingBot-Depth：https://github.com/Robbyant/lingbot-depth
- Any2Full：https://arxiv.org/abs/2603.05711 ；https://github.com/zhiyuandaily/Any2Full
- Marigold-DC：https://github.com/prs-eth/Marigold-DC
