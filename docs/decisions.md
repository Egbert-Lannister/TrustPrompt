# 实施决策与核对记录

记录日期：2026-10-09（America/New_York）。原计划是方法与实验设计依据，不是已有完成证明。用户要求的哈希、版本和源码追溯优先于原计划 §7 的“不做哈希和封存”。

## 1. 范围、环境与日程

- TrustPrompt 是独立 Git 仓库，分支 `codex/minimal-loop`。GLARE 只读；原有未提交修改保留。
- 本机实测 RTX 4070 Laptop 8188 MiB；Python 3.12.7、torch 2.11.0+cu126、CUDA 可用。完整版本见 `environment.json`。先做 CPU 数据/评测闭环，再做少量 GPU 推理，不启动训练、云资源、邮件或协议申请。
- `.venv` 是项目自己的虚拟环境，目前通过 `--system-site-packages` 复用已安装科学计算库。首次 ensurepip 遇到 Windows 沙箱临时目录权限问题，采用 `--without-pip` 成功建立运行环境；没有更改全局依赖。这不是完全独立安装的依赖环境，所有实测包版本均被记录。
- 2026-10-09 查询 [CVPR 官方日期页](https://cvpr.thecvf.com/Conferences/2027/Dates)：注册 2026-11-10、提交 11-16、补充材料 11-23，均 AoE，与原计划一致。原计划的10-06至10-08工作不视为已完成；按依赖推进。
- 原计划提到“H节修订划分”，但该计划文件并无H节。实际证据来自旧项目数据审计 manifest 的 `source_index_audit`；将经源元数据再次核验的映射单独纳入 `configs/splits/mirror3d_nyu_v1.json`。

## 2. 数据源与编号

- E盘挂载后，普通沙箱不能读；只读访问通过自动审批执行。已有完整 `nyu.zip`、CSV和 selected H5，完整NYU MAT不存在，`.mat.part` 不作为数据源。
- 原 selected H5 无 `rawDepths`。仅在 URL/size/ETag/Last-Modified 相等、`raw_slices_unchanged=1`、精确索引presence=True时复用原始切片。旧预处理缓存、预测与N−1 SampleRecord均不复用。
- 独立 HTTP Range 脚本从官方MAT下载必要 metadata/rawDepths/labels，限100MiB。本次只下载30,408,705字节，生成5帧14,064,714字节独立cache。身份、命令、每数组hash、下载次数见同名JSON。
- 实测标注N对应源 `images[N]/rawDepths[N]/depths[N]/labels[N]`；官方export命名N−1是发布标注与脚本不一致。21帧613,095像素回归复现：N有99.87147179%在平面1mm内，N−1为0.07992236%。局部异常像素不剔除，不替换GT。阈值是总样本回归阈值，不要求每帧>99%。
- 原CSV有58/125错误scene，不能直接分组。官方MAT全125个对应scene复核零冲突。历史曝光场景排除测试，固定55/18/23 train/val/test场景（76/19/30帧）；本次仅选5个不同train场景，未读test深度。划分hash进入实验记录。

## 3. 深度、内参、真值与材质

- 所有数组统一裁剪 `y=12:468,x=16:624`，输出456×608。`rawDepths` 已配准到RGB网格、米、未补洞；原值保留，有限且 `0<Z<=10m` 有效。官方工具箱maxDepth=10且截顶，因此10m可为饱和值，不证明真实距离恰好10m。
- GT严格按 `uint16(float32(depths)*1000)+delta mod 65536` 重建，再转米。不能将delta加到rawDepths上。编号回归直接用整数毫米比较，避免float米往返改变1mm阈值。
- 精修镜面与背景来源不同：镜面是作者平面修正；背景来自NYU补洞，不是独立第一表面物理测量。全图结果标 `mirror_refined_background_inpainted`，另存 `precise_mirror_only` 结果。污染定义在两类参考上有不同科学可信程度，本次只作工程诊断。
- 官方工具箱RGB去畸变与投影到MATLAB数组的主点约定差1px；且写回的Z来自传感器，而非严格变换后的RGB相机Z。当前保留发布像素网格不再平移，公共K使用depthOut投影分支（裁剪后cx=308.5824494,cy=240.7361663）作为近似。`strict_rgb_z_qualified=False`随样本、逐图、汇总和receipt传递，正式RGB-Z主表模式拒绝未认证样本。具体源码证据见 `intrinsics_audit.md`。
- Mirror3D平面回归使用作者自己的近似K（519,519,304,228），不混用物理RGB K。均为轴向深度，未改为射线距离、未做尺度对齐。
- 894个对象语义类不等于材质；直接排除mirror/window/glass三个类会将其他玻璃/金属对象误标普通。第一阶段仅精确镜面=2，其余未知=0，保留原语义在源cache供后续审查。材质不参与方法推理和污染区域分母。

## 4. 评测公式解释

- `C = valid_prompt & valid_gt & (abs(P-GT)/GT > tau)`，tau主值0.05。Cfar/Cnear按符号分；hole是无有效提示但GT有效。所有区域基于原始未补洞提示，所有方法共享。
- 原文未具体定义boundary：采用C的四邻接内边界（图外视为False），再按Chebyshev半径band=5方形膨胀，最后与GT有效域相交。band可与C/hole重叠；clean去除band。此定义固定并有手算回归。
- log-RMSE使用自然对数；delta为严格小于1.05/1.10/1.25；IR按原文固定5%条件；DOR按严格 `pred>1.10*GT`，分母C∪band。
- 空域JSON为null且status=empty，表中NA。无效预测保留分母，delta计失败；连续误差、IR/DOR在存在无效预测时为null/status=invalid_predictions，不静默只算有效像素；IR/DOR附上下界。聚合时传播这种无效结果，只有真正空域可排除，并记录数量。
- 逐图等权→场景内均值→非空场景等权，不做像素加权。样本唯一键含dataset/sensor/scene/frame，scene split跨传感器共用，避免HAMMER未来三传感器冲突。
- §3.2 clean因band被去除，与§5.2监督标签有歧义。可信度正类采用所有有效提示上的rel<=tau（clean_all），不把正确但靠近污染边界的像素标成错误。AUPR用average precision，同分整组统计；风险覆盖曲线不随同分像素排列改变。
- 只有一个方法时不伪造配对bootstrap；函数已实现10,000次场景级配对及Holm，待真实对照启用。原始提示不输出可信度，AUROC/AP记not_provided，不能用GT构造w冒充模型输出。
- `raw_prompt` 在提示原生格上EDT最近邻填洞，再中心采样最近邻缩放；全空提示输出NaN并按无效预测协议评估。数据Sample本身从不填洞。

## 5. 后续方法中的计划冲突

- PromptDA §5.4声称“三处改动零初始化与原模型相等”，与提示重写/分位数改变数值的文字不兼容。将来只对“相同输入、相同归一化/插值，仅新增零权重可信度通道”验证逐值相等；提示重写、鲁棒归一化、加权插值分别开关消融，不能声称全系统零初始化等价。
- “可信度可以直接当不确定度”尚待验证与校准；本阶段不作此结论。
- UniDepth先做RGB-only原生推理，无GT、提示、材质作为模型输入；本地可用的是V2-S，不冒称V2-L。PromptDA/HAMMER等代码事实与许可在真正接入时再核，不将计划陈述当已验证。
- 鲁棒对齐是提示驱动的免训练基线；其逆深度拟合残差阈值与评测深度rel阈值分开记录，不能调测试结果选参数。
- 已实施鲁棒参数：1000次seeded RANSAC；`abs(s*m+t-1/P)/(1/P)<=0.05`为拟合内点条件；Huber同域delta=0.05、最多20次IRLS。每个拟合支持均匀最多抽4096点，至少50点。局部8×8网格/3×3邻块支持，系数在块中心双线性插值、边缘常值延拓。MAD为相对逆深度残差的未乘1.4826中位绝对偏差，不冒称高斯标准差。
- 全局与局部失败不通过正值clamp修饰；不支持/非正预测为NaN进入固定分母协议。本次真实局部外推失败已记入progress，保留原数据与结果。
- 基线cache读取时间不是推理时间：逐图结果使用evaluation_prediction_seconds/timing_component，原UniDepth推理时间单独保留，未测端到端组合速度。旧cache没有新images索引digest时，必须逐项绑定receipt中的样本/RGBhash/scene/split及frame_count；新cache额外记录index SHA256和sensor。

## 6. 状态与可追溯性

- 原计划副本不修改内部陈述；本文件记录采用解释。`reuse_provenance.json`给实际旧文件hash和Git HEAD，旧树有未提交修改，不能只用HEAD代表旧文件。
- 每次评测保存配置、seed、命令、split manifest、源码hash、源码zip、Git HEAD/status/diff、输入/输出hash、日志、逐图和场景指标。当前早期实验在未提交代码上运行，source snapshot可还原其代码。
- `代码已实现`、`小规模验证通过`、`正式实验完成`分开；当前无正式训练/最终测试结果。未执行邮件、协议接受、数据上传、付费资源或长训练。

## 7. 计划修订 v2（2026-10-09）

完整列表见 `plan_revision_v2.md`。本节只记录实现时作出的具体选择。

- R4 逐块对齐 v2 的 ε 取 1e-6 × 支持集逆提示中位数：只依赖传感器，固定后不再调；回退以像素为单位，而不是整块，以免把有效的局部拟合一起丢掉。全局也非正时保持 NaN 并计入无效预测。
- R5 Gate 的 τ_gate = 0.05，比较量为深度相对误差 |P−D_a|/D_a（与评测 τ 同口径，与对齐内点阈值的逆深度口径不同，两者分开记录）。w 只在有效提示处定义；可信度指标沿用 clean_all 正类。
- R3 RGB-Z 上界：反解公式 `Z_rgb = (Z_d + c·T)/(c·r)`，c 为 R⁻¹ 的第三行；外参只从哈希校验过的官方文件解析，平移分量对应哪个轴由 `depth_world2rgb_world.m` 的赋值语句决定，不凭记忆假设。阈值 0.5%。
- test 锁：`run_eval` 在 `experiment_role != formal_main_result` 时拒绝 `split=test`，在创建输出目录前就报错。
- 跨平台：Linux 工作区与 Windows 的运行写入不同输出目录，receipt 中带 `execution_platform` 与解释器路径。

## 8. PromptDA 代码事实核对（原计划 §0；第 2 步前置）

2026-10-09 读取官方仓库 `DepthAnything/PromptDA` main 分支，提交 `9161547ca7151e14fcc8f490ad3cc02fa9437483`（2026-01-29）。只读源码，未下载权重、未运行。

| 文件 | SHA256 |
|---|---|
| promptda/promptda.py | `bc0ca4bb4ec61106f01f5da514ea25c45cd652c0b526409c3eef161aa14ea7b6` |
| promptda/model/blocks.py | `ca0b2b2463d4c8186ff6a7e4595ac293bb63e8316f31e7b5dba9b14c4c0f832b` |
| promptda/model/dpt.py | `446a7a43cb1d2a64a15e9244b6f6dfc1c680364daed0e17ba8763b43279ad962` |
| LICENSE | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |

- 许可：Apache-2.0。核实。
- 归一化：`normalize` 对整张提示（含 0 值空洞）取 `torch.quantile(..., 0.)` 与 `quantile(..., 1.)`，即 min/max；`denormalize` 用同一对值反变换。核实。
- 输出：`nclass=1` 分支 `output_conv2` 末端为 `nn.Sigmoid()`（`output_act='sigmoid'`），故输出被限制在 `[min(P), max(P)]`。核实。
- 注入：每个融合块的 `resConfUnit_depth` = Conv2d(1,F,3) → act → Conv2d(F,F,3) → act → 零初始化 Conv2d(F,F,3)；提示用 `F.interpolate(mode='bilinear', align_corners=False)` 缩放到该层分辨率。核实，与原计划 §0 描述一致。
- 推论：空洞未填时 min=0；镜面虚像若是全图最远值会决定 max。原计划“空洞先最近邻填充”的预处理是必要的，基线表注明。
- 未核实：PromptDA-S-Transparent 权重的训练细节（HAMMER、10K 步）与 HuggingFace 上的文件，需在 Windows 下载权重时核对模型卡。

## 9. NYU 外参的符号约定（R3 实施）

- 官方 `camera_params.m`：`R = -[9 个数]`，`R = reshape(R,[3 3])`（MATLAB 列优先），`R = inv(R')`；`depth_world2rgb_world.m`：`T = [t_x; t_z; t_y]`，`X_rgb = R·X_d + T`。解析严格按文件语句回放，未知语句报错。
- 得到的 R 行列式为 −1（约等于 −I），不是旋转。由于 `rgb_world2rgb_plane` 用 X/Z、Y/Z 投影，对所有坐标取负投影不变，工具箱的像素位置与 (−R, −T) 完全一致。上界计算采用 (−R, −T)，并在每个有效像素检查 Z_rgb > 0；若符号约定错误会直接报错。
- 5 帧训练集最坏差 0.473%（阈值 0.5%），判定为有界；结论只覆盖这 5 帧，95 帧开发集到位后重跑并以其为准。

## 10. 填充上界之后的方向（2026-10-12，已由 R14 取代）

依据 `docs/fill_analysis.md`：大面积近平面区域用模块 C 填充，小物体改为屏蔽提示后交给 PromptDA；选择规则 v2 尝试环带留出验证替代 BIC。阈值只在 train/val 上选。T v0 优先级不变。

## 11. R14 实现中的选择（2026-10-12）

- **DEPTHOR++ 检测器复现**（论文无代码）：公式与参数按 arXiv 2509.26498v1 原文（δ=0.5、α=15、p=0.04、k=40、u=0.9，γ 断点 0.95/0.85）。偏差如下，均写入每次运行的记录：
  - 论文针对稀疏 dToF 点，算全部点对。我们的提示是稠密的，所以在步长网格上取 ≤ 30,000 个点，对 j 的平均用 4,096 个随机参考点估计，逐像素结果取最近的有效网格点。
  - 相对深度 r 用已缓存的 UniDepthV2-S 深度。论文用 DAv2-Small，并且没有说明 r 是否取过倒数；符号项要求 r 与深度同序。
  - 坐标归一化为 (x/W, y/H)；Otsu 用 256 个 bin。
  - 每个样本的随机种子只取决于 detector.seed 和样本键，所以评测和 PromptDA 缓存得到同一个掩码。
- **检测器的已知行为**（单元测试记录）：3% 面积的虚像块 γ 仍 > 0.95，被门控整体跳过，尽管分数能区分出来；19% 面积时全局排序项会让干净点的排序也偏移，误检升高。NYU val 冒烟中，10/19 帧被门控跳过，触发 Otsu 的帧有时剔除大部分提示。
- **选择规则**：AIC、GRIC（Torr 式，σ 取最一般可用候选的 1.4826·MAD，截断 2，惩罚 k·log 3n；λ1·d·n 对所有候选相同，省略）、扇区留出（区域质心为中心分 8 个扇区，误差取留出扇区上逆深度相对误差的中位数，再对各折取平均）、径向留出（外半环拟合，内半环 Chebyshev ≤ ring//2 验证）。所有规则共用 k3 内部平面性检查，保证与 v1 可比。
- **检测器可信度的量化**：评测前把 1/(1+A) 四舍五入到 1e-3，再计算 trust_quality。AUROC 只受同值并列影响，可忽略；风险-覆盖率曲线最多 1001 个点。
- **选择遗憾诊断**：评测时把逐区域各候选的 GT 误差写入记录（`diag_gt`），只用于计算 oracle-match 和 regret；规则本身不读 GT，测试确认加入后输出完全一致。

## 12. T v0 实现中的选择（2026-10-13）

- **编码器**：用 PromptDA-S 检查点里的 ViT-S/14 编码器，即经 Depth Anything V2 微调的 DINOv2-S，没有单独下载原版 DINOv2-S。理由：权重已在本地、加载路径已验证、许可同为 Apache-2.0，且不新增依赖。这是对计划“DINOv2-S”的偏差，记录在每次特征缓存的 receipt 中。特征通过 forward hook 取自编码器最后一个 LayerNorm（按结构定位），取最深一层的 patch token，自适应平均池化到 24×32。提示只影响解码器，不影响这些 token。
- **分辨率**：T 在 PromptDA 提示网格 192×256 上工作，所有数组都用 resize_nearest（中心网格）采样，与评测一致。消费方用中心对齐的双线性插值把可信度上采样回原分辨率。
- **标签**：0 = 干净（|P−GT|/GT ≤ τ），1 = 污染，255 = 忽略（无效提示或无效 GT），与评测区域定义一致。模拟样本由模拟器的 type 标签映射得到（远/近 → 1，空洞 → 忽略）。
- **训练集与选择**：只用 train split。checkpoint 和二值阈值在 train split 的留出场景上选择（只用真实提示）：HAMMER scene10，NYU 按场景名 crc32 % 5 == 0 留出约五分之一。val 不参与训练和选择，保持为干净的开发报告。T 在 train split 上的评测属于样本内，汇总表单独标注。
- **四种训练方式**：real、sim、real_sim、real_sim_nofeat（不用编码器特征的消融）。下游 PromptDA 运行只用留出场景 AUROC 最高的那一种，自动选择，不看 val。
- **损失与采样**：BCE，pos_weight = 干净/污染像素比（限制在 1–20）；两个数据集等权采样；水平翻转同时作用于输入、标签和特征网格（特征翻转是近似）。
- **Gate + PromptDA**：D_g 用 PromptDA 缓存配置里的种子（20261011）对齐，与评测里 Gate 所用的 D_g（种子 20261009 / NYU）可能有 RANSAC 层面的微小差异，记录在案。

## 13. T v1：对准目标的选择（2026-10-13）

依据 `docs/t_v0_analysis.md`：T v0 误标少但漏检多，而漏检和误标的代价不对称。
- **选择目标**：替换误差。把被标记的提示像素换成 D_g（k0 填充），其余保留提示，计算 |D−GT|/GT 的均值，在主范围（NYU 镜面 / HAMMER 物体，模拟样本再加上模拟污染区域）和全图上各算一次，每张图取两者平均，再按数据集等权平均。checkpoint、阈值（0.05–0.95）、是否后处理三者联合选择，版本也按这个目标选；只用 train 留出场景，不看 val。
- **代理的局限**（2026-10-14 更正）：用 D_g 代替实际的下游补全，会**高估**误标的代价，因为 PromptDA 补得比 D_g 好；被误标的干净像素在代理里要付出 D_g 的误差（约 4–10%），这和许多刚超过 τ 的污染像素的误差相当。实际结果是选择偏向少标（阈值 0.10–0.25），与我原先的预期（偏向多标）相反。见 `docs/t_v1_analysis.md`。
- **后处理**：对标记区域做 3×3 闭运算并填补内部孔洞，只作用于有效提示像素。是否启用在留出场景上选择，并直接写进缓存的可信度图（新增像素的 trust 取 min(trust, 阈值/2)），所以下游不用改。
- **特征正则化**：对投影后的编码器特征加 Dropout2d(0.3)，训练时每个样本以 0.2 的概率整幅置零。不用特征的版本保留为消融。
- **训练数据 v1**：在 v0 的基础上加存 GT、GT 有效掩码和主范围掩码，这些只用于选择，不作为网络输入。

## 14. R15：提示修复与高召回阈值（2026-10-14，在看到下游结果之前确定）

- **提示修复**（PromptDA 变体 repair_prior）：被屏蔽的提示像素不再用 D_g 填，而是用模块 C 的填充：以 D_g 为底图，在可信环带上拟合 k0–k3，按 v1 选择。然后照常交给 PromptDA-L。掩码来源有两种：GT 污染掩码（oracle，用来测上界）或 T 缓存。提示层面的检查（NYU val 17 帧镜面像素）：修复后误差 0.089，D_g 填充为 0.140。
- **高召回阈值**：在 train 留出场景上，选主范围召回 ≥ 0.95 且误标最少的阈值；如果达到 0.95 召回要误标 10% 以上的干净像素，就保留模型阈值。NYU 为 0.67（召回 0.952，误标 5.8%）；HAMMER 达到 0.95 召回需要误标 60%，因此保留 0.25。结果见 `results/trust_t_v1_highrecall_threshold.json`，由 `scripts/select_recall_threshold.py` 生成，val 未参与。

## 15. 数据从 E 盘迁回项目目录（2026-10-14，用户要求）

- 新位置：`cache/`（HAMMER PromptDA 缓存，与 NYU 缓存并列，没有重名）、`checkpoints/promptda`、`external/PromptDA`（已加入 .gitignore）、`data/source_cache/hammer_subset_v1`、`data/source_cache/mirror3d`（nyu.csv / nyu.zip，从 GLARE 的数据目录只读复制，GLARE 不改动）。
- 方式：robocopy 复制，不移动、不删除 E 盘上的任何东西。校验三项：每个文件的大小一致；每个缓存预测文件的 SHA256 与写入时 images.json 记录的一致；三个检查点的 SHA256 与之前运行记录的一致。最后由 restore_promptda.ps1 确认源码在审计过的提交上。
- 配置和脚本里的路径全部改为项目内路径。旧运行记录（receipt）里的 E 盘路径保持原样，作为历史。缓存读取只用 predictions/ 下的文件名，所以移动后旧缓存仍可直接使用，receipt 的哈希也不变。
- `fetch_nyu_subset.py` 里可选的 GLARE 旧 h5 默认路径仍指向 E 盘；该文件不存在时会自动跳过，不影响运行。

## 16. R16 中的协议核查与定义（2026-10-15）

- **NYU 划分**：`configs/splits/mirror3d_nyu_v1.json` 来自 GLARE 的源索引审计（全部 125 张图重新核验），把历史上用过的场景都放在测试之外，按场景划分为 55/18/23（76/19/30 帧）。它不是 Mirror3D 官方划分（73/52 张图）。因此开发集与官方测试集有重叠，我们的 NYU 结果不能和按官方协议报告的数字直接比较。如需官方协议对比，必须排除官方测试帧后重训 T，并在官方 52 张图上评测。
- **编码器权重**：`configs/trust/encoder_{nyu,hammer}.yaml` 使用的是 `prompt-depth-anything-vits`（普通 PromptDA-S），不是在 HAMMER 上微调过 10K 步的 `vits-transparent`。T 的特征不含 HAMMER 信息。
- **conformal 风险的定义**（R16 第 2 项）：对每张图，loss = 主范围内“测得更远”的污染提示像素（P > GT·(1+τ)，τ = 0.05）中，没有被标记（trust ≥ λ）的比例；没有这类像素时 loss = 0。loss ∈ [0, 1]，随 λ 增大单调不增。按 conformal risk control（Angelopoulos 等，2022）取 λ̂ = 满足 (n/(n+1))·R̂(λ) + 1/(n+1) ≤ α 的最小 λ。在图像可交换的前提下，保证测试图像的期望 loss ≤ α。HAMMER 的校准集只有 scene10，与 val 的 scene11 是否可交换存疑，报告时要注明。在 192×256 网格上计算。

## 17. T v2 判据不成立后的决定（2026-10-16）

- 预设判据（运行前写定）：按 T v2 方向截断，要在 HAMMER val 和留出场景上都拿到真实方向截断收益的大部分。实际只拿到 20–40%，不成立。
- q\* = h/(g+h) 置信度阈值是看到 argmax 结果之后设计的。它只用训练拟合场景估计 g、h，但仍然只算诊断，不能作为方法改进报告；它也没有满足判据。
- 不租 GPU 训练带约束通道的补全模型。重新考虑的条件是出现新证据，例如在非平面真值数据上，GT 掩码 + 真实方向截断的上限明显大于约 5%。
- 网格分解（`scripts/cap_decomposition.py`）用 192×256 网格最近邻采样，只用于解释全分辨率结果，不替代 `docs/bound_results.md`。

