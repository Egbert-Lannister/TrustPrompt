# TrustPrompt 计划修订 v2

> 修订日期：2026-10-09（America/New_York）。基线文档：`docs/implementation_plan.md`（原文 SHA256 `1ca68455…0522448`，原文不改）。
> 本文件只记录**相对原计划的改动**；未提到的部分（任务定义、C1–C3、模块 A–D 结构、§3 评测协议、§11 改进顺序）继续以原计划为准。
> 每条改动注明依据。“依据”指本仓库内已运行并保存的证据，不指计划中的预期。
> 状态标记：〔已实现〕代码存在且测试通过；〔小规模验证〕真实数据 smoke 通过；〔正式〕正式实验完成；〔待办〕未开始。

## 0. 修订摘要

| # | 原计划 | 修订 | 依据 |
|---|---|---|---|
| R1 | 两名学生 + 1×A100 开发，逐日日程从 10-06 起 | 人员保留；GPU 分本地 4070 8GB 与按阶段审批的云端两层；日程从 10-09 重排，方法冻结 10-28 | 实测环境 `docs/environment.json`；进度约落后 3 天 |
| R2 | NYU 报 C/clean/hole/band/all | NYU 主指标只用镜面区域；clean 只作“偏离传感器”代理；NYU 不报 hole | 背景 GT 由同一 rawDepths 补洞得到，raw_prompt clean AbsRel=0.0011 是循环比较（`results/mirror3d_raw_prompt_smoke_v2`） |
| R3 | 深度一律为 RGB 相机轴向 Z | NYU 计算“传感器 Z 与 RGB Z”的逐像素差上界；上界 ≤ 0.5% 时按“配准 Z，误差有界”入主表并加脚注 | `docs/intrinsics_audit.md`：project_depth_map.m 写入变换前的深度 |
| R4 | 逐块对齐：(s,t) 双线性插值，块内点不足继承全局 | 增加 v2：插值后逆深度 ≤ ε 的像素回退全局解，并单独报告回退比例；v1 结果保留 | `results/mirror3d_robust_local_smoke_v2`：1,454 个非正逆深度 |
| R5 | RobustAlign 基线 = 对齐先验替换全图 | 增加 **RobustAlign-Gate**：提示与对齐结果一致处保留提示，否则用对齐先验。作为 C3 必须超过的免训练强基线 | 全局对齐 C AbsRel 0.116，但 clean 从 0.001 升到 0.039 |
| R6 | 先接全部基线和数据集，再做 T | **先做 C1**：PromptDA 在 NYU 开发集上的继承现象、GT 掩码屏蔽上界、Gate 基线，三组数字先出 | C1 是论文前提，成本最低 |
| R7 | 4 折留一传感器（结构光 / 主动立体 / ToF / iPhone） | 核心折 = HAMMER 内 d435 / l515 / helios 三折；NYU(Kinect)、ClearGrasp(RealSense) 作跨数据集纯测试；iPhone 折在 ARKitScenes + 自采数据到位后加入 | 本机无 HAMMER/ARKitScenes；MP3D/ScanNet 协议未提交 |
| R8 | §5.4 三处改动零初始化逐值相等 | 只有“可信度通道注入”保证逐值相等；提示重写、加权插值、鲁棒归一化作为独立开关消融 | `docs/decisions.md` §5 |
| R9 | NYU labels 中非镜/窗/玻璃类 → 材质 1 | 非镜面一律材质 0（未知）；材质不参与推理和区域分母 | `docs/decisions.md` §3 |
| R10 | §7.5 不做哈希和封存 | 每次运行保存源码哈希、快照、输入输出哈希 | 用户要求；已实现于 `tp/provenance.py` |
| R11 | 引用“H 节修订划分” | 原文无 H 节；以 `configs/splits/mirror3d_nyu_v1.json`（55/18/23 场景，76/19/30 帧）为准 | `docs/decisions.md` §1 |
| R12 | UniDepthV2-L 作度量先验消融 | 开发期用本地 UniDepthV2-S；L 与 DA3/MoGe-2/Depth Pro 在云端跑一次进基线表 | 本地只有 V2-S 权重 |

## 1. 执行环境（R1）

实测三类执行位置，各自能做的事不同：

| 位置 | 实测条件 | 负责 |
|---|---|---|
| Claude 在用户电脑上的 Linux 工作区（挂载 TrustPrompt/GLARE） | 2 核、3 GB 内存、无 GPU、无 torch；可达 PyPI/GitHub，**不可达** NYU、HuggingFace；看不到 E 盘 | 代码、测试、CPU 评测（读已有 npz/cache）、文档、提交 |
| 用户 Windows（RTX 4070 Laptop 8 GB，D:\Anaconda，项目 `.venv`） | GPU 推理、可访问 NYU 与 E 盘 | 数据下载、UniDepth/PromptDA 推理缓存、DINOv2 特征缓存、T 的小规模训练 |
| 云 GPU（按 §6 阶段审批） | 待开通 | 大模型基线全量推理、PromptDA 微调、批量消融 |

规则：Windows 和云端运行由团队按 README 中的命令执行；输出写回 TrustPrompt 后，评测、汇总与核查由 Claude 在工作区完成。所有 receipt 记录实际 `sys.executable` 与平台，跨平台结果不混写进同一目录。

## 2. 数据与评测口径修订

### 2.1 Mirror3D-NYU（R2、R3、R11）

- 划分固定：train 55 场景/76 帧，val 18/19，test 23/30。**test 深度在方法冻结前不读取。**
- 开发集：train+val 共 95 帧。超参数与模型选择只看 val。
- 主指标区域：`C ∩ mirror`、`Cfar ∩ mirror`、`band ∩ mirror`（GT 为 Mirror3D 平面精修，独立于传感器）。
- `clean` 指标保留，命名改为 `clean_vs_inpainted`，只用来衡量“方法偏离传感器多少”，不作为 GT 精度。NYU 不报 `hole`。
- RGB-Z（R3）：用 NYU 工具箱 `camera_params.m` 的外参，对每个有效提示像素精确求解 `Z_rgb`，统计 `|Z_rgb − Z_raw| / Z_raw` 的最大值与 99.9 分位。
  - 若 ≤ 0.5%（τ=5% 的 1/10）：样本标 `rgb_z_bounded=True`，允许进入主表，表注写明上界；
  - 若 > 0.5%：用求得的 `Z_rgb` 重写提示和 GT 网格（同一变换作用于两者，不按方法区别对待），再评测；
  - 两种情况都不移动像素网格；1 px 主点歧义另作敏感性分析（主点 ±1 px 下结果变化）。
- 30 帧测试集统计力有限：NYU 结论写成“描述性 + 场景级配对 CI”，不作为唯一镜面证据。MP3D/ScanNet 获批后作为镜面主测试。

### 2.2 HAMMER（R7）

- 许可已核实：MIT（数据集页面，2026-10-09）。原计划“许可未写明、邮件作者”一项取消。
- 整包体积超出本地存储（用户反馈，2026-10-10）。改为远程选择性获取：`scripts/hammer_remote.py` 先只读 ZIP 中央目录得到文件清单与大小，再按场景、传感器、帧间隔只下载所需文件；子集规则写入 `configs/hammer_subset_v*.yaml` 并进入实验记录。若子集仍超出本地容量，放到外接盘或云存储（G1 阶段），不复制到 GLARE。
- 接入前核对：目录与 naked 配对方式（原计划 §2.2 的 [核对] 项不变），以 index 结果为准。
- 审计结论（`docs/hammer_audit.md`，2026-10-11）：
  - 主口径改为 **objects_only**（GT 与 naked GT 差 > 1 mm 的物体像素），与 NYU 的 mirror_only 对应；全图结果保留但注明混入 D435 背景偏差（+2–4%）与 Helios 暗面误差。
  - HAMMER 无材质标签：原计划“透明→3、反光→4”的映射无法从数据获得。分材质分析与“GT 材质掩码 + PromptDA”基线依赖人工逐实例标注（约 100 个实例），标注表版本化后再用；在此之前该基线改用“物体掩码”（GT vs naked）版本，并在表中注明。
  - 开发阶段物化 train 每 20 帧 + val 每 5 帧；训练直接读源数据。
- 划分：官方 scene2–11 训练、12–14 测试。**新增 val**：训练场景中编号最大的一个（预定 scene11，以实际目录为准）作 val，T 与填充的超参数只在 val 上选。
- PromptDA-S-Transparent 见过 HAMMER，只在 HAMMER test 上与之比较并注明。

### 2.3 其余数据集

- ClearGrasp real：本机 E 盘已有目录，优先接入，作纯测试集（R7 中的 RealSense 跨数据集测试）。
- ARKitScenes upsampling：先核对体积，只下载 val 子集中的若干场景做 iPhone 噪声拟合与 iPhone 折测试。
- Hypersim：只下载含镜子/玻璃/窗户类别的场景子集（按语义标签筛选），不全量下载。
- GlassRecon：保持原计划，纯测试，优先级低于以上三项。
- MP3D/ScanNet：**需要用户今天提交使用协议**；获批前代码照写，不阻塞其他工作。
- 自采 iPhone：按原计划 §8 执行，人员已确认；10-20 前完成第一批 6–8 个场景，便于早期验证真值构造脚本。

## 3. 方法与基线修订

### 3.1 模块 A：逐块对齐 v2（R4）

- v1 保持不变，作为已记录的失败版本保留。
- v2：在 v1 输出上，对插值后逆深度 `s(x)·m(x)+t(x) ≤ ε` 或非有限的像素，改用全局模型；全局也非正时保持 NaN 并计为无效预测。ε = 1e-6 × 全局逆深度中位数（固定，不调）。
- 诊断中记录：回退像素数、回退比例、哪些块导致外推。表中 v1 与 v2 并列。

### 3.2 RobustAlign-Gate（R5）

- 定义：`D = P_up`，若提示有效且 `|P_up − D_a| / D_a ≤ τ_gate`；否则 `D = D_a`。`D_a` 为全局或逐块 v2 对齐先验。
- `τ_gate` 固定为 0.05（与对齐内点阈值同一约定，不在测试上调）；val 上的 0.02/0.10 只作敏感性报告。
- 输出 `w_gate = 1[|P−D_a|/D_a ≤ τ_gate]`，按 §3.2 计算可信度 AUROC/AP，作为 T 的直接对照。
- 它对应“没有学习的 T + 只有 k0 档的填充”。T3 中“去掉 T”的消融即为此基线。

### 3.3 先做 C1（R6）

在 NYU 开发集上，按顺序出：
1. PromptDA-S、PromptDA-L（冻结，官方推理流程，提示空洞最近邻填充）的 C/IR/DOR；
2. PromptDA + GT 镜面掩码屏蔽（掩码内用对齐先验填充）——上界；
3. RobustAlign-Gate；
4. PriorDA v1.1 及其 coarse_only。

同时核对原计划 §0 中 PromptDA 的三条代码事实（min/max 归一化、sigmoid 输出范围、bilinear 注入），把核对结果写进 decisions.md。

### 3.4 其他

- PromptDA 改造（R8）：零初始化等价测试只针对通道注入。
- 训练：T 在本地 4070 上用缓存特征训练（参数量 5–8M，缓存后无需大模型前向）；PromptDA-L LoRA 微调只在云端。若云端预算未批，降级为 PromptDA-S 本地微调并在论文中如实写明。

## 3.5 C1 开发诊断后的修订（R13，2026-10-11）

依据 `docs/c1_analysis.md`：
- C1 限定为“提示式补全对**大面积、连贯**错误提示的继承”。NYU 镜面上 PromptDA-L 显著差于免训练对齐；HAMMER 小物体上 PromptDA-L 反而最好。F1（继承率 vs 污染面积）升为核心图。
- 数据优先：大面积污染（镜子、玻璃门/窗）的真实数据——自采 iPhone 优先，MP3D/ScanNet 镜面子集获批后加入。
- 方法顺序：先在 CPU 上用神谕掩码评估填充候选 k0–k3 的上界（模块 C），再训练 T。NYU 结果显示神谕掩码 + k0 填充 ≈ 直接输出 D_g，镜面上的收益主要取决于填充质量。

## 4. 新增检查点

原计划“不设继续/停止门”保持不变：以下检查点**不自动停止任何工作**，只在到点时把事实交给用户决定是否调整范围。

| 日期 | 检查内容 | 若不满足，交给用户决定的选项 |
|---|---|---|
| 10-14 | HAMMER 是否已下载并通过 3–5 帧对齐检查 | 推迟 LOSO 到 ICCV / 改用 ClearGrasp + NYU 作主测试 |
| 10-16 | C1：PromptDA 在 NYU 开发集 C 区域是否继承（IR 显著 > 0） | 调整 C1 表述 |
| 10-21 | 端到端（T + 填充 + 冻结 PromptDA）在 val 上 C 区域是否优于 Gate 基线 | 加大 T 投入 / 转投 ICCV 2027 |
| 10-28 | 方法冻结 | 之后只跑测试和消融 |

## 5. 日程（从 10-09 起）

| 时间 | 工作 | 执行位置 |
|---|---|---|
| 10-09–10-11 | 第 1 步：RGB-Z 上界、对齐 v2、Gate 基线、NYU 开发集扩到 95 帧；提交 MP3D/ScanNet 协议；核对 HAMMER/ARKitScenes 许可与体积 | 工作区 + Windows |
| 10-11–10-14 | PromptDA-S/L 推理接入（1→5→95 帧）；HAMMER 下载与 loader；ClearGrasp loader | Windows + 工作区 |
| 10-13–10-16 | C1 现象表（NYU 开发集 + HAMMER train/val）；云阶段 G1 基线推理 | 云 + 工作区 |
| 10-13–10-18 | 模拟器 v0（镜面渲染、穿透、金属、传感器噪声）；DINOv2-S 特征缓存 | 工作区 + Windows |
| 10-17–10-21 | T v0 训练；填充 v1；提示重写 → 冻结 PromptDA，得到首个端到端数字 | Windows / 云 G2 |
| 10-21–10-28 | TrustPromptDA 微调；HAMMER 三折 LOSO；填充 v2 | 云 G3 |
| 10-28 | 方法与超参数冻结 | — |
| 10-29–11-08 | 测试集一次性评测、消融、F1–F4、写作 | 云 + 工作区 |
| 11-10 / 11-16 / 11-23 | 注册 / 截稿 / 补充材料 | — |

## 6. 云端预算申请（待用户审批）〔估算〕

单价按 A100 80GB 按需 1.5–2.5 美元/小时估算，实际以选定供应商报价为准。按阶段申请，每一阶段结束后报告实际用量和结果，再申请下一阶段。

| 阶段 | 内容 | GPU 小时 | 上限金额 |
|---|---|---:|---:|
| G1 | 基线全量推理（PromptDA-L/S/S-T、PriorDA、UniDepthV2-L、DA3、MoGe-2、Depth Pro）在 NYU 开发集 + HAMMER + ClearGrasp；先验/DINO 缓存 | 40–60 | 150 美元 |
| G2 | T 训练与扫参（约 15 次，每次 2–4 h），填充 v2 | 50–80 | 200 美元 |
| G3 | TrustPromptDA 微调：主模型 + 3 折 LOSO + 约 4 个关键消融（每次 12–18 h）；Marigold-DC 仅测试集 | 150–230 | 550 美元 |
| G4 | 最终评测、bootstrap、冗余 | 40–60 | 150 美元 |
| 合计 | | 280–430 | **1,050 美元**（原计划估算 700–1,750） |

约束：竞价实例 + 每 30 分钟同步 checkpoint；不在云端存放未获许可再分发的数据衍生物；不使用测试集做任何选择。开通账号与付款由团队完成，Claude 提供每阶段的命令、配置与资源估算。

## 7. 状态

- 本修订：〔已写入〕。
- 第 1 步进展见 `docs/progress.md`。
