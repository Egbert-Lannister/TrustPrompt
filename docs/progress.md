# 进展记录

日期：2026-10-09，America/New_York。此前工作因用户中断停在核查；本次重新检查空目录后开始实现。GLARE保持只读。

## 已实现并小规模验证

- 独立Git、项目环境、路径配置与原计划副本。
- Sample v1 pickle-free NPZ读写、dtype/shape/单位/无效值检查、场景泄漏检查。
- Mirror3D正确N编号、原始提示、裁剪、modular delta、精确mask、源scene核验、禁读test守卫。
- raw_prompt native-grid nearest fill；几何反投影/投影/平面与最近邻内参约定。
- 固定污染/干净/空洞/边界区域；AbsRel/RMSE/logRMSE/delta/IR/DOR；逐图→逐场景；可信度统计函数、paired bootstrap/Holm函数。
- 真实5帧数据获取与准备、合成5帧工程测试、真实5帧raw基线运行和对齐图。

## 已执行证据

- 数据获取：`data/source_cache/nyu_smoke_v1.json`，30,408,705字节下载，25个数组hash，未读取测试深度。
- 实际21帧编号回归：N 99.87147179% / N−1 0.07992236%在1mm内，613,095像素；证据 `docs/audit_data.json`。
- 初轮真实运行：`results/mirror3d_raw_prompt_smoke_v1`，5帧/5个训练场景，无无效预测。C AbsRel=0.6625801，clean=0.00113237，hole=0.1010521，all=0.0452196，IR=1，DOR=0.3856979。
- 上述是未对齐、逐场景等权的 **真实训练集工程诊断**；背景为inpainted参考、strict RGB-Z不合格，不能直接作为论文主结果。v1之后已补充机器可读资格和镜面population字段，最终复跑记录将追加。
- 合成运行：`results/synthetic_raw_prompt_smoke_v1`，仅验证工程，不能计入真实实验。

## 当前限制与后续

1. 本地NYU只有部分源cache。最小5帧已解决；扩大训练需继续按冻结场景读取所需数据，不能打开测试集调试。
2. NYU实际RGB相机Z与投影索引约定仍需精确注册核验。当前采用发布格诊断，严禁隐去这个限制。
3. 镜外材质未知、背景参考不独立；后续材质/真值质量需单独审计。
4. 先利用已有UniDepthV2-S权重验证少量GPU推理与免训练鲁棒对齐，再考虑PromptDA-S。当前没有长训练或云计算需求。
5. HAMMER、其他数据集与许可、训练预算、留一传感器和完整消融尚未完成。未代为接受协议、发邮件、上传数据。

正式实验完成：**否**。当前完成状态仅表示代码和小规模验证，不等于原计划全部研究完成。

## 本次最终验证与扩展

- 全套测试：`python -m pytest -q -p no:cacheprovider`，**85 passed**，21.28秒。1条外部python-dateutil弃用警告，不影响测试结果。见 `validation.json`。
- 最终raw运行：`results/mirror3d_raw_prompt_smoke_v2`，补齐strict RGB-Z资格、镜面分母、样本身份和来源hash。5张RGB/提示/GT/mask/区域/误差图均已人工视觉检查；镜面轮廓与图中镜子位置一致，不等于像素级物理标定完全认证。
- UniDepthV2-S：先1帧后5帧、RGB-only原生infer，全部有限正值、输出depth与pointcloud Z逐值相等。权重不复制、不下载；原第三方源码hash运行前后无变化。最大PyTorch allocated=348,976,640字节（约333 MiB），reserved约390 MiB。详见 `unidepth_smoke.md`。
- RobustAlign：全局RANSAC+IRLS与局部8×8/3×3支持拟合均完成真实5帧评测，参数固定，未按结果调参。完整表由JSON生成到 [smoke_results.md](smoke_results.md) / `smoke_results.json`。
- 全局对齐在本次训练诊断中 C AbsRel=0.1155376、clean=0.0392351、IR=0.0755861、DOR=0.1228094。相较raw的污染误差下降，但clean误差增加；不能称为无退化，也不能从5个训练场景推断泛化。
- 局部对齐在playroom_0007/frame440产生1,454个非正/非有限逆深度导致的无效输出；原样保存，无静默丢弃。其clean/hole/all连续指标NA，C AbsRel=0.1890410。最终run `results/mirror3d_robust_local_smoke_v2` 的receipt状态为 `completed_with_invalid_predictions`。不算该模型验证成功；这是已定位的真实失败。
- 配对10,000次场景bootstrap已实际执行：`results/mirror3d_paired_comparison_v2`。对输入hash、固定区域分母逐项核对后比较；遇到空/无效场景保持NA。仅描述性训练诊断，无多数据集显著性结论。
- 所有4个方法、5帧的固定区域pixel counts一致。测试域未打开用于模型选择。早期v1和修订v2都保留；文档指向最终v2文件，历史产物不被覆盖。

## 下一阶段按依赖执行

1. 对局部拟合的负逆深度外推做诊断，设计并单独报告正值约束/全局回退变体；不得擦除本次失败记录或通过评测过滤改善指标。
2. 解决NYU的严格RGB-Z及背景GT资格后才进入正式主表；此前只运行发布格工程诊断。
3. 本次只读检查E:/GLARE_Data及first_surface目录，见到mirror3d/cleargrasp/booster/robi/transnormal，未发现HAMMER；尚未进行HAMMER整包下载或协议操作。接入HAMMER前先核实官方许可、体积和train小片段格式，不能把已有其他数据当作它。
4. PromptDA-S尚未接入/下载；先核对源码与许可证，再按1帧→5帧测试测量实际显存，随后才扩展模型/模拟器/训练。当前不需要付费云或长训练；原计划GPU小时均未被本次smoke验证。

代码初始提交：`64700f1`。后续审查修复和文档更新有独立提交。各run的真实代码HEAD、未提交差异与源码快照以该run receipt为准，不能用最终HEAD替代历史运行版本。

## 2026-10-09 计划修订 v2 与第 1 步

计划修订写入 `docs/plan_revision_v2.md`（R1–R12、检查点、日程、云预算申请）。原计划副本未改。

### 执行环境（新核实）

Claude 在用户电脑上的 shell 是挂载项目目录的 Linux 工作区：2 核、3 GB 内存、无 GPU、无 torch，无法访问 NYU 与 HuggingFace，看不到 E 盘。CPU 评测在此用独立 venv（Python 3.10.12，numpy 1.26.4、scipy 1.13.1，版本同 `requirements-local.txt`）运行；GPU 推理和 NYU 下载需在 Windows 上执行（`scripts/windows/step1_dev.ps1`）。

### 1b 对齐 v2 与 Gate 基线 —— 已实现，小规模验证通过

- `tp/align/robust.py`：新增 `local_depth_v2`（插值后逆深度 ≤ ε 的像素回退全局解，ε = 1e-6 × 支持集逆提示中位数）与 `gate_prompt`。v1 输出不变。
- `tp/eval/run_eval.py`：新增方法 `robust_local_v2`、`gate_global`、`gate_local_v2`；Gate 输出二值一致图 w，按 clean_all 标签算 AUROC/AP；新增 `--split`，test 在方法冻结前拒绝运行；NYU 汇总带 `metric_semantics`（R2）。
- 修复：`baselines/cached_prior.py` 在 POSIX 主机上无法解析 Windows 写入的预测路径（改用 PureWindowsPath 取文件名）。首次运行因此失败的三个目录 `results/mirror3d_{robust_local_v2,gate_global,gate_local_v2}_smoke_v1` 保留，receipt 为 failed；有效运行为 `_smoke_v2`。
- `tests/conftest.py`：支持 `TP_TEST_TMP`（挂载目录不允许删除时把测试临时目录放到外部）；Windows 默认行为不变。
- 测试：`TP_TEST_TMP=... python -m pytest -q`，97 passed，1 skipped（torch 不在 Linux 工作区）。
- 同 5 帧训练集结果（`docs/step1_results.md`，由逐图 JSON 生成）：
  - v2 消除了 playroom_0007/440 的 1,454 个无效像素（回退比例 0.52%），其余 4 帧回退 0 像素，逐值与 v1 相同。
  - 镜面区域（R2 主口径）：全局对齐 C AbsRel 0.0645、IR 0、DOR 0；逐块对齐 C AbsRel 0.2132、DOR 0.379。逐块拟合在大镜面附近会把虚像深度当作“局部真值”吸收进去，这正是计划中 T 需要大感受野和“局部/全局分歧”特征的理由。仅 5 帧训练诊断。
  - Gate（全局）：镜面区域与全局对齐几乎相同（C 0.0645），整图 clean（相对补洞参考的偏离）从 0.0392 降到 0.0246；二值 w 的场景平均 AUROC 0.839（Gate 局部 v2 为 0.886）。
  - 配对 bootstrap（参考 = 全局对齐）：`results/mirror3d_paired_comparison_step1_v1`，5 场景，只作描述。

### 1a NYU RGB-Z 上界 —— 代码已实现，等待数据

- `tp/data/nyu_rgbz.py` + `scripts/nyu_rgbz_bound.py`：从官方工具箱 ZIP 解析外参（文件 SHA256 必须与 `intrinsics_audit.md` 记录一致，无法识别的 MATLAB 语句直接报错），逐像素精确反解 `Z_rgb`，取像素四角射线的最坏值；同时报告两种主点约定。判定规则见修订 R3。
- 合成测试 4 项通过。真实运行需要工具箱 ZIP：Linux 工作区无法访问 cs.nyu.edu，需在 Windows 下载（脚本第一步）。

### 1c NYU 开发集扩展 —— 代码已实现，等待 Windows 运行

- `fetch_nyu_subset.py --dev`：选出全部 train+val（76+19=95 帧），上限 900 MiB；N−1 前帧属于 test 场景时只跳过该前帧并记录，不读 test 深度。
- `prepare_mirror3d.py --dev`、`configs/paths.dev.yaml`、`configs/unidepth_dev.yaml`（`split: dev`，阶段名 `real_dev_verified`）、`configs/mirror3d_dev_eval.yaml`。
- `cache_unidepth.py` 的 dev 模式允许 val、拒绝 test，新增测试覆盖。

### 下一步

1. Windows 运行 `scripts/windows/step1_dev.ps1`（下载约 0.03 MB 工具箱 + 最多 900 MiB NYU 切片；UniDepth 95 帧约几分钟）。
2. 之后在 Linux 工作区运行 RGB-Z 上界与 6 个方法 × {train, val} 的评测，更新 `step1_results.md`。
3. 同时开始第 2 步：PromptDA 源码与许可核对、推理包装（GPU 部分在 Windows）。

## 2026-10-09 晚：Windows 第一次运行 step1_dev.ps1

### 发生了什么
- 工具箱 ZIP 下载成功，SHA256 `6211b20f…` 与审计记录一致。
- NYU 切片下载在第 17 帧（annotation 347）时 TLS 握手超时，3 次重试耗尽后中止；已下载约 125 MB。缓存 `data/source_cache/nyu_dev_v1.h5` 状态为 INCOMPLETE，`complete` 标志未写入。
- 之后的 prepare 正确拒绝了未完成的缓存；但脚本没有在失败处停下（PowerShell 的 `$ErrorActionPreference` 不作用于外部程序），继续执行了 UniDepth 步骤并最终打印 “Done”，造成误导。没有产生错误数据：prepare 只留下 receipt 为 failed 的目录，UniDepth 在创建输出前就失败。
- 失败的 prepare 目录已改名为 `data/processed/mirror3d_dev_v1_failed_incomplete_source`（保留，不删除）。

### 修复（已实现、测试通过）
- `fetch_nyu_subset.py`：新增 `--resume`。只续传 INCOMPLETE 缓存，且要求 annotation 列表、划分文件哈希、官方 MAT 的 URL/大小/ETag/Last-Modified 都与中断前一致；已写入的帧逐字段重新计算哈希，与上次记录完全一致才保留，否则删除重下。在真实的中断缓存上演练：16 帧保留，347 帧重下。新增 `--retries`（默认 8 次，指数退避上限 60 s），单次请求超时 60 s，每次运行的下载量与重试次数写入 `resume_history`。dev 模式单次运行上限提高到 1500 MiB（按已下载部分估算全部 95 帧约 0.7–0.9 GB，900 MiB 余量不足）。
- `step1_dev.ps1`：每步检查退出码，失败即停；已完成的步骤自动跳过；缓存存在时自动加 `--resume`。
- 测试：102 passed，1 skipped。

### 1a RGB-Z 上界 —— 5 帧训练集已运行
- 第一次运行 `results/nyu_rgbz_bound_smoke_v1` 失败（receipt 停在 started）：解析器遇到 `R = reshape(R, [3 3])` 拒绝猜测。按官方文件实际写法修正解析：9 元素行向量取负 → MATLAB 列优先 reshape → `inv(R')`；平移按 `T = [t_x; t_z; t_y]` 的实际顺序。
- 官方 R 的行列式为负（≈ −I）。`rgb_world2rgb_plane` 只用 X/Z、Y/Z，整体取负不改变投影，因此采用 (−R, −T) 作为物理刚体变换，并检查所有恢复出的 Z_rgb 为正。解释记入 decisions.md §9。
- `results/nyu_rgbz_bound_smoke_v2`：5 帧 921,519 个有效提示像素，|Z_rgb − Z_raw|/Z_raw 最大 0.473%，99.9 分位 0.419%，中位 0.125%；两种主点约定结果相同到 1e-5。按 R3 规则（≤ 0.5%）判定为 `rgb_z_bounded`，但余量很小，95 帧开发集到位后重跑，若超过 0.5% 则按 R3 改写到 RGB Z。

## 2026-10-10 第 1 步完成：NYU 开发集（95 帧）

### 数据（Windows 运行 `step1_dev.ps1`，续传后完成）
- `data/source_cache/nyu_dev_v1.h5`：95 帧（train 76 / val 19），状态 COMPLETE；续传一次，本次下载 578.8 MB，重试 26 次。第一次中断时已写入的 16 帧经哈希校验后保留。
- `data/processed/mirror3d_dev_v1`：95 个样本，73 个场景（55 train + 18 val）。编号回归：92 帧（3 帧的 N−1 属于 test 场景，按规则跳过）2,860,800 像素，N 正确率 99.845%，N−1 为 0.064%。
- `cache/unidepth_dev_v1`：95 帧，`real_dev_verified`，点云 Z 与深度逐值相等。

### RGB-Z 上界（R3）—— 已完成
- `results/nyu_rgbz_bound_dev_v1b`：95 帧 19,160,518 个有效提示像素，最大 0.473%，99.9 分位 0.419%，中位 0.120%。判定 `rgb_z_bounded`，NYU 可按“配准 Z、误差 ≤ 0.473%”进入主表。
- `results/nyu_rgbz_bound_dev_v1_killed_background`：第一次尝试在后台运行时进程被终止，receipt 为空，保留作记录。

### 基线评测（6 方法 × train/val）—— 小规模验证通过，非正式结果
- 表格：`docs/dev_results.md`（由逐图 JSON 生成）。全部 12 个运行 `small_scale_validation_passed`，无效预测 0。
- Linux 工作区的后台进程无法跨调用存活，因此 `run_eval` 新增 `--time-budget-s` 与 `--resume`：逐帧写 `partial_rows.jsonl`，续跑要求配置、源码哈希、manifest 完全一致；续跑记录在 receipt 的 `resumed_sessions`。合成数据测试验证“分段续跑”与“一次跑完”的逐图结果相同。`mirror3d_dev_robust_local_v2_val_v1` 中一次是外部超时硬中断后续跑（receipt 停在 started，jsonl 完整）。
- 全局对齐方法改为跳过局部分块拟合（全局结果逐位不变，有测试），首个 robust_global val 运行因代码变更改名为 `_abandoned_code_change`。
- 配对 bootstrap 修正：原实现只要有一个场景该区域为空就整体 NA；现改为与聚合一致——空区域场景（与方法无关）排除并列出，含无效预测的场景仍使整体 NA。旧输出改名 `_superseded_empty_scene_rule`，新输出 `results/mirror3d_dev_paired_val_v2`。
- 主要观察（val，镜面像素，19 帧 / 18 场景，参考 = 全局对齐）：
  - 全局对齐 C AbsRel 0.135，DOR 0.313；比 5 帧冒烟（0.065、0）差，冒烟样本偏乐观。
  - 逐块 v2 比全局差：C AbsRel +0.126，95% CI [0.014, 0.264]（16 个非空场景）。逐块拟合吸收镜中虚像的现象在开发集上依然成立。
  - Gate（全局）与全局对齐在镜面 C 上无差别（−0.0005，CI 跨 0），继承率略升 +0.015 [0.003, 0.032]：二值一致性门控会把“与先验碰巧一致”的错误提示保留下来。
  - 无提示 UniDepthV2-S：C AbsRel +0.092 [0.048, 0.143]，DOR +0.337 [0.129, 0.551]。
  - Gate 的二值 w：场景平均 AUROC val 0.811 / train 0.838。
- 面板标题此前固定写 “REAL TRAIN SMOKE”，val 运行也如此；已改为按 split 显示，今后的运行生效。

### HAMMER
- 用户反馈整包过大，本地放不下。新增 `scripts/hammer_remote.py`：通过 HTTP Range 只读 ZIP 中央目录得到全部文件列表与大小（index），再按场景/传感器/帧间隔只下载选中的文件（fetch，CRC 校验 + SHA256 记录，可续传）。用内存 ZIP 与伪造 Range 响应做了端到端测试。Linux 工作区无法访问 campar.in.tum.de，需在 Windows 运行 `scripts/windows/hammer_index.ps1`。

### 下一步
1. Windows 运行 `hammer_index.ps1`；我据此设计子集并给出体积。
2. 第 2 步：PromptDA-S/L 推理包装（Windows GPU），在 NYU 开发集上出 C1 现象表。

## 2026-10-10 HAMMER 远程索引

- Windows 运行 `hammer_index.ps1` 成功：只读 ZIP 中央目录 60.6 MB，得到 `data/source_cache/hammer_index_v1.json`。归档 170.46 GB（ETag `"681dead6-27b00c96a3"`，Last-Modified 2025-05-09），432,596 个文件，压缩后 158.7 GiB。服务器支持 Range。
- 目录结构：64 个序列（`sceneN_trajM_k` 与对应 `_naked_`），每个序列有 d435 / l515_depth / l515_rgb / polarization / tof 五个相机目录。`polarization/` 下已有 RGB、`depth_d435` / `depth_l515` / `depth_tof`（三种传感器深度已投到该视角）、`_gt`、`_instance`、`_pose` 与 intrinsics，正好对应一个 Sample × 3 种传感器。体积主要来自偏振原始数据 `pol/`。
- 子集 `configs/hammer_subset_v1.yaml`：polarization 视角全部帧 + naked 序列的 GT/位姿，65,070 个文件，下载 15.8 GiB，落盘 19.1 GiB；`hammer_subset_v1_stride2.yaml` 隔帧取，下载 7.9 GiB，落盘 9.6 GiB（dry-run 结果）。
- 下载脚本 `scripts/windows/hammer_fetch.ps1`：需指定目标目录 `-Dest`，先检查剩余空间，失败自动等待 60 s 续传，最多 20 次。
- 说明：Claude 在用户电脑上的 shell 是独立的 Linux 虚拟机（内核 6.8 Ubuntu），不是用户的 WSL，无法访问 campar.in.tum.de，因此下载需在 Windows 或用户自己的 WSL 中运行。

## 2026-10-11 HAMMER 接入

- 下载完成：`E:\TrustPrompt_data\hammer_subset_v1`，65,070 个文件全部通过大小与 CRC 校验，17.03 GB。
- Claude 在用户电脑上的 shell 无法挂载 E 盘该目录（挂载失败），可列目录与暂存单个文件。因此逐帧审计在 Claude 工作区用暂存文件完成；全量处理由 Windows 脚本执行。
- 审计（`docs/hammer_audit.md`）：16 位毫米深度；GT 为轴向 Z；naked 按位姿配对；无材质标签；D435 背景系统偏远 2–4%。
- 代码〔已实现、测试通过〕：
  - `tp/data/hammer.py`：每帧 × 3 传感器生成 Sample，`behind` = 配对的 naked GT；按场景划分（2–10 train / 11 val / 12–14 test），test 默认拒绝；每个样本记录全部源文件 SHA256。
  - `scripts/prepare_hammer.py`：物化开发子集（train 每 20 帧、val 每 5 帧，约 1,100 个样本、约 3.4 GB）。全量约 52 GB 不物化，训练时由加载器直接读 E 盘。
  - `run_eval`：按数据集选择主口径（NYU mirror_only / HAMMER objects_only），输出 `per_image_objects_only.json`；面板标题按数据集显示。
  - `cache_unidepth`：按 manifest 记录运行，同一 RGB 只推理一次，三个传感器样本共享；`CachedPrior` 身份加入 sensor。
  - `compare` 自动识别主口径；`summarize_dev --dataset hammer`。
  - 测试 113 passed / 1 skipped（含合成 HAMMER 目录上的加载、配对、test 锁、prepare→evaluate 端到端）。
- 对齐检查（5 帧 × 3 传感器，`results/hammer_alignment_check_v1/`）：边缘对齐良好；L515 在玻璃瓶、酒杯、餐具上出现预期的偏远（穿透）区域。
- 待运行：`scripts/windows/hammer_dev.ps1`（物化 → UniDepth 缓存 → 6 基线 × val/train → 配对比较 → 表格）。
- 新增人工任务：HAMMER 逐实例材质标注（约 100 个实例），用于分材质分析与“GT 材质掩码”基线。

## 2026-10-11 HAMMER 开发评测：D 盘写满

- `hammer_dev.ps1` 运行到 `gate_local_v2 val` 第 98/327 个样本时 D 盘写满（552 GB 盘剩 0.6 GB）。已完成：raw_prompt、unidepth_v2_s、robust_global、gate_global、robust_local_v2 的 val。失败目录改名为 `results/hammer_dev_gate_local_v2_val_v1_failed_disk_full`。
- 本项目在 D 盘的 HAMMER 产物约 30 GB：results 19.6 GB（其中对齐稠密图 alignment/*.npz 共约 11 GB，未压缩预测每个运行 1.2 GB），开发子集 3.2 GB，UniDepth 缓存 5.4 GB（其中点云约占 70%）。
- 修复（测试 114 passed）：`run_eval` 新增存储策略 `save_predictions` / `save_alignment_maps`（默认 all，NYU 行为不变；HAMMER 配置为前 20 / 前 6 个样本），所有预测的数组哈希 `prediction_array_sha256` 一律记录；`cache_unidepth` 新增 `store_points: false`（先校验点云 Z 与深度逐值相等再丢弃点云）。

## 2026-10-11 HAMMER 开发评测完成（小规模验证通过，非正式结果）

- 子集：train 272 帧（9 个场景，每 20 帧取 1）、val 109 帧（scene 11，每 5 帧取 1），× 3 传感器 = 1,143 个样本；1,134 个与 naked 配对，9 个位姿不符（无 behind，对象口径为空并记录）。UniDepth 缓存 1,143 个样本、`real_dev_verified`。
- 11 个评测运行全部 `small_scale_validation_passed`，无效预测 0；val 6 个方法，train 4 个方法（逐块对齐只跑 val）。表格 `docs/hammer_dev_results.md`，含按传感器分解。
- 统计限制：val 只有 1 个场景，场景级 bootstrap 只能描述；train 有 9 个场景。确认性统计要等测试集（3 个场景）或 LOSO 折。
- 主要观察（物体像素）：
  - 原始提示污染区 AbsRel：val 0.12–0.16、train 0.12–0.27（按传感器）；危险高估率 0.24–0.54。
  - 鲁棒对齐把污染区误差降到 0.07–0.12，但继承率仍有 0.21–0.39，危险高估率 0.18–0.36，即仍有大量错误提示被保留或拟合进去。
  - 与 NYU 镜面相反：HAMMER 上逐块对齐优于全局（val 污染区 0.072 vs 0.092，危险高估率 0.18 vs 0.27）。小物体时逐块拟合的支撑集主要是正确背景；大镜面时逐块拟合会吸收虚像。没有单一固定对齐策略在两类场景都好，这正是可信度网络 T 与“按区域选择低自由度候选”要解决的问题。
  - Gate 的继承率高于对应的对齐方法（val 0.42 vs 0.30）：与先验碰巧一致的错误提示被保留。
  - 无提示 UniDepthV2-S 在近距离桌面场景整体偏远（危险高估率约 0.98–0.99），不能作为度量深度直接使用。
