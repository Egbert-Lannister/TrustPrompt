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

## 6. 状态与可追溯性

- 原计划副本不修改内部陈述；本文件记录采用解释。`reuse_provenance.json`给实际旧文件hash和Git HEAD，旧树有未提交修改，不能只用HEAD代表旧文件。
- 每次评测保存配置、seed、命令、split manifest、源码hash、源码zip、Git HEAD/status/diff、输入/输出hash、日志、逐图和场景指标。当前早期实验在未提交代码上运行，source snapshot可还原其代码。
- `代码已实现`、`小规模验证通过`、`正式实验完成`分开；当前无正式训练/最终测试结果。未执行邮件、协议接受、数据上传、付费资源或长训练。
