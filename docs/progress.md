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
