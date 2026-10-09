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
