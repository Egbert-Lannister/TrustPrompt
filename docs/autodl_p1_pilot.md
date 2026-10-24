# AutoDL P1 pilot 实测与完整训练预算

本页记录 pilot 结束时的验证和预算快照。用户随后已批准 100 元完整训练并启动 F0，后续状态见 [完整训练执行记录](autodl_p1_full_training.md)。

2026-10-24 pilot 结束时：**一小时预算的 F1 pilot 通过，当时完整 F0/F1 尚未启动。** 实际 operator 墙钟 59.07 分钟；训练器按事先设置预留 60 秒保存，正常退出码 0，没有触发超时终止。此结果验证工程稳定性和资源需求，不是模型精度评测或正式实验结果。

## 实际运行

| 项目 | 实测 |
|---|---|
| 设备 | NVIDIA RTX 6000D，sm_120；CUDA 可见总显存 83.049 GiB |
| CPU / RAM / 共享内存 | 24 核配额 / 120 GiB / 60 GiB |
| 环境 | 独立 `p1-cu128`：torch 2.11.0+cu128、torchvision 0.26.0+cu128、cuDNN 91900 |
| 配置 | F1，480×640，batch 6，workers 8，seed 20261024；无 smoke 参数 |
| 数据 | Hypersim 渲染数据与模拟提示；4,418 帧、65 个官方 train 场景；未用于传感器精度评测 |
| 完成量 | 2,598 步，15,588 次样本处理（不是 15,588 张独立图像） |
| 训练循环 / 启动 / 退出保存 | 3,526.862 s / 14.198 s / 1.129 s |
| 总吞吐 | 4.419793 样本/s，含周期保存，排除启动和最后保存 |
| PyTorch 分配显存峰值 | 33.0058 GiB |
| NVML 显存采样最大值 | 40.1650 GiB；237 次约 15 秒间隔采样，不是连续峰值 |
| 容器 memory.current 采样最大值 | 20.5469 GiB，包含文件缓存；不等于进程 RSS |
| OOM / OOM-kill / 采样错误 | 0 / 0 / 0 |

开始/结束为 14:49:27–15:48:32 UTC（America/New_York 10:49:27–11:48:32）。用户原计划为 48 GB 级卡，但实际设备可见约 83 GiB；本结果仅代表这台设备，不能当作另一种 48 GB 卡的实测吞吐。

历史 cu126 环境在最小 CUDA 运算上报 `no kernel image`，因此另建 cu128；普通依赖沿用原固定版本。新环境 `pip check`、CUDA deform_conv2d 前后向和 62 项 CPU 测试通过，另 1 项 P2 数值梯度测试明确排除。失败与恢复日志都保留，详见 [执行记录](autodl_p1.md)。Hugging Face 镜像用于固定版本公开权重，大小与 SHA-256 校验未放宽。

## 检查点与退出核验

- 第 2,000 步周期检查点成功落盘，训练随后继续；退出时 `last.pt` 更新到 **第 2,598 步**，大小 **1,023,597,413 B**。
- 训练端独立 SHA-256：`000a5106b1d6430eee44978f3c926eb4900e137b4448744961840462d65b697b`。
- CPU mmap 载入检查通过：836 个模型状态张量（含 buffer）、1,890 个优化器张量均有限；优化器非空；torch/CUDA/NumPy/Python 随机数状态存在；checkpoint、receipt 和最后保存步数一致。训练帧、场景划分及代码/配置恢复合同匹配。耗时 3.923 s，峰值 RSS 1,596,108,800 B。
- 129 条日志的损失和梯度均有限；置信度头梯度范数范围 0.1368–7.7690，确认梯度到达。训练器在每步检查非有限损失与梯度；日志并不包含每一步数值。CPU 内容检查不替代实际续训复现或推理测试。
- 退出后未发现 `scripts.train_omnidc` 进程或 GPU 计算进程；数据盘剩余 41.43 GiB。实例未关闭，仍可能继续计费。
- 大检查点保存在远端，没有重复传输到本地。配置、回执、逐步日志、帧清单、源码快照和差异等小文件已取回，逐文件 SHA 与训练端检查记录一致。

## 完整 F0/F1 的后续预算

单价采用用户提供的 **7 元/GPU 小时**。每个变体目标为 `floor(100000/6)=16666` 步，即 **99,996** 次样本处理。F1 已完成 15,588 次，后续仅计算剩余 **14,068 步 / 84,408 次**。

| 后续项目 | GPU 小时估算 | 费用估算 |
|---|---:|---:|
| F0 从零训练主体 | 6.285 | ¥43.99 |
| F1 从第 2,598 步续训的剩余主体 | 5.305 | ¥37.13 |
| 两模型合计，含两次启动与最后保存的代理开销 | 11.598 | ¥81.19 |
| 上项统一加一次 20% 预留 | **13.918** | **¥97.42** |

建议为后续 F0/F1 预留 **约 14 GPU 小时、100 元**。F0 尚未单独测速，采用本次 F1 吞吐作代理。20% 是规划余量，不是统计置信界。后半程 5 个互不重叠且至少 300 秒的窗口为 4.3274–4.4367 样本/s；按该观察范围外推，两模型剩余训练主体为 11.55–11.84 小时，不能称为 95% 置信区间。

本次 operator 占用折算 **¥6.89**。从首次观察到 GPU（14:21:58 UTC）到 pilot 启动，环境适配约 27.5 分钟，另折算 **¥3.21**；两者合计约 **¥10.10**，不是平台账单，实际开始计费时间未知，且未包含之后等待/整理报告时间。未来预算不重复计入 pilot 已完成的 F1 样本，也不含后续评测、其他开机时间；`final.pt` 写入/哈希与恢复优化器的额外启动时间尚未单独计时，由预留承担其不确定性。

## 来源、产物与复现

- 训练源码：`8515378a160c207ae11e94677836ef2917f5162c` 的 42 文件 v4 归档，远端无 Git 仓库，回执不伪造 `git_commit`。归档 SHA 为 `c7d7b50aea1c2bac1eaa0b7b6da6f519c0bdff7f78a0870d4cba176dba9938eb`。
- OMNI-DC 源码：`a5ecd35d087ed5e5fcc2be93989d1db3fde194fd`；两个公开权重的 revision、真实 SHA 和加载记录见训练 receipt 的 `model_version`。模型源码干净且固定。
- 远端训练目录：`/root/autodl-tmp/TrustPrompt/results/p1_f1_20261024_224929`；远端环境：`/root/trustprompt-envs/p1-cu128`。
- 本地原始记录：[pilot 目录](../results/autodl_p1_20261024/pilot/)。[检查点完整性回执](../results/autodl_p1_20261024/pilot/checkpoint_validation.json)、[资源汇总](../results/autodl_p1_20261024/pilot/pilot_operator_20261024T144927Z_a49f31b2/resource_summary.json)、[诊断图](../results/autodl_p1_20261024/pilot/figures/pilot_diagnostics.png)及 [PDF](../results/autodl_p1_20261024/pilot/figures/pilot_diagnostics.pdf) 已生成。图已目视检查，未平滑训练损失，未把损失曲线当作质量评测。
- 汇总与绘图脚本位于 `results/autodl_p1_20261024/`，输入/输出/脚本 SHA 与执行命令写入相应 JSON。源代码快照、数据划分、种子和实际运行命令均在原始训练目录内。
- 当前联合预算回执：[JSON](../results/autodl_p1_20261024/pilot/summary/pilot_summary.json) / [Markdown](../results/autodl_p1_20261024/pilot/summary/pilot_summary.md)，绑定独立 CPU 检查点验证；全部本地小产物的大小与 SHA 见 [artifact manifest](../results/autodl_p1_20261024/pilot/artifacts_manifest.json)。最初只核对日志/SHA 的汇总仍保留在 operator 子目录，属于较早的验证范围，不替代当前联合回执。

本地重算预算（新输出目录，保留已有结果）：

```powershell
.\.venvs\baselines\Scripts\python.exe -B results/autodl_p1_20261024/summarize_pilot.py --run-dir results/autodl_p1_20261024/pilot/p1_f1_20261024_224929 --operator-dir results/autodl_p1_20261024/pilot/pilot_operator_20261024T144927Z_a49f31b2 --checkpoint-sha256 000a5106b1d6430eee44978f3c926eb4900e137b4448744961840462d65b697b --checkpoint-validation results/autodl_p1_20261024/pilot/checkpoint_validation.json --price 7 --output-dir results/autodl_p1_20261024/pilot/summary_recomputed
```

pilot 报告提交时，完整训练命令已准备但尚未执行；用户随后批准了100元预算，当前启动情况见 [完整训练记录](autodl_p1_full_training.md)。继续训练须保持同一 cu128 环境与远端源码；新增评测脚本不能写入原训练源码树，否则会改变恢复合同。已批准的顺序为 F0 后恢复 F1；完成后再另行安排独立开发集检查点评测。
