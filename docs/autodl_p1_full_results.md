# AutoDL P1 F0/F1 完整训练结果

> 后续状态（2026-10-25）：两个final权重及元数据已取回本地并校验SHA，last.pt留在远端；本地固定开发集评测已完成，P1因G3未通过停止。见[本地评测报告](omnidc_p1_local_eval.md)与[传输记录](omnidc_p1_transfer.md)。下文保留训练验收当时的状态和关机失败证据，不表示当前仍缺少本地final。

核验日期：2026-10-25，America/New_York。作业：`full_operator_f0_f1_20261024`。

**F0 与 F1 均完成 16,666 步训练，最终回执均为 `training_complete`，两项训练端 CPU 检查点验证均通过。自动关机失败，尚未确认平台停机或停止计费。**

关机证据 `mirror/job/operator_shutdown.json` 记录：2026-10-25 04:53:00 UTC（当地 00:53:00）调用 `/usr/bin/shutdown` 时出现 `OSError(8, 'Exec format error')`。用户已被提示在 AutoDL 控制台检查并手动关机。此轮只读本地同步证据，未连接远端、修改关机实现或重新启动任何作业。硬截止保护不能替代平台停机确认，费用上限也不能据此宣称已保证。

## 实际执行结果

| 阶段 | UTC 起止 | 步数 | 模型更新数 | operator 阶段墙钟 | 回执状态 |
|---|---|---:|---:|---:|---|
| F0 训练 | 10-24 17:17:47–23:21:10 | 0→16,666 | 16,666 | 21,802.755 s（6 h 03 m 22.8 s） | `pilot_complete` |
| F0 零步收尾 | 10-24 23:21:10–23:21:28 | 16,666→16,666 | 0 | 18.121 s | `training_complete` |
| F1 从原 pilot 续训 | 10-24 23:21:35–10-25 04:52:11 | 2,598→16,666 | 14,068 | 19,836.340 s（5 h 30 m 36.3 s） | `pilot_complete` |
| F1 零步收尾 | 10-25 04:52:11–04:52:27 | 16,666→16,666 | 0 | 16.091 s | `training_complete` |

四阶段 exit code 均为 0。计时训练模式沿用既有 `pilot_complete` 状态名；达到目标后零步恢复仅保存正式 `final.pt`，不增加优化器更新。结论同时依据阶段快照、最终回执及 CPU 验证，未把计时模式状态本身当作正式完成证明。

两次 CPU 验证结束后，operator 于 **2026-10-25 04:52:34 UTC（当地 00:52:34）** 写入 `completed`，operator 总墙钟 **41,688.790 s / 11.580219 h**。F1 使用原 pilot 目录，原始 2,598 步不重复计入本轮训练量。

| 指标 | F0 本轮 | F1 本轮续训 |
|---|---:|---:|
| 实际处理样本数 | 99,996 | 84,408 |
| 训练循环墙钟（含循环内周期保存） | 21,786.970 s | 19,819.857 s |
| 累计训练吞吐 | 4.589716 样本/s | 4.258759 样本/s |
| 初始化/恢复时间（trainer 内计时） | 13.078 s | 14.476 s |
| PyTorch allocated 峰值 | 33.005791 GiB | 32.998264 GiB |
| 最终训练步 | 16,666 | 16,666 |

每个模型累计处理 16,666×6=99,996 个样本；配置中的 100,000 是名义目标，经整步截断。F1 本轮之外的 pilot 已处理 15,588 个样本。这里的吞吐是训练循环墙钟口径，不是 GPU 内核时间。终端 `checkpoint_save_seconds` 接近零，因为目标步已在循环中保存；不能解释为完整检查点写盘只需微秒。

资源日志共 **1,386** 个样本、错误行 **0**，采样间隔中位数 **30.112 s**，已记录的 `oom`/`oom_kill` 均为 0。NVML 采样峰值 **41,697 MiB / 40.719727 GiB**（F0 40.167 GiB、F1 40.720 GiB）；cgroup `memory.current` 采样峰值 **22.144093 GiB**，包含文件缓存，`memory.peak` 不可用。这些采样最大值不等同于连续精确峰值，也不同于 PyTorch allocated 指标。实际 GPU 为 **NVIDIA RTX 6000D**，可见显存约 **83.05 GiB**，不是 48 GB 卡的实测。

## 本地与训练端核验

- `collection_complete=true`，39 项小产物共 4,313,887 字节；本地逐文件复算 SHA-256 与大小，全数匹配 manifest 和 download inventory。ACK 回读通过。manifest SHA-256：`46a95082f0413ac90e100d5aa6e636b2f0296269b62db3e9c122de8d17256445`。
- 本地可重复核验脚本通过 **432 项**检查，涵盖作业身份、路径、哈希、阶段起止步、样本数、源码快照、数据清单、日志和原 pilot 快照。独立复核另确认恢复合同与阶段证据一致。
- 两份训练端 CPU 验证回执均 `passed`，各检查 **836 个模型张量、1,890 个优化器张量**，四类 RNG（torch/CUDA/NumPy/Python）齐全；固定版本校验器检查模型与优化器张量有限，并确认 `final_model_equals_last=true`。
- 两组日志各 **834** 条，按 20 步间隔加最终步记录，步数单调且无重复，全部已记录数值有限。F1 日志保留原 pilot 前缀。日志有限不等于观测了每个未记录步骤的全部内部张量。
- 本地只核对已同步的 CPU 验证回执，**没有下载或重新加载大检查点**。训练端验证器 SHA-256：`23f31469512a562aa7849bf59babf6b991107ad61a2a3070acad53cbc8f63503`。
- `handoff.json` 保留启动快照；终态依据为 `mirror/completion.json`、阶段回执与 `mirror/collector_status.json`。

## 检查点与可恢复位置

以下路径相对于远端 `/root/autodl-tmp/TrustPrompt/`。权重文件仍在远端；本地已有其元数据、源码、配置、数据清单、日志和验证回执。

| 模型 | 文件路径 | 字节数 | SHA-256 |
|---|---|---:|---|
| F0 | `results/p1_f0_20261025_011748/checkpoints/final.pt` | 341,916,195 | `7fa22ec1c7e9f7059362031dd8556fea9e399525679395b1069dc393eab8e553` |
| F0 | `results/p1_f0_20261025_011748/checkpoints/last.pt` | 1,023,597,413 | `7cd301b7b148e8d25fb6c8203aaa723b8824fab62335a3b28065a3438f194b4a` |
| F1 | `results/p1_f1_20261024_224929/checkpoints/final.pt` | 341,916,195 | `2749fbca74a180eef8cf2addee42f23bf6036cbaeb121c1a7e2d9d680963f66c` |
| F1 | `results/p1_f1_20261024_224929/checkpoints/last.pt` | 1,023,597,413 | `b1ccc73120179bbf05ada5ea211c95ade755b6b2513cec02e2ae7407119cc011` |

`last.pt` 含优化器和 RNG，用于恢复；`final.pt` 用于推理。两文件容器内容不同，文件 SHA 不应相同；相同的是模型张量。冻结的 DAv2 不包含在这些训练检查点中，后续推理仍需固定版本基础权重。

F1 本轮起始检查点 SHA：`000a5106b1d6430eee44978f3c926eb4900e137b4448744961840462d65b697b`，来源为原 pilot 第 2,598 步。

## 配置与来源

- seed **20261024**；batch **6**；裁剪 **480×640**；workers **8**；学习率 **0.0001**；weight decay **0.01**；warmup **500**；final LR ratio **0.1**；grad clip **1.0**。F0/F1 均不启用 F2 gate。
- 仅使用 Hypersim train，**4,418 帧 / 65 场景**，mirror/window/clean 分别 1,285/1,777/1,356。未用验证或测试集选择 checkpoint 或超参数；本次没有精度评测。
- 训练帧清单字节 SHA：`16084f19bf96c53584313ce245dd15fe85acb545a74045726fdd12605f69d810`。F0/F1 与原 pilot 一致。
- 训练部署来源：`8515378a160c207ae11e94677836ef2917f5162c`，配置 SHA：`9b74ff3b2b902c607ad1faf5ea006f6687179e3b71da4aa92eb02453631f29cd`。远端没有 `.git`，因此原回执保留 `git_commit=null`，不将本地报告修改当成训练版本。
- OMNI-DC 上游：`a5ecd35d087ed5e5fcc2be93989d1db3fde194fd`；模型权重 SHA：`8d43f6ac7b372cf0cb84a0ebfebab4bbacf295cab4c2700d87d10ae1749ce5c0`；DAv2 SHA：`a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345`。完整 revision、加载偏差和依赖见各运行的 `receipt.json`。
- Python 3.12.3、torch **2.11.0+cu128**、torchvision **0.26.0+cu128**；包清单 SHA：`07dfdf0ece61f7c46eb0d984c5aa824288478e8f8b7401f2860f50e169272228`。两个变体保持同一环境与训练源码。

## 费用口径与未确认项

用户批准后续 **100 元**，单价 **7 元/小时**。该预算从 **2026-10-24 17:03:13 UTC** 开始，包含本轮准备、训练及检查。以下均为时间乘以单价的估算，不是平台账单。

| 时间口径 | 小时 | 费用估算 |
|---|---:|---:|
| operator 启动至 completed | 11.580219 | 81.06 元 |
| 本轮预算起点至 completed（含准备） | 11.822562 | 82.76 元 |
| 本轮预算起点至失败的关机尝试 | 11.829728 | **82.81 元** |

截至关机尝试，估算低于 100 元 **17.19 元**。**因关机失败，最终总费用未知；如果实例继续计费，之后每小时另增加 7 元。** 不能将收集完成、ACK 或 SSH 状态当成平台停费证明，也不能将 82.81 元称为最终支出。

此前 pilot 及其环境准备、pilot 结束到本轮预算起点之前的空闲时间均排除在上述本轮数字之外。原 pilot 的 trainer 进程墙钟约 3,542.189 s，单按该口径为 **6.89 元**，不包含它的外层 operator、准备和空闲；既有 pilot 报告另列历史费用。原回执保存在 `mirror/job/pilot_before_resume/`，历史总账单不能用本次镜像完整重建。后续推理评测和取回权重产生的成本也不包含在这里。

## 复核与下一步

只读本地证据、重新生成机器可读汇总的命令（不访问远端、不训练）：

```powershell
Set-Location D:\Python_Script\TrustPrompt
.\.venvs\baselines\Scripts\python.exe results\autodl_full_20261024\verify_local_results.py
```

- [机器可读结果与逐项检查](../results/autodl_full_20261024/full_results.json)
- [原始终态](../results/autodl_full_20261024/mirror/completion.json)、[下载清单](../results/autodl_full_20261024/mirror/download_inventory.json)、[关机错误](../results/autodl_full_20261024/mirror/job/operator_shutdown.json)
- [执行设计与授权记录](autodl_p1_full_training.md)

本次完成的是 **P1 F0/F1 完整训练与完整性核验**。训练损失不能证明 F1 优于 F0，也不能代表深度精度。后续应在确认停机、取回权重并核验 SHA 后，使用已有开发集评测入口和预先固定的 P1 判据；本报告没有运行该评测，没有访问测试集，也没有启动后续训练。
