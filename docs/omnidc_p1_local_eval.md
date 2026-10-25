# OmniDC P1 本机开发集评测

**P1 gate 未通过。** 不进入 P2，不据此增加 GPU 支出；先检查失败项并修订后续研究安排。

这是一轮已完成的开发集诊断，不能写作正式 test 结果或多种子结论。下文直接引用保存的 gate 决策、统计量和状态，不重算阈值。

## Gate 与统计范围

| 比较 | 数据 / 人群 / 区域 | 参考 → 候选 | 参考 AbsRel | 候选 AbsRel | 绝对变化 | 相对变化 | 绝对变化 CI | 统计状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | mirror3d_nyu / val / C | F0 → F1 | 0.76547692 | 0.071148702 | -0.69432821 | -90.7053% | [-0.94423854, -0.44501076] | ok |
| G2 | hammer / all_dev / C | F0 → F1 | 0.20764686 | 0.15141286 | -0.056233996 | -27.0816% | [-0.083031451, -0.03030823] | ok |
| G3_nyu | mirror3d_nyu / val / clean | R0 → F1 | 0.013106094 | 0.04221018 | 0.029104086 | 222.0653% | [0.011314365, 0.051642504] | ok |
| G3_hammer | hammer / all_dev / clean | R0 → F1 | 0.02809083 | 0.050065973 | 0.021975143 | 78.2289% | [0.01103139, 0.034072525] | ok |

| 比较 | 有效场景 scene_count | 非空图 images | 空区域图 excluded_empty_images | 失效图 n_invalid | 重采样单位 | confidence | 重采样次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | 16 | 17 | 2 | 未单列 | scene | 0.95 | 10000 |
| G2 | 10 | 1134 | 9 | 未单列 | scene | 0.95 | 10000 |
| G3_nyu | 16 | 16 | 3 | 未单列 | scene | 0.95 | 10000 |
| G3_hammer | 10 | 1134 | 9 | 未单列 | scene | 0.95 | 10000 |

变化定义为候选减参考，负值代表误差下降；CI 对应绝对 AbsRel 差值，不能当作相对变化的 CI。有效场景、非空图、空区域图和失效图数均使用 gate 原键；原记录未提供 n_invalid 时写“未单列”，不补成 0。NA 不是零，也不是通过。单场景时若 unit=frame_block，应按帧块 bootstrap 解读，不能声称多个独立场景支持。

G1：通过；G2：通过；G3（两个数据集联合规则）：未通过。gate 没有单独存储两个 G3 子项的通过标记，本报告不自行补算。

G1 的已记录阈值为相对变化 ≤ -30.0000%；G2 要求 F1 的 C AbsRel 点估计低于 F0；G3 沿用既定“未明显更差”规则（差值 CI 下界 > 0 且平均相对上升 > 20% 才记为明显更差）。G3 的 clean 限于 NYU mirror_only / HAMMER objects_only 范围，不能解释成全图非劣性证明。

gate 随机种子：20261025；完整统计与失效身份见 [results/p1_gate_20261025_034550/gate.json](<D:/Python_Script/TrustPrompt/results/p1_gate_20261025_034550/gate.json>)（SHA256 `2d2b35b131e2ee15a8912493d0eda6863f9203e62b8fc5fc15b45894342df38f`）。

## 实际样本与评测口径

| 组 | 图数 | 场景数 | train / val 图数 | 传感器图数 | 缓存报告无效预测像素 | 缓存裁剪像素 |
| --- | --- | --- | --- | --- | --- | --- |
| f0_nyu | 30 | 25 | 11 / 19 | {"kinect1": 30} | 0 | 0 |
| f0_hammer | 1143 | 10 | 816 / 327 | {"d435": 381, "helios": 381, "l515": 381} | 0 | 0 |
| f1_nyu | 30 | 25 | 11 / 19 | {"kinect1": 30} | 0 | 0 |
| f1_hammer | 1143 | 10 | 816 / 327 | {"d435": 381, "helios": 381, "l515": 381} | 0 | 0 |

四组分别代表两模型在两数据集上的预测；不能相加后声称是四份独立样本。train 在此是独立场景的开发诊断分区，不表示用这些 NYU/HAMMER 图进行本轮微调。G1/G3-NYU 使用 val，HAMMER 使用 all_dev；训练/验证按场景隔离，test 未读。

主指标为米制深度，不做尺度对齐。NYU 使用公开 registered-Z 真值，strict_rgb_z_qualified=false，因此只能作 registered-Z 开发诊断，不能声称已经核定严格 RGB 相机轴向 Z。HAMMER 使用已核定相机轴向 Z 的口径。区域固定且与方法无关；空区域为 NA，无效预测显式计数并按现有策略传播，不通过静默过滤改善指标。

本轮只有一个训练随机种子 `20261024`；评测随机种子 `20261024`。CI 反映已保存的样本/场景重采样不确定性，不包含训练种子间方差。

| 数据集 | R0 裁剪图数 | 报告裁剪像素 | 全图 clean 饱和交集 | focus clean 饱和交集 | focus C 饱和交集 |
| --- | --- | --- | --- | --- | --- |
| nyu | 0 | 0 | 0 | 0 | 0 |
| hammer | 40 | 15420 | 0 | 0 | 3 |

R0 旧 uint16 缓存在约 13.107 m 饱和；审计显示本次全图和 focus clean 的饱和交集均为 0。这支持解释 G3 的 clean 比较，但不消除量化精度差异，也不能推广为所有区域无影响。F0/F1 保存 float32，保留无效值和 >13 m 值。G1/G2 使用 F1 与 F0，未以 R0 作为这两个 gate 的参考。

## 耗时与显存

| 组 | predict 总秒 | 均值 / 中位 / 最大秒 | 调用数/秒 | 缓存墙钟秒 | train / val 指标秒 |
| --- | --- | --- | --- | --- | --- |
| f0_nyu | 21.1739 | 0.70579668 / 0.68163685 / 1.4831307 | 1.4168386 | 62.694175 | 4.2062373 / 4.9257978 |
| f0_hammer | 2637.3112 | 2.3073589 / 1.9183069 / 6.2459975 | 0.43339595 | 3451.1506 | 347.79637 / 212.8954 |
| f1_nyu | 22.361447 | 0.74538158 / 0.69779755 / 1.4635723 | 1.3415947 | 64.030671 | 3.5693092 / 4.8971963 |
| f1_hammer | 2447.3207 | 2.1411379 / 1.6967975 / 6.1420301 | 0.46704137 | 3079.0888 | 759.40798 / 181.99573 |

| 组 | torch peak allocated GiB | 已报告 / 缺失帧数 | 运行设备 |
| --- | --- | --- | --- |
| f0_nyu | 2.2431459 | 30 / 0 | {"cuda": "12.6", "device": "cuda", "torch": "2.11.0+cu126"} |
| f0_hammer | 3.8010235 | 1143 / 0 | {"cuda": "12.6", "device": "cuda", "torch": "2.11.0+cu126"} |
| f1_nyu | 2.2431459 | 30 / 0 | {"cuda": "12.6", "device": "cuda", "torch": "2.11.0+cu126"} |
| f1_hammer | 3.8010235 | 1143 / 0 | {"cuda": "12.6", "device": "cuda", "torch": "2.11.0+cu126"} |

predict 总秒为逐帧同步的 predict 调用计时之和，排除模型加载、样本读取、输入准备、序列化和指标计算；调用数/秒不是端到端吞吐。缓存墙钟从 receipt 开始到缓存完成，包含模型加载、IO 和序列化，在指标评测之前结束。train / val 指标秒仅计缓存预测读取、指标计算和评测输出。上述计时可能嵌套，不能直接相加；也不代表 GPU 租赁计费时长。

显存为逐图重置后 torch.cuda.max_memory_allocated 的最大值；不是 reserved、NVML 或设备总使用量。缺失帧数保留，不能把 NA 当作 0 GiB。

## 权重、加载验证与可追溯证据

| 模型 | 训练 run | step | final.pt bytes | final.pt SHA256 |
| --- | --- | --- | --- | --- |
| F0 | results/p1_f0_20261025_011748 | 16666 | 341916195 | 7fa22ec1c7e9f7059362031dd8556fea9e399525679395b1069dc393eab8e553 |
| F1 | results/p1_f1_20261024_224929 | 16666 | 341916195 | 2749fbca74a180eef8cf2addee42f23bf6036cbaeb121c1a7e2d9d680963f66c |

传输验证：[results/autodl_full_20261024/final_transfer_verified.json](<D:/Python_Script/TrustPrompt/results/autodl_full_20261024/final_transfer_verified.json>)，stage=`local_copy_verified`，receipt_final_sha_match=`True`；完成时间 `2026-10-25T05:25:04.552686+00:00`。传输包 SHA256：`0dcfced49832cb5ec94ccfe00a4d81440113c9e6edd6899983f46496e6b991bc`。final SHA 由已核验的传输与缓存记录交叉绑定；本报告脚本不重新反序列化权重。

两模型的真实 final.pt 已由生产入口 FineTunedOmniDC 加载并完成四组真实开发预测，其模型版本、frozen 基座版本、load_info、训练 receipt/config/frames SHA 与源码 SHA 保存在 summary.caches[*].model_version。代码没有独立的 load_finetuned 同名函数；验证的是实际 FineTunedOmniDC 入口。

CPU 测试共 25 项，来自两次完整通过的 17 + 8；未把此前中断的联合测试计入通过数。8 项保存→加载回归使用 CPU 合成夹具，覆盖 final 结构、冻结 DAv2 排除、参数恢复与错误合同拒绝；不能替代上面的真实权重加载证据。

- `gate`：17 passed，116.5 秒；命令：`.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_p1_gate.py`。
- `saved_checkpoint_cpu_roundtrip`：8 passed，15.22 秒；命令：`.venvs/baselines/Scripts/python.exe -u -m pytest tests/test_omnidc_finetuned_roundtrip.py -q`。

按用户最新通知，AutoDL 已无卡且已告知可关机；平台停机尚未确认。SSH 关闭、本地评测完成或训练结束都不能证明平台已停止计费。

汇总：[results/p1_local_eval_20261025/summary.json](<D:/Python_Script/TrustPrompt/results/p1_local_eval_20261025/summary.json>)，SHA256 `26e3b83d4451ff4f1944c85165303211da8a0bf47fcca0e46deb750f0084b6a8`，生成时间 `2026-10-25T07:46:47.821536+00:00`。评测前 Git commit：`c2aca04584beb116cc5b839a15d0933268862fac`；未提交源码差异由 preflight 与每次运行的源码 SHA/快照记录，不能只凭 commit 标识本轮代码。

最终协议审计包含两模型所有图的样本 SHA、区域分母与真值资格一致性；完整 evidence 索引含 76 条文件记录。本报告只读取汇总 metadata，并未再次读取数据、模型张量或执行推理。

复现各组评测的原始命令：

- `f0_nyu`：[cache/p1_f0_final_nyu_dev/receipt.json](<D:/Python_Script/TrustPrompt/cache/p1_f0_final_nyu_dev/receipt.json>)；原始 command：`["D:\\Python_Script\\TrustPrompt\\.venvs\\baselines\\Scripts\\python.exe", "-m", "scripts.cache_omnidc_finetuned", "--run-dir", "results/p1_f0_20261025_011748", "--variant", "f0", "--checkpoint", "final.pt", "--checkpoint-sha256", "7fa22ec1c7e9f7059362031dd8556fea9e399525679395b1069dc393eab8e553", "--dataset", "nyu", "--split", "val", "--split", "train", "--output", "cache\\p1_f0_final_nyu_dev", "--evaluate"]`。
- `f0_hammer`：[cache/p1_f0_final_hammer_dev/receipt.json](<D:/Python_Script/TrustPrompt/cache/p1_f0_final_hammer_dev/receipt.json>)；原始 command：`["D:\\Python_Script\\TrustPrompt\\.venvs\\baselines\\Scripts\\python.exe", "-m", "scripts.cache_omnidc_finetuned", "--run-dir", "results/p1_f0_20261025_011748", "--variant", "f0", "--checkpoint", "final.pt", "--checkpoint-sha256", "7fa22ec1c7e9f7059362031dd8556fea9e399525679395b1069dc393eab8e553", "--dataset", "hammer", "--split", "val", "--split", "train", "--output", "cache\\p1_f0_final_hammer_dev", "--evaluate"]`。
- `f1_nyu`：[cache/p1_f1_final_nyu_dev/receipt.json](<D:/Python_Script/TrustPrompt/cache/p1_f1_final_nyu_dev/receipt.json>)；原始 command：`["D:\\Python_Script\\TrustPrompt\\.venvs\\baselines\\Scripts\\python.exe", "-m", "scripts.cache_omnidc_finetuned", "--run-dir", "results/p1_f1_20261024_224929", "--variant", "f1", "--checkpoint", "final.pt", "--checkpoint-sha256", "2749fbca74a180eef8cf2addee42f23bf6036cbaeb121c1a7e2d9d680963f66c", "--dataset", "nyu", "--split", "val", "--split", "train", "--output", "cache\\p1_f1_final_nyu_dev", "--evaluate"]`。
- `f1_hammer`：[cache/p1_f1_final_hammer_dev/receipt.json](<D:/Python_Script/TrustPrompt/cache/p1_f1_final_hammer_dev/receipt.json>)；原始 command：`["D:\\Python_Script\\TrustPrompt\\.venvs\\baselines\\Scripts\\python.exe", "-m", "scripts.cache_omnidc_finetuned", "--run-dir", "results/p1_f1_20261024_224929", "--variant", "f1", "--checkpoint", "final.pt", "--checkpoint-sha256", "2749fbca74a180eef8cf2addee42f23bf6036cbaeb121c1a7e2d9d680963f66c", "--dataset", "hammer", "--split", "val", "--split", "train", "--output", "cache\\p1_f1_final_hammer_dev", "--evaluate"]`。

报告生成命令（JSON argv，避免 shell 转义歧义）：

```json
["D:\\Python_Script\\TrustPrompt\\.venv\\Scripts\\python.exe", "D:\\Python_Script\\TrustPrompt\\results\\p1_local_eval_20261025\\write_results_report.py"]
```

报告脚本 SHA256：`40b527c395b5c91b99be9523ed7efa46bcf2dab1dd280508e6c7b21af85bbb94`；报告生成时间 UTC：`2026-10-25T07:46:48.233266+00:00`。
