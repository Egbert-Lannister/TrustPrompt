# P1 final 权重取回与本地评测准备

2026-10-25（America/New_York）。两个完整训练运行目录已从用户提供的 AutoDL 实例取回，未上传开发集或在服务器推理。按用户最新说明，传输时实例处于无卡模式；未知无卡计费单价，不按 7 元/GPU 小时推算这段费用。

## 已核对的本地产物

| 运行目录 | final.pt 字节数 | SHA-256 |
|---|---:|---|
| `results/p1_f0_20261025_011748/` | 341,916,195 | `7fa22ec1c7e9f7059362031dd8556fea9e399525679395b1069dc393eab8e553` |
| `results/p1_f1_20261024_224929/` | 341,916,195 | `2749fbca74a180eef8cf2addee42f23bf6036cbaeb121c1a7e2d9d680963f66c` |

每个目录包含 final.pt、receipt.json、config.json、training_frames.json、log.jsonl、command.txt、working_tree.diff、source_snapshot.zip、deployment_manifest.json、deployment_working_tree.diff，共 20 个文件。逐文件大小与 SHA-256 同时匹配原始训练终态、小产物清单及归档内容；两份 final.pt 也与各自 receipt.final_sha256 一致。原始归档 686,274,560 字节，SHA-256 `0dcfced49832cb5ec94ccfe00a4d81440113c9e6edd6899983f46496e6b991bc`。

两个 last.pt 未下载，保留在服务器上用于可能的续训。校验完成后已告知用户“现在可以关闭无卡实例”，并关闭本次 SSH/SFTP 会话。依照用户最新“告诉我关机”的指令，本轮没有主动关闭实例；平台停机和账单状态仍待用户确认。此前自动关机 Exec format error 的记录保留，不能用旧请求作成功证明。

## 加载与判据测试

- `tests/test_p1_gate.py`：17 passed，116.50 秒。包括实际合成 sample→evaluator JSON→gate 的通过、G1/G2/G3 独立失败、无效值、缺失/未完成数据，以及严格 N1 边界。
- `tests/test_omnidc_finetuned_roundtrip.py`：8 passed，15.22 秒。保存检查点→生产加载器→CPU 实际预测，验证 trainable 权重与 buffer、冻结 DAv2 排除、错误权重合同拒绝。
- 当前实际入口为 `FineTunedOmniDC`，仓库没有 `load_finetuned` 函数，因此测试走现有生产入口，没有添加重复加载器。
- 新旧测试组合一次在 38 个进度点后中止，未报告该组合整组通过；上述两次有明确完成结果。
- 真实 F0 NYU 与 F1 HAMMER 单帧 GPU 推理、指标计算均通过，invalid=0。F0 初次 smoke 的 wandb 临时目录退出清理有权限警告（进程 exit 0），后续将 TEMP/TMP 指向项目内独立目录；不改变模型和数值配置。

## 全量评测范围与命令

只读 NYU val 19 张及 train 留出 11 张，HAMMER scene2–scene11 的 1,143 个开发样本。原始 E 盘当前未挂载，但已物化的 D 盘开发样本齐全；不读 test，不下载额外数据集。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\step16_p1_eval.ps1
```

执行环境：MKL_THREADING_LAYER=SEQUENTIAL，OMP/MKL/OPENBLAS_NUM_THREADS=4，WANDB_MODE=disabled。全量启动时 Windows PowerShell 5 的 Get-PSDrive.Free 返回 null，原脚本错误当作磁盘已满，在推理前退出；`common.ps1` 增加 .NET DriveInfo.AvailableFreeSpace 回退，保留原 6 GiB 最小空间阈值，实测约 22.4 GiB 可用后重启。失败日志单独保留，没有覆盖结果或更改评测配置/判据。

后续四个完整缓存与指标输出均已完成，驱动 exit 0，总墙钟 02:16:59；F0/F1/R0 的 sample SHA、区域分母和 GT 资格交叉核对全部通过。G1/G2 通过、G3 未通过，P1 停止。正式本轮开发诊断见 [本地评测报告](omnidc_p1_local_eval.md)、[固定判据报告](omnidc_p1_gate.md)；本节的 smoke 和传输验证仍单独归类为工程检查。

## 证据

- [传输验证回执](../results/autodl_full_20261024/final_transfer_verified.json)
- [本地评测预检、源码版本与测试记录](../results/p1_local_eval_20261025/preflight.json)
- [开发协议与 R0 截断审计](../results/p1_local_eval_20261025/protocol_audit.json)
- [全量运行日志](../results/p1_local_eval_20261025/step16.log)

R0 HAMMER 旧缓存共 15,420 个截断像素，其中 clean 范围为 0，不影响 G3；对象污染范围有 3 个，相关 R0 描述性比较需注明。NYU 使用发布版 registered-Z，strict_rgb_z_qualified=false；评测保持既有开发口径，不宣称已认证的精确 RGB 相机轴向 Z。
