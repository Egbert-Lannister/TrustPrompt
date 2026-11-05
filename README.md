# TrustPrompt

研究**带深度提示的补全模型**在镜面、玻璃和透明物体上的行为。这类模型（PromptDA、PriorDA、OMNI-DC、CDM、LingBot-Depth、Any2Full、Marigold-DC，以及测试时自适应方法 CAPA）默认信任传感器深度，而在这些表面上，传感器会成片测错，绝大多数是测远。

> **状态（2026-11-04）**：方法 v1 已冻结，并在 test 集（NYU Mirror3D test 场景、HAMMER 场景 12–14）上按预注册的比较族做了唯一一次正式评测（[`docs/holm_test_v1.md`](docs/holm_test_v1.md)、[`docs/test_results.md`](docs/test_results.md)、[`docs/conformal_test.md`](docs/conformal_test.md)）。其余结果都来自开发集（NYU Mirror3D train/val，HAMMER 场景 2–11）。每个预检的判读规则都在运行前写定并提交。
>
> 当前总结：[分析报告草稿 `docs/analysis_paper.md`](docs/analysis_paper.md)。

## 目前的结论

1. **继承普遍存在。**
   - NYU 镜面上，6 个前馈模型的继承率（IR）在 0.17–0.93 之间。
   - 测试时自适应会进一步放大继承：CAPA 把 IR 从冻结先验的 0.07 推到 0.95，镜面误差扩大 6.6 倍。
   - 继承程度随污染面积上升。HAMMER 桌面小物体上的继承较轻。
2. **唯一稳定有效的区分信号**，是读数与冻结单目先验之间的不一致度：NYU 全图 AUROC 0.95，HAMMER 0.74。方法 v1 基于这个信号：可信度检测器 T、conformal 阈值、屏蔽不可信读数后用先验填充，可以接在任意补全模型前面。在 NYU val 上，6 个模型都显著改善。
   - **正式 test 结果**（冻结后只读一次，Holm α = 0.05）：12 个比较中 10 个显著。NYU 镜面上 C 区 AbsRel 下降 0.32–0.63，HAMMER 物体上 5 个模型下降 0.012–0.065。
   - 例外一：CDM 在 NYU 上有 1 张图出现无效预测，按规则记为 NA。
   - 例外二：PromptDA-L 在 HAMMER 上没有改善（+0.0007），可改善空间本来就只有 0.005。
   - 代价：PromptDA-L 在两个数据集上的干净区都明显变差（NYU +178%，HAMMER +51%），其余 5 个模型没有明显变差。
   - conformal 保证在 NYU 上成立（平均漏检 0.027），在 HAMMER 新场景上不成立（0.129 > 0.1）。详见报告第 7 节。
3. **六条寻找"局部"信号的路线都没有成立**，每条都有预注册规则：

   | 路线 | 判定 | 报告 |
   |---|---|---|
   | 把检测训练进 OMNI-DC（P1） | 未通过：干净区代价 G3 不过 | [omnidc_p1_gate](docs/omnidc_p1_gate.md)、[耦合分析](docs/omnidc_p1_coupling.md)、[F1+](docs/omnidc_p1plus_gate.md) |
   | 读数当上界（Marigold-DC 引导） | 未通过（M1） | [marigold_guidance_plan](docs/marigold_guidance_plan.md) |
   | 错读数是第二层表面 | 不成立（0.439，CI 0.372–0.504） | [layer2_check](docs/layer2_check.md) |
   | 自适应过程中的残差变化（CAPA） | 继承成立，信号不成立 | [tta_precheck](docs/tta_precheck.md) |
   | 先验自带的不确定度 Σ | 不成立 | [tr_precheck](docs/tr_precheck.md) |
   | 模型内部定位（PromptDA 激活修补） | 无效：HAMMER 上没有效应；NYU 上照抄集中在最细的两个融合分支 | [patch_precheck](docs/patch_precheck.md) |

4. **把方法 v1 的掩码接到 CAPA 前**（补充实验 S2）：CAPA 的错误区误差在 NYU 下降 87%，在 HAMMER 下降 44%；HAMMER 上没有代价，NYU 上会删掉镜面里约 3% 的正确读数（[tta_tmask](docs/tta_tmask.md)）。test 上重现：NYU 下降 73%（CI 0.52–0.96），HAMMER 下降 60%（9 张图），两个数据集的干净区都几乎没有代价（[tta_test](docs/tta_test.md)）。
5. **解释：局部自洽。** 镜子里的 RGB 和传感器读数描述的是同一个虚像，局部找不到矛盾；只有全局先验能识别。详见报告第 6 节。

## 目录

| 位置 | 内容 |
|---|---|
| `tp/` | 数据结构、区域定义与指标（`tp/eval`）、对齐、仿真、OMNI-DC 训练组件 |
| `baselines/` | 各补全模型的统一封装（固定源码提交和权重） |
| `scripts/` | 缓存、评测、训练和各个预检脚本 |
| `configs/` | 数据、基线、评测和预检配置 |
| `tests/` | 单元测试 |
| `docs/` | 计划、进展、决策与各阶段报告 |
| `data/`、`cache/`、`checkpoints/`、`external/`、`results/` | 数据、预测缓存、权重、外部源码、运行结果（都不进 git） |

## 环境

在 PowerShell 中进入本目录：

```powershell
Set-Location D:\Python_Script\TrustPrompt
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

- `.venv`：用 `python -m venv --without-pip --system-site-packages .venv` 建立，复用 Anaconda 已装的包（Python 3.12.7、torch 2.11.0+cu126）。PromptDA 和评测脚本用它。完整版本见 `docs/environment.json`。
- `.venvs/baselines`：GPU 基线环境。CAPA 预检用它，另外需要 `pip install peft`。

新机器建立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

各基线的源码和权重由 `python -m scripts.setup_baselines` 按 `configs/baselines/registry.json` 获取并核对 SHA256。只使用公开、无门槛的仓库；许可证会记录下来，但不会代替用户接受。

## 复现最近的预检

都是开发集，test 会被拒绝。每次运行都写入新的 `results/<名称>_<时间>/` 目录，其中包括 config、seed、命令、源码快照、diff、receipt（含模型版本和权重 SHA256）和指标。

```powershell
# 第二层检验（CPU）
.\.venv\Scripts\python.exe -m scripts.layer2_check
# CAPA 继承与残差变化（GPU；先跑 --limit 2 做工程检查）
.\.venvs\baselines\Scripts\python.exe -m scripts.tta_precheck
# 先验不确定度信赖域（GPU；复用上一步的 CAPA 运行结果）
.\.venvs\baselines\Scripts\python.exe -m scripts.tr_precheck
# PromptDA-L 激活修补（GPU）
.\.venv\Scripts\python.exe -m scripts.patch_precheck
```

更早阶段（基线缓存、T 的训练、方法 v1 的 Holm 检验、OMNI-DC P1）的命令记录在各自的报告和 `docs/progress.md` 中。

## 评测口径

- 深度一律是米制的轴向 Z。主评测不对 GT 做尺度或仿射对齐。区域由未补洞的提示和 GT 决定。
- 污染区 C：|P − GT| / GT > τ（τ = 0.05）。主范围：NYU 取镜面像素，HAMMER 取物体像素（与去掉物体后的真值相差 > 1 mm）。干净区指正确读数，并去掉 C 边界外 5 像素的过渡带。
- 继承率 IR：C 区中，预测与读数相差 < 5%、而与 GT 相差 > 5% 的像素比例。危险高估率 DOR：C 区及过渡带中，预测 > 1.10 × GT 的像素比例。
- 空区域记为 NA；无效预测单独计数，不从分母中剔除。按场景做 bootstrap，并在场景层面聚合。
- NYU 的配准 Z 还没有认证为严格的 RGB 相机 Z（`docs/intrinsics_audit.md`）；Mirror3D 镜面真值由平面拟合得到（见报告第 7 节的局限）。
- 划分按场景进行：NYU 为 55/18/23 个场景，HAMMER 场景 2–10 为 train、11 为 val、12–14 为 test。test 只能通过 `--formal-test` 读取，且必须通过冻结核对。

## 数据位置

所有数据都在本目录内，都不进 git：

| 内容 | 位置 |
|---|---|
| HAMMER 原始子集 | `data/source_cache/hammer_subset_v1` |
| Mirror3D 元数据与标注 | `data/source_cache/mirror3d` |
| 处理后的样本 | `data/processed/{mirror3d_dev_v1,hammer_dev_v1,mirror3d_test_v1,hammer_test_v1}` |
| 预测缓存 | `cache/` |
| 权重 | `checkpoints/` |
| 外部源码（固定提交） | `external/` |

GLARE 只作为参考：本项目不 import、不修改 GLARE 的任何代码或结果。

## 文档索引

- [`docs/analysis_paper.md`](docs/analysis_paper.md)：分析报告草稿，当前的总结。
- [`docs/progress.md`](docs/progress.md)：实际完成的工作、运行结果、阻塞与下一步。
- [`docs/decisions.md`](docs/decisions.md)：口径决定、预注册规则、计划修订。
- 计划：[`plan_revision_v2.md`](docs/plan_revision_v2.md)（方法 v1）、[`plan_omnidc.md`](docs/plan_omnidc.md)（OMNI-DC 内化）、[`plan_layered.md`](docs/plan_layered.md)（分层，已放弃）、[`plan_tta.md`](docs/plan_tta.md)（CAPA 与信赖域）、[`plan_patching.md`](docs/plan_patching.md)（激活修补）。
- 正式 test：[`plan_test.md`](docs/plan_test.md)（预注册）、[`freeze_method_v1.md`](docs/freeze_method_v1.md)、[`holm_test_v1.md`](docs/holm_test_v1.md)、[`test_results.md`](docs/test_results.md)、[`conformal_test.md`](docs/conformal_test.md)、[`tta_test.md`](docs/tta_test.md)。
- 方法 v1 与基线：[`baseline_results.md`](docs/baseline_results.md)、[`holm_plugin_v1.md`](docs/holm_plugin_v1.md)、[`holm_plugin_cf_v1.md`](docs/holm_plugin_cf_v1.md)、[`conformal_results_d20.md`](docs/conformal_results_d20.md)、[`c1_analysis.md`](docs/c1_analysis.md)。
- 相关工作与数据：[`related_work_audit.md`](docs/related_work_audit.md)、[`data_inventory.md`](docs/data_inventory.md)、[`capture_plan.md`](docs/capture_plan.md)。
- AutoDL 训练记录（历史）：[`autodl_p1_full_results.md`](docs/autodl_p1_full_results.md)、[`omnidc_p1_transfer.md`](docs/omnidc_p1_transfer.md)。

## 正式 test 评测的复现

```powershell
.\.venv\Scripts\python.exe -m scripts.run_test_eval --dry-run   # 列出步骤与完成情况
.\.venv\Scripts\python.exe -m scripts.run_test_eval             # 全部步骤，可断点续跑
```

每个读取 test 的脚本都需要 `--formal-test`，并先用 `tp.formal.guard()` 核对 `configs/freeze_method_v1.json` 中 37 个冻结文件的 SHA256。不加这个开关时 test 照旧锁定。

## 下一步（待定）

- test 已读取，方法 v1 不再修改；CAPA 的 test 对比也已完成。
- 新的方法改进只能在开发集上进行，而且需要新的 test 数据，这次的 test 不能再用于选择方法。
