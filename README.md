# TrustPrompt

独立的提示可信度深度补全实验项目。已完成 Mirror3D-NYU 的5帧真实训练集最小闭环，并扩展到 UniDepthV2-S、全局/局部鲁棒对齐。全部85项测试通过；局部对齐的真实失败已保留并报告。结果见 [smoke_results.md](docs/smoke_results.md)，正式实验尚未完成。

## 环境

在 PowerShell 中进入本目录：

```powershell
Set-Location D:\Python_Script\TrustPrompt
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

本机 `.venv` 通过 `python -m venv --without-pip --system-site-packages .venv` 建立，复用Anaconda已安装包，不依赖GLARE Python包。Python 3.12.7、NumPy1.26.4、SciPy1.13.1、h5py3.11.0、torch2.11.0+cu126；完整实测见 `docs/environment.json`。CPU闭环无需torch。

新的机器可建立独立环境并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

GPU基线另外需要与机器匹配的torch、torchvision、timm、safetensors及原模型依赖，参考 `docs/unidepth_smoke.md`；不在最小闭环自动安装。

## 已运行的基线扩展

现有cache已通过RGB、样本、预测hash和receipt身份绑定检查。UniDepth缓存的原推理耗时与评测时的读盘/对齐耗时分开记录，不将读盘速度冒称模型速度。

```powershell
.\.venv\Scripts\python.exe -m scripts.cache_unidepth --limit 1 --output cache/unidepth_preflight_v1
.\.venv\Scripts\python.exe -m scripts.cache_unidepth
.\.venv\Scripts\python.exe -m tp.eval.run_eval --config configs/mirror3d_unidepth_eval.yaml --output results/mirror3d_unidepth_smoke_v1
.\.venv\Scripts\python.exe -m tp.eval.run_eval --config configs/mirror3d_robust_eval.yaml --method robust_global --output results/mirror3d_robust_global_smoke_v1
.\.venv\Scripts\python.exe -m tp.eval.run_eval --config configs/mirror3d_robust_eval.yaml --method robust_local --output results/mirror3d_robust_local_smoke_v2
.\.venv\Scripts\python.exe -m tp.eval.compare --reference results/mirror3d_raw_prompt_smoke_v2 --candidates results/mirror3d_unidepth_smoke_v1 results/mirror3d_robust_global_smoke_v1 results/mirror3d_robust_local_smoke_v2 --output results/mirror3d_paired_comparison_v2
.\.venv\Scripts\python.exe -m scripts.summarize_smoke
```

这些目录当前已存在；命令展示原运行方式，复跑须选择新输出名并对应修改配置中的prior_cache。5帧配对bootstrap只作描述，不支持泛化/正式显著性结论。

## 数据路径与复现

修改或复制 `configs/paths.example.yaml`；已有原始标注和权重只通过路径读取。新cache、NPZ、预测和结果均保存在本目录。不存在完整NYU MAT时，小规模获取命令仅访问官方MAT指定字节范围，下载上限100MiB，不读取测试深度：

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_nyu_subset --output data/source_cache/nyu_smoke_v1.h5
.\.venv\Scripts\python.exe -m scripts.prepare_mirror3d --output data/processed/mirror3d_smoke_v1
.\.venv\Scripts\python.exe -m tp.eval.run_eval --config configs/mirror3d_smoke.yaml --method raw_prompt --output results/mirror3d_raw_prompt_smoke_v2
.\.venv\Scripts\python.exe -m tp.eval.make_tables --per-image results/mirror3d_raw_prompt_smoke_v2/per_image.json --output results/mirror3d_raw_prompt_smoke_v2/metrics.csv
```

输入cache/NPZ已存在时跳过获取/转换。所有输出目录和样本拒绝覆盖；复跑请用新的输出名。提供完整MAT时在paths配置设置 `mirror3d.mat`。读取E盘可能需要应用的只读权限，但任何新文件仍写TrustPrompt。

自 2026-10-14 起，项目不再读取 E 盘，所有数据都在本目录内（均不进 git）：

| 内容 | 位置 |
|---|---|
| HAMMER 原始子集 | `data/source_cache/hammer_subset_v1` |
| Mirror3D 元数据与标注（从 GLARE 数据只读复制） | `data/source_cache/mirror3d` |
| 预测缓存（NYU 与 HAMMER） | `cache/` |
| PromptDA 检查点 | `checkpoints/promptda` |
| PromptDA 官方源码（提交 9161547） | `external/PromptDA` |

迁移和校验用 `scripts/windows/migrate_e_to_d.ps1`；源码和检查点缺失时，用 `scripts/windows/restore_promptda.ps1` 重新获取并核对哈希。旧运行记录里的 E 盘路径是历史记录；缓存读取只使用文件夹内的相对文件名，所以移动后仍然有效。

纯合成工程测试（不可引用为真实结果）：

```powershell
.\.venv\Scripts\python.exe -m scripts.make_synthetic
.\.venv\Scripts\python.exe -m tp.eval.run_eval --config configs/synthetic_smoke.yaml --output results/synthetic_raw_prompt_smoke_v2
```

## 评测口径

- 深度以米表达轴向Z，主评测无尺度/仿射对齐；区域由未补洞提示和GT决定。
- Mirror3D采用标注N→h5索引N，场景依据修正后NYU `scenes[N]`；固定55/18/23个train/val/test场景。本次仅5个train场景。
- **NYU注册Z尚未认证为严格RGB相机Z**，结果携带 `strict_rgb_z_qualified=false`，只作工程/数据诊断。正式主表模式拒绝这类未认证数据。详见 `docs/intrinsics_audit.md`。
- 全图参考的背景来自补洞，不能视为独立第一表面GT。单独输出 `per_image_mirror_only.json`、`summary_mirror_only.json`。
- 空域NA；无效预测显式计数并传播，不静默过滤。图像→场景→数据集等权聚合。
- 原始提示全空时无法补洞，保留无效预测。raw_prompt不输出w，因此可信度指标NA。

每个run内包含 `receipt.json`、`config.json`、`split_manifest.json`、`command.txt`、`run.log`、`source_snapshot.zip`、源码hash/Git状态/diff、预测、逐图/逐场景指标及5张图。既可追溯未提交代码，也不会将新结果写回GLARE。

## 计划修订 v2 与第 1 步

计划修订见 `docs/plan_revision_v2.md`。第 1 步中需要 Windows（NYU 网络、E 盘、CUDA）的部分：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\step1_dev.ps1
```

CPU 部分（Linux 或 Windows 均可；Linux 工作区需 `TP_TEST_TMP` 指向可删除的临时目录）：

```bash
python -m pytest -q -p no:cacheprovider
python -m scripts.nyu_rgbz_bound --toolbox data/source_cache/nyu_toolbox/toolbox_nyu_depth_v2.zip --manifest data/processed/mirror3d_dev_v1/manifest.json --output results/nyu_rgbz_bound_v1
for m in raw_prompt unidepth_v2_s robust_global robust_local_v2 gate_global gate_local_v2; do
  for s in train val; do python -m tp.eval.run_eval --config configs/mirror3d_dev_eval.yaml --method $m --split $s --output results/mirror3d_dev_${m}_${s}_v1; done
done
python -m scripts.summarize_step1
```

`--split test` 在方法冻结前会被拒绝。

## 文档和状态

当前方向见 `docs/plan_revision_v2.md` §3.7（R16，2026-10-16 按 T v2 结果调整）：方法 v1（对齐 → T v1 → conformal 阈值 → 屏蔽并用 D_g 填充 → PromptDA-L）只冻结结构和规则。“错误提示是单侧约束”降为分析内容（`docs/t_v2_results.md`）。论文主线是三部分：测量（IR / DOR 与协议）、检测（可插拔的 T）、保证（conformal）。对比方法与 Holm 检验见 `docs/baselines.md`，自采数据方案见 `docs/capture_plan.md`。相关工作审计见 `docs/related_work_audit.md`，数据清单见 `docs/data_inventory.md`。

- `docs/implementation_plan.md`：原计划副本与来源说明。
- `docs/progress.md`：实际完成、运行结果、阻塞与下一步。
- `docs/decisions.md`：定义冲突、采用解释、计划修订。
- `docs/plan_revision_v2.md`：计划修订 v2（范围、日程、口径、云预算申请）。
- `docs/step1_results.md`：第 1 步结果表（由逐图 JSON 生成）。
- `docs/audit_data.json` / `.md`：实际数据核查和21帧编号回归。
- `docs/reuse_provenance.json`：选择性迁入的旧代码文件hash。
- `configs/splits/mirror3d_nyu_v1.json`：可版本控制的完整场景划分。

T v0 / v1 / v2 都已训练，但只有开发集诊断。目前没有正式实验，也没有论文主结果：测试集未读取，HAMMER 三传感器留一测试也还没做。
