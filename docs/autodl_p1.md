# AutoDL P1 执行记录（2026-10-24）

当前状态：**无卡环境准备进行中；云端 GPU pilot 和正式 F0/F1 均未开始。**

用户已提供现有 AutoDL 实例并授权 SSH 操作、公开数据下载和首个 1 小时 pilot。实例目前无 GPU；由用户在准备就绪后加载卡。完整 F0/F1 要在 pilot 实测报告与预算确认后开始。密码仅通过交互式登录使用，不存储于项目文件。没有修改服务器的登录认证设置。

## 核验与来源

- 开始时本地提交 `1477a59b246ea5f2b3fa25368e3e4b430a5875c4`，工作树干净；`docs/omnidc_p1_smoke.md` 的本机 S1–S6 通过记录存在。它只证明小裁剪本机训练可用，不能替代 480×640、batch 6 的云端实测。
- 远端 Ubuntu 22.04.5、Python 3.12.3；无卡 cgroup 限制为 0.5 CPU、2 GiB RAM。系统盘约 30 GiB、数据盘约 50 GiB，首次检查均基本空闲。宿主机 `free` 显示的内存不作为容器可用内存。
- 项目：`/root/autodl-tmp/TrustPrompt`；固定 CUDA 环境：`/root/trustprompt-envs/p1-cu126`。不修改平台自带 torch 2.8/cu128 环境。
- 官方 PyTorch、GitHub、Apple Hypersim 地址可访问。Hugging Face 直连失败，用户建议并同意 `HF_ENDPOINT=https://hf-mirror.com`；镜像固定 revision 链接 HEAD 200，权重仍必须通过 registry 的大小与 SHA-256 校验。
- 从已有官方 ZIP 索引复算：Hypersim 实际选定数据 3,111,797,066 B，索引 118,467,991 B，两个权重 3,024,484,918 B，合计约 6.25 GB；这是传输/原始文件估计，不含环境、解压后文件、检查点和临时空间。远端 plan 阶段会重新核对大小。
- 只上传明确列出的源码、配置与测试；不上传本地数据、权重、旧结果、凭据或 Git 历史。`scripts/build_p1_payload.py` 记录本地提交、部署实际字节哈希与工作树差异；全部文本在归档中转换为 LF。

## 部署前修复

- `p1_setup.sh` 支持 `--cpu-only` 和分阶段恢复，同一个 venv 安装目标 CUDA wheels。无卡阶段不实例化大模型，下载并发最多 2，补齐固定 OpenCV headless 版本。
- 帧清单校验改为 canonical JSON SHA-256，避免 Windows CRLF 与 Linux LF 造成假失败。预期值 `ba4768d0f8a0af30f30bc973148d1a6befc830ebf6cabf2becbd28193985cd2f`，选帧规则与训练集不变。
- HTTP Range 下载器先核对响应状态与范围，再有限读取正文；服务器忽略 Range 时立即停止，避免误下载整场景 ZIP。
- pilot 外层限制 60 分钟进程墙钟时间，最多额外 30 秒退出宽限；训练循环预留 60 秒保存时间，并在不足 2000 步时也保存可续训的 `last.pt`。这不会自动关闭 AutoDL 实例，不能等同于账单 GPU 小时或 GPU kernel 时间。
- 续训需变体、配置、源码、数据选择与模型 registry 哈希一致，并且必须有检查点；非有限梯度立即报错。训练方法、训练帧、步数和损失定义不变。
- 部署运行回执保留本地来源提交与实际上传源码哈希；无 `.git` 的远端 `git_commit` 保持 null，不伪造仓库状态。

## 本轮执行记录

- 部署归档 v2：42 个源码/配置/测试文件；`results/autodl_p1_20261024/p1_source_v2.tar.gz` SHA-256 `fe717dc0b9502b8c3017ccd361ffae10bb31918c2b9bc214c1daed500d3c5128`。远端逐文件校验与两个 shell 的 `bash -n` 已通过。早期归档保留，实际训练将使用 v2 或之后明确记录的版本。
- 本机针对 P1 的检查：**58 passed, 1 deselected，86.33 s**。命令为 `p1_setup.sh` 的测试清单，在 `.venvs/baselines` 中运行，设置 OMP/MKL/OpenBLAS 线程数为 1。
- 唯一排除项 `tests/test_onesided.py::test_gradients_through_omnidc_ddi_with_the_hinge` 属于当前 F0/F1 不使用的 P2 模块 B。原测试包含 640 个差分输入，至少需要 1,280 次数值前向；本轮完整检查曾停在此项而主动中止，不算通过。保留测试本体和精度，启用模块 B 前在合适资源上重跑。
- 首次 env 安装因官方 R2 大文件速度约 120–150 KB/s 而主动中断，阶段退出码 143，原日志保留。已完成的公开 NVIDIA wheel 通过硬链接保留，避免重复下载。
- 官方索引确认 torch 2.11.0+cu126 Linux cp312 wheel SHA-256 为 `9235c0c2f5032d1f8af75dd97493f3ad12aaedf81c0f3fab2d13c1d90bc54ff9`。改从阿里云公开 PyTorch wheel 镜像分段获取同一文件，4 个有界线程、每段最多 8 MiB；整文件哈希匹配才交给 pip。torchvision 0.26.0 和 triton 3.6.0 也从官方索引核对哈希。版本不变。
- 恢复脚本 `results/autodl_p1_20261024/prepare_autodl.sh` 和两个 wheel 获取脚本保留于本地及远端操作目录。它只完成 CPU setup，不包含训练命令。

## 可重复命令

本地生成新的唯一归档名：

```powershell
python -m scripts.build_p1_payload --output results/autodl_p1_20261024/p1_source_final.tar.gz
```

远端使用固定路径，环境与公共下载均可按失败阶段恢复：

```bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/TrustPrompt
export TP_P1_VENV=/root/trustprompt-envs/p1-cu126
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/linux/p1_setup.sh --cpu-only --stage env
bash scripts/linux/p1_setup.sh --cpu-only --from model
```

GPU 由用户加载且环境、数据全部验证后，检查设备再运行 **仅 pilot**：

```bash
bash scripts/linux/p1_setup.sh --stage tests
bash scripts/linux/p1_run.sh pilot
```

完整训练开关 `TP_P1_ALLOW_FULL_TRAIN=1` 当前不设置。任何首次失败的训练/下载日志保留，恢复不覆盖旧实验。setup 日志与回执位于远端 `results/p1_setup/`，操作日志位于 `results/autodl_operator/`；本地归档和取回记录位于 `results/autodl_p1_20261024/`。
