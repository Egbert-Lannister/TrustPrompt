# AutoDL P1 执行记录（2026-10-24）

当前状态：**无卡准备已完成，环境、模型权重、Hypersim 数据、62 项 CPU 测试及三帧输入检查全部通过。等待用户加载 GPU；云端 pilot 和正式 F0/F1 均未开始。**

用户已提供现有 AutoDL 实例并授权 SSH 操作、公开数据下载和首个 1 小时 pilot。实例目前无 GPU；由用户在准备就绪后加载卡。完整 F0/F1 要在 pilot 实测报告与预算确认后开始。密码仅通过交互式登录使用，不存储于项目文件。没有修改服务器的登录认证设置。

## 核验与来源

- 开始时本地提交 `1477a59b246ea5f2b3fa25368e3e4b430a5875c4`，工作树干净；`docs/omnidc_p1_smoke.md` 的本机 S1–S6 通过记录存在。它只证明小裁剪本机训练可用，不能替代 480×640、batch 6 的云端实测。
- 远端 Ubuntu 22.04.5、Python 3.12.3；无卡 cgroup 限制为 0.5 CPU、2 GiB RAM。系统盘约 30 GiB、数据盘约 50 GiB，首次检查均基本空闲。宿主机 `free` 显示的内存不作为容器可用内存。
- 项目：`/root/autodl-tmp/TrustPrompt`；固定 CUDA 环境：`/root/trustprompt-envs/p1-cu126`。不修改平台自带 torch 2.8/cu128 环境。
- 官方 PyTorch、GitHub、Apple Hypersim 地址可访问。Hugging Face 直连失败，用户建议并同意 `HF_ENDPOINT=https://hf-mirror.com`；镜像固定 revision 链接 HEAD 200，权重仍必须通过 registry 的大小与 SHA-256 校验。
- 从已有官方 ZIP 索引复算：Hypersim 实际选定数据 3,111,797,066 B，索引 118,467,991 B，两个权重 3,024,484,918 B，合计约 6.25 GB；这是传输/原始文件估计，不含环境、解压后文件、检查点和临时空间。远端 plan 阶段会重新核对大小。
- 只上传明确列出的源码、配置与测试；不上传本地数据、权重、旧结果、凭据或 Git 历史。`scripts/build_p1_payload.py` 记录本地提交、部署实际字节哈希与工作树差异；全部文本在归档中转换为 LF。

## 部署前修复

- `p1_setup.sh` 支持 `--cpu-only` 和分阶段恢复，同一个 venv 安装目标 CUDA wheels。无卡阶段不实例化大模型，下载并发默认 2；资源试跑后可显式选择最多 16，补齐固定 OpenCV headless 版本。
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
- 部署代码提交为 `997e105`。torch wheel 下载 405.4 秒并通过整文件官方 SHA-256 校验；torchvision 同样通过。此前 120–150 KB/s 的官方下载日志与镜像成功回执均已取回本地。
- 训练集合独立复核：4,418 帧来自 **65 个实际训练场景**，不是原文写的 66 个；66 是官方 train 候选场景数，`ai_016_002` 经原先 ≥5% 镜面阈值筛除后无训练帧。类别数仍为 1,285/1,777/1,356；不改变任何样本或方法。该事实已同步修正主线计划。
- 补齐模型与样本来源回执：GPU 加载前验证上游实际 HEAD、tracked 源码状态和各权重的真实 SHA-256；`model_version` 保存版本及加载信息；`training_frames.json` 保存有序帧清单与实际场景划分。对应训练单测 **25 passed（54.06 s）**，包含原先 21 项和新增 4 项；现在云端 setup 清单为 62 项，另有 1 项明确排除，尚待远端执行。
- 第二次安装也遇到 R2 通用小包下载缓慢，主动终止后保留 `prepare.log`（143）；恢复脚本改为 `prepare_autodl_v2.sh`：直接安装已完成的公开 NVIDIA wheels，通用依赖走阿里云 PyPI 镜像，torch/torchvision 显式固定 `+cu126`。环境本体与研究配置不变，最后仍执行原 setup 的依赖/导入核验。
- 部署 v3：`e8da1ae0405f8bb7823d4ad37d050a38d264e870` 的干净源码归档，42 文件；压缩包 SHA-256 `4956f7d6b5f30ccd98a13954f77c62a5199fc49a0468de3034441fa369041b24`。远端逐文件校验通过；后续已由 v4 替代。
- 远端 env 阶段 **已完成**：`results/p1_setup/20261024T102552Z-2011`，`pip check` 及 torchvision CPU NMS 通过；实测 torch 2.11.0+cu126、torchvision 0.26.0+cu126、OpenCV headless 4.11.0.86，CUDA build 12.6。未加载模型到 GPU，cgroup OOM/oom_kill 均为 0。完整实际依赖以该阶段 `packages.txt` 为准，未固定的传递依赖可能与本机不同。
- model/plan 在 `results/p1_setup/20261024T103045Z-2060` 完成，semantic 为并发试跑主动中断；后续从新阶段恢复。远端后台进程和日志记录于本地 `results/autodl_p1_20261024/handoff.json`，该文件不含密码。
- 定期自动访问服务器的提议被自动审批拒绝，未创建自动任务；继续在当前会话跟进。也未安装持久化 root SSH 密钥。
- 公开 OMNI-DC 和 DAv2-L 两个权重 **大小/SHA-256 均验证通过**，上游源码固定版本验证通过；回执已取回 `results/autodl_p1_20261024/baselines_setup_receipt.json`。远端 plan 完成：语义 14,116 文件/0.250507775 GB，RGB+depth 候选上界 28,232 文件/4.872510174 GB，符合 40 GB 上限。
- 下载并发试跑（每档 256 个真实公开语义文件，不是模型实验）：2/4/8/16 线程分别约 2.43/5.72/7.55/15.90 文件每秒。16 线程进程峰值 RSS 302.96 MiB，平均 cgroup CPU 增量 0.176 核，OOM/oom_kill 均 0。因此把 CPU 下载线程可选上限调整到 16，默认仍为 2，本次后续显式用 16。目标最大成员不足 1 MiB，ZIP 均 stored；并发不复制完整索引。cgroup memory.current 含可回收文件缓存，不能把它误当成进程常驻内存。
- 原 semantic 阶段为试跑而主动中断，退出 143；555 条已完成记录解析正常。两轮试跑新增文件进入同一经 CRC/SHA 记录的缓存，后续从 semantic 恢复会验证并复用，不改变数据选择。试跑日志、完整指标和样本名单保存在 `results/autodl_p1_20261024/worker_probe*.json`。
- 当前部署 v4：干净提交 `8515378a160c207ae11e94677836ef2917f5162c`，42 文件；归档 SHA-256 `c7d7b50aea1c2bac1eaa0b7b6da6f519c0bdff7f78a0870d4cba176dba9938eb`。远端全部源码哈希通过；以 `--cpu-only --workers 16 --from semantic` 恢复，阶段回执 `results/p1_setup/20261024T110516Z-2246`。尚未运行模型训练。

## 无卡准备最终验证（2026-10-24 UTC）

- semantic 在 11:20:51 完成：14,116 文件。select 在 11:25:09 完成：镜面 4,613、窗户 2,095、干净 1,677 帧；canonical JSON SHA-256 与事先固定值完全一致。
- RGB/深度在 11:43:03 完成：16,770 文件，实际传输 2.861289291 GB。没有改变训练选择规则；过滤后实际训练集合仍为 4,418 帧 / 65 个 train 场景，角色 1,285 / 1,777 / 1,356。
- 云端 CPU suite 在 11:43:14 完成：**62 passed, 1 deselected in 9.18s**；排除项仍是上文说明的 P2 完整数值梯度检查。完整日志已取回本地 `results/autodl_p1_20261024/data_stages/run.log`。
- 离线数据预检退出 0：重新校验 30,886 个唯一文件、3,111,797,066 B 的实际大小和 SHA-256；没有缺失、额外成员或未完成 `.part`。同时再次验证部署 42 个文件及差异记录。
- 只解码 train 的 `ai_001_001/cam_00` frame 1 / 4 / 21，分别来自 mirror/window/clean；实际运行轴向 Z 转换、480×640 缩放、f1 提示合成与增强，检查 shape、有限值、有效掩码、中位数归一化和标签定义，均通过。它们是实际下载的 **Hypersim 渲染帧及模拟提示**，属于训练输入工程检查，不是真实传感器评测，也没有运行模型或训练。验证/test 文件仅做不解码的字节哈希检查。
- 数据预检总计 36.49 s，进程峰值 RSS 240,885,760 B（约 229.7 MiB），OOM/oom_kill 均为 0。回执、命令、日志和三列对齐图在本地 `results/autodl_p1_20261024/data_preflight/`，远端对应 `results/p1_data_preflight_v1/`。图已取回并目视检查；各模态使用同一次裁剪/翻转，镜面来源帧的镜面部分被随机裁剪移出大半，不另换更好看的样本。
- 数据清单 SHA-256：`bc8f498a187c3f002434b6cc000a1c84f67a732e6bba4a3a54277bfc87b6c083`；按文件名排序的实际内容清单 SHA-256：`714e56df24e599b6c250213cd38ebbd99d67e345db02777f3ba23af594a44ca0`。
- 下一步由用户加载 48 GB 级 GPU；资源门槛、估算依据和 CUDA 算子检查见 `docs/autodl_p1_gpu_preflight.md`。建议 32 GiB RAM，保守启动门槛为 16 GiB；显存能否容纳 batch 6 / 480×640 仍待实际 pilot。完整 F0/F1 训练尚未授权。
- 本地 F0/F1 检查点评测入口已提交 `4ffb047`，32 项 CPU 检查通过，真实 checkpoint 加载待 pilot。它没有上传到当前服务器训练部署，远端仍为 `8515378`。

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
# 已完成环境、模型和 plan 后，本次实际恢复命令：
bash scripts/linux/p1_setup.sh --cpu-only --workers 16 --from semantic
```

GPU 由用户加载且环境、数据全部验证后，检查设备再运行 **仅 pilot**：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/linux/p1_setup.sh --stage tests
bash scripts/linux/p1_run.sh pilot
```

完整训练开关 `TP_P1_ALLOW_FULL_TRAIN=1` 当前不设置。任何首次失败的训练/下载日志保留，恢复不覆盖旧实验。setup 日志与回执位于远端 `results/p1_setup/`，操作日志位于 `results/autodl_operator/`；本地归档和取回记录位于 `results/autodl_p1_20261024/`。
