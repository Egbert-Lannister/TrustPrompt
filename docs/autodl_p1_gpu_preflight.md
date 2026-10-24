# AutoDL P1：挂载 GPU 前后的只读预检

本页依据当前 `scripts/linux/p1_run.sh`、`scripts/train_omnidc.py`、`tp/omnidc/data.py` 和固定配置核查。命令不下载数据、不构造完整 OMNI-DC、不改配置、不启动 pilot。无卡阶段可以完成下面的三帧读取检查；GPU 检查必须等用户挂载 GPU 后执行。

**实际 GPU 适配更新（2026-10-24）：** 用户加载的设备报告 RTX 6000D / sm_120，cu126 最小 CUDA 张量检查失败。现已建立 `/root/trustprompt-envs/p1-cu128`，torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / cuDNN 91900；sm_120 编译支持、CUDA deform_conv2d 前后向、pip check 及 62 项 CPU 检查通过。当前 pilot 必须使用这个新环境。下方原始 cu126 命令保留为准备过程记录，不能直接用于本次设备；已执行的 cu128 检查脚本及回执在 `results/autodl_p1_20261024/cu128/`。后续恢复同一训练应保持 cu128 环境及远端源码树不变。

## 无卡阶段原始预检约定（历史记录）

- 使用同一个 `TP_P1_VENV=/root/trustprompt-envs/p1-cu126`；`p1_run.sh` 会激活它，未设置时才回退 `.venv-p1`。不能误用实例原有的 torch 2.8 / cu128 基础环境。目标版本是 torch 2.11.0、torchvision 0.26.0、CUDA wheel cu126、opencv-python-headless 4.11.0.86。
- `configs/omnidc/p1.yaml` 固定 batch 6、8 个数据加载进程；正式 pilot 不传 `--smoke`，训练裁剪为 480×640。`--smoke` 会改变为 batch 1、256×320，不能用它的吞吐量代表 P1。
- 固定训练清单实际为 **4,418 帧、65 个场景**，全为 Hypersim 官方 train：mirror 1,285、window 1,777、clean 1,356；mirror 的面积比例至少 0.05。旧计划中“66 个场景”与当前筛选结果不一致。该计数来自本地已固定 `frames_v1.json` 的真实筛选，不是远端下载完成的证明。
- 无卡实例当前仅 0.5 CPU / 2 GB RAM，适合下载及有限帧读取，**不足以启动当前完整模型与 8-worker pilot**。挂卡后必须读实际 cgroup 限额，不能只看 `nproc` 或主机 `free` 输出。有效 CPU ≥8 是与 8 workers 匹配的**吞吐建议**，不是数学上的运行必要条件；CPU 较少可能能跑，但 GPU 会等待数据。
- RAM 容量 **至少 16 GiB** 是未实测远端峰值前采用的保守启动门槛，建议 32 GiB；它不是经过远端峰值测量得到的精确最低值。内存还需留给权重初始化、8 个合成进程、预取和 pinned-memory 队列。2 GB 阶段不得尝试完整模型加载。
- 计划使用 48 GB 级 GPU。小张量 CUDA 检查只能验证驱动、架构和算子可用；batch 6 / 480×640 是否适配，仍须由授权 pilot 的实际峰值确认。既有笔记本 smoke 是小裁剪、小 batch，不能推出正式配置的显存峰值。
- 当前无卡实例 `/dev/shm` 已由主任务核对为 **2.0 GB 空闲**，并非 64 MB。挂卡后仍要重查。当前数据加载设置建议至少 2 GiB 共享内存容量，启动时至少约 1.5 GiB 空闲；仅 64 MB 的容器会有明确的共享内存耗尽风险。
- `results/` 所在磁盘应保留 **至少 5 GB（十进制）空闲**。系统盘环境与数据盘应分开，避免把 CUDA wheel 环境也放入 50 GB 数据盘挤占检查点空间。

当前代码会在训练前校验 OMNI-DC upstream commit、工作树状态和两个权重文件的 size/SHA256。它不会替代训练帧完整性检查：`load_frame` 只在取到该帧时读取数据，因此先查全部训练文件的 manifest/size，再读取 3 帧，可以将明显缺失提前暴露出来。

## 内存与检查点估算依据

单个 480×640 样本的主要返回数组为 RGB float32×3、dep float32、GT float32、labels int64、valid bool，总计每像素 29 字节。本地实际 `make_sample` 返回 NumPy 数组为 **8,908,804 字节**；dataset 再加 K、pattern、index 等小张量后约 8.91 MB。batch 6 约 53.45 MB。

当前 DataLoader 使用 `num_workers=8`、`persistent_workers=True`、`pin_memory=True`，未覆写 `prefetch_factor`，PyTorch 多进程默认每 worker 预取 2 个 batch。仅 16 个预取 batch 的张量载荷就约 **855 MB（0.797 GiB）**；尚未包括 worker 内的原分辨率 HDF5、float64 射线、镜像渲染中间数组、主进程当前 batch 及 pinned-memory 复制。这解释了 `/dev/shm` 与 RAM 的独立要求；它不是总 RSS 实测值。

本地仅读取 `model.safetensors` 的 JSON header，排除 `depth_module.` 后，其余 state tensor 总计 **341,686,032 字节**。f1 无额外 gate，且冻结的 DAv2 不写入训练检查点；按 state 加 AdamW 两个动量粗估，`last.pt` 约 **1.025 GB / 0.955 GiB**。少量冻结参数、buffer、序列化开销会使实际值略有差异。覆盖时先写 `last.tmp` 再原子替换，旧、新文件共存的瞬时占用约 2.05 GB。脚本内部还根据真实 payload 大小检查临时文件所需空闲空间；建议的 5 GB 余量覆盖这一过程及日志。pilot 正常结束只保留 `last.pt`，不产生完整训练的 `final.pt`。

## 无 GPU：全部训练文件存在性检查及最多 3 帧读取

在远端 checkout 中执行下面命令，先保持无卡。它不导入 torch、不启动 DataLoader、不加载权重；先检查 4,418 帧对应 13,254 个文件的 manifest 与大小，只对 mirror/window/clean 各一帧的 9 个文件读取并复核 SHA256，然后走真实的 Z 深度转换、480×640 缩放和 f1 合成。任何缺失、哈希或合成错误直接失败，不自动换帧。

不要用 `HypersimTraining.__getitem__` 代替本检查：它在无有效裁剪时最多重试 10 帧，会突破这里最多 3 帧的限制。三帧成功也不能取代下载器对全部文件的 CRC/SHA256 校验。

```bash
cd /root/autodl-tmp/TrustPrompt
export TP_P1_VENV=/root/trustprompt-envs/p1-cu126
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$TP_P1_VENV/bin/python" -B - <<'PY'
import hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from tp.data.hypersim import load_frame
from tp.omnidc.data import training_frames, resize_frame, make_sample

root = Path.cwd()
cfg = yaml.safe_load((root / "configs/omnidc/p1.yaml").read_text())
d = cfg["data"]
data = root / d["root"]
selection = json.loads((root / d["selection"]).read_text())
min_mirror = float(yaml.safe_load((root / d["training"]).read_text())["min_mirror_share"])
frames = training_frames(data, min_mirror, tuple(d["splits"]))
assert len(frames) == 4418 and len({r["scene"] for r in frames}) == 65
assert Counter(r["role"] for r in frames) == {"mirror": 1285, "window": 1777, "clean": 1356}
assert len({(r["scene"], r["camera"], r["frame"]) for r in frames}) == len(frames)
assert all(r["split"] == selection["scenes"][r["scene"]]["split"] == "train" for r in frames)
def names(r):
    base = f'{r["scene"]}/images/scene_{r["camera"]}'
    stem = f'frame.{r["frame"]:04d}'
    return [f'{base}_geometry_hdf5/{stem}.semantic.hdf5',
            f'{base}_geometry_hdf5/{stem}.depth_meters.hdf5',
            f'{base}_final_preview/{stem}.tonemap.jpg']
needed = {name for r in frames for name in names(r)}
records = {}
with (data / "manifest.jsonl").open() as fh:
    for line in fh:
        row = json.loads(line)
        if row["name"] in needed:
            records[row["name"]] = row
bad = [name for name in needed if name not in records or not (data / name).is_file()
       or (data / name).stat().st_size != records[name]["size"]]
assert not bad, f"Missing/mismatched training files: {len(bad)}; first={bad[:5]}"
print("TRAIN FILE SIZES OK:", len(needed), "files; 4418 frames / 65 train scenes")
chosen = [next(r for r in frames if r["role"] == role) for role in ("mirror", "window", "clean")]
for i, r in enumerate(chosen):
    for name in names(r):
        h = hashlib.sha256()
        with (data / name).open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == records[name]["sha256"], name
    frame = resize_frame(load_frame(data, r["scene"], r["camera"], r["frame"],
                         selection["scenes"][r["scene"]]["camera"]["M_cam_from_uv"]))
    sample = make_sample(frame, "f1", np.random.default_rng([cfg["seed"], i, 0]), d["p_sparse"])
    assert sample["rgb"].shape == (3, 480, 640) and sample["gt"].shape == (1, 480, 640)
    assert all(np.isfinite(v).all() for v in sample.values())
    assert np.any(sample["dep"] > 0) and np.any(sample["gt"] > 0)
    print("FRAME OK", r["role"], r["scene"], r["camera"], r["frame"],
          "numpy_bytes=", sum(v.nbytes for v in sample.values()))
print("CPU READ SMOKE PASSED: exactly 3 frames; no GPU or model load")
PY
```

这三个固定首帧当前为 `ai_001_001/cam_00` 的 frame 1（mirror）、4（window）、21（clean）。本地只读验证已通过它们的真实读取、缩放和 f1 合成，每帧返回 8,908,804 字节；未据此宣称远端文件已完整。远端命令还会执行上面的 manifest/size 与三帧 SHA256 检查。

## 挂载 GPU 后：环境、资源与关键 CUDA 算子检查

先等待无卡阶段的数据准备进程正常完成，确认所有下载和测试成功，再启动 1 小时 pilot。下面仅分配极小张量，检查 DySPN 实际依赖的 `torchvision.ops.deform_conv2d` 前向和反向，不加载 3 GB 模型或运行训练。CPU 核数低于 8 仅输出警告；RAM、共享内存、结果盘和计划 GPU 级别不足则停止。

```bash
cd /root/autodl-tmp/TrustPrompt
export TP_P1_VENV=/root/trustprompt-envs/p1-cu126
"$TP_P1_VENV/bin/python" -B - <<'PY'
import importlib.metadata as md
import json, os, shutil, sys
from pathlib import Path
import torch, torchvision, cv2
from torchvision.ops import deform_conv2d

GiB = 2 ** 30
assert Path(sys.prefix).resolve() == Path(os.environ["TP_P1_VENV"]).resolve()
assert torch.__version__.split("+")[0] == "2.11.0"
assert torchvision.__version__.split("+")[0] == "0.26.0"
assert torch.version.cuda == "12.6", "Expected the prepared cu126 environment"
assert md.version("opencv-python-headless") == "4.11.0.86"
def cg(name):
    p = Path("/sys/fs/cgroup") / name
    return p.read_text().strip() if p.is_file() else None
meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024
           for line in Path("/proc/meminfo").read_text().splitlines()}
ram_limit = cg("memory.max")
assert ram_limit is not None, "Inspect actual cgroup memory controller before proceeding"
ram = min(meminfo["MemTotal"], int(ram_limit)) if ram_limit != "max" else meminfo["MemTotal"]
assert ram >= 16 * GiB, f"Conservative RAM gate failed: {ram / GiB:.2f} GiB"
cpu = float(len(os.sched_getaffinity(0)))
cpu_max = cg("cpu.max")
assert cpu_max is not None, "Inspect actual cgroup CPU controller before proceeding"
quota, period = cpu_max.split()
if quota != "max":
    cpu = min(cpu, int(quota) / int(period))
if cpu < 8:
    print("WARNING: effective CPU", cpu, "< 8; loader throughput may starve the GPU")
shm = shutil.disk_usage("/dev/shm").free
free_disk = shutil.disk_usage("results").free
assert shm >= 1.5 * GiB, f"Insufficient free /dev/shm: {shm / GiB:.2f} GiB; recommend 2 GiB capacity"
assert free_disk >= 5e9, f"Need >=5 GB free on results disk, got {free_disk / 1e9:.2f} GB"
assert torch.cuda.is_available(), "CUDA unavailable; check GPU attachment and CUDA_VISIBLE_DEVICES"
assert torch.cuda.device_count() == 1, "Select the single allocated pilot GPU explicitly"
free_gpu, total_gpu = torch.cuda.mem_get_info(0)
assert total_gpu >= 44 * GiB and free_gpu >= 40 * GiB, "Expected a mostly free 48 GB class GPU"
x = torch.ones(1, 1, 8, 8, device="cuda", requires_grad=True)
w = torch.ones(1, 1, 3, 3, device="cuda", requires_grad=True)
offset = torch.zeros(1, 18, 8, 8, device="cuda", requires_grad=True)
y = deform_conv2d(x, offset, w, padding=(1, 1))
y.square().mean().backward()
torch.cuda.synchronize()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, w, offset))
print(json.dumps(dict(venv=sys.prefix, torch=torch.__version__, torchvision=torchvision.__version__,
                     cv2=cv2.__version__, gpu=torch.cuda.get_device_name(0),
                     gpu_total_gib=total_gpu / GiB, gpu_free_gib=free_gpu / GiB,
                     effective_cpu=cpu, ram_capacity_gib=ram / GiB,
                     ram_current=cg("memory.current"), shm_free_gib=shm / GiB,
                     results_free_gb=free_disk / 1e9, cuda_deformconv_forward_backward="passed"), indent=2))
PY
```

这里 44 GiB 总显存/40 GiB 空闲只是“48 GB 级、基本空闲”的资源筛查，不是 batch 6 显存充分性的数学证明。若 cgroup 控制器布局发生变化，命令会要求人工核对，不能用主机总 RAM 代替容器限额。若 CPU 阶段曾在交互 shell 中将 `CUDA_VISIBLE_DEVICES` 设为空，需确认 GPU 分配后在正确 shell 中恢复；不要因此切换到基础 Python。

## 正式启动仍由主任务执行

预检通过后，已授权的一小时入口是 `TP_P1_VENV=... bash scripts/linux/p1_run.sh pilot`；本页命令不调用它。当前 wrapper 使用 GNU `timeout` 的 60 分钟墙钟限额，必要时额外最多 30 秒 kill 宽限；trainer 在内部给最后保存预留最多 60 秒。它不等于平台计费自动停止，也不等于 GPU kernel 用时。

正常退出应有 `receipt.json` 的 `pilot_complete`、非零 steps/samples、可读 `checkpoints/last.pt`、实际 `training_frames.json`（4,418/65）及日志峰值。`pilot_no_training`、OOM、下载/读盘失败或被 watchdog 杀掉都不是 pilot 成功。首轮真实 batch 6 出现 OOM 时保留失败记录，再审查资源或配置；不要默默降 batch 或裁剪尺寸后把结果继续称为原配置。
