# Marigold-DC 引导线：把 T 的标记写进引导目标（v1，2026-10-18）

状态：**M0、M1 已完成（小规模验证，开发集诊断）。按事先写定的门槛，没有变体通过，M2 不运行。** 结果见 `docs/marigold_guidance_nyu.md`，判读见文末 §5。

## 1. 做什么

Marigold-DC（ICCV 2025，prs-eth/Marigold-DC @ c6d756b）在扩散去噪的每一步：

1. 把 Tweedie 预估的深度解码出来；
2. 用 mean|D − P| + mean (D − P)² 让它贴合所有有效提示像素；
3. 把梯度缩放后更新潜变量。

所有提示像素一视同仁，所以镜面上的错误读数也会被拟合进去。

`baselines/guidance.py` 逐步重写了这个循环，只改了两处：

- **目标：** L = Σ [ w·ρ(e) + h·ρ(max(e, 0)) ] / n，其中 ρ(e) = |e| + e²，e = D − P。
  - w 是等式权重；
  - h 是单边权重：预测可以在读数前面，但不能在读数后面。
- **锚点：** 用来设定仿射范围的像素（上游用全部有效像素）。这样远处的反射读数不会把初始深度范围撑大。

不传 guide 时，`Guide.equality()` 按上游原式计算损失。在 CPU 上的小型随机管线中，它和上游管线的输出逐位相同（`tests/test_guidance.py`）。

## 2. 变体（`scripts/cache_completion.py`）

| 变体 | 送入模型的深度 | 引导目标 | 读取 GT |
|---|---|---|---|
| raw | 全部提示 | 上游 | 否 |
| tmask | 删除 T < λ 的像素 | 上游 | 否 |
| tonesided | 全部提示 | 未标记像素：等式；被 T 标记的像素：单边 | 否 |
| tsoft | 全部提示 | 等式权重 T，单边权重 1 − T | 否 |
| oracle_mask | 删除 C | 上游 | 是 [oracle] |
| oracle_onesided | 全部提示 | 未标记：等式；C：单边 | 是 [oracle] |

- raw、tmask、oracle_mask 走上游管线本身；三个引导变体走重写的循环。
- λ 的来源：NYU 用 T v1 和 conformal α = 0.1 的阈值；HAMMER 用交叉拟合的 T，每张图用各自所在折的阈值。

**事前诊断**（`docs/onesided_validity.md`）：被 T 标记的像素里，真实深度 ≤ 读数 × (1 + τ) 的比例为：

| 数据集 | 比例 |
|---|---|
| NYU，主范围 | 99.4% |
| HAMMER，主范围 | 99.0% |

读数偏近的只占 0.6–1.0%。所以单边约束在被标记的像素上几乎不会和真值冲突；即使是误报的干净像素，读数也接近真值，单边约束仍然给出一个正确的上界。

## 3. 阶段与判据（事先写定）

**M0：重写的循环是否忠实，以及单张图的代价**（`scripts/check_marigold_guidance.py`，当前为 v2）

- **v1 的结果（2026-10-18，`docs/marigold_guidance_check_v1.md`）：判定为“不忠实”，但这个结论不说明代码有问题。**
  - 规则要求“和上游相比像素相对差 p99 < 0.5%”。可上游自己跑两次，p99 就有 0.9–1.6%（bf16 下 CUDA 核不确定，经过 50 步引导被放大），所以这条规则在生产设置下本身就过不了。
  - 重写循环与上游的差（p99 0.7–3.1%）和上游自身的重跑差是同一量级。
  - 同时暴露出显存问题：峰值 10.6 GB，超过 8 GB 显卡，Windows 把显存换页到共享内存，每张图约 17.7 分钟。
- **v2 的规则（看到 v1 结果后写定，见 decisions §20）：**
  1. **精确性**：开确定性 CUDA，用生产的模型和精度，在小设置下（384 px、10 步）跑 3 张 NYU val 图：上游、上游重跑、重写循环。
     - 若上游两次逐位相同，要求重写循环和上游也逐位相同；
     - 若确定性没有达到，要求重写循环与上游之差的 p99 ≤ 2 × 上游重跑差的 p99。
  2. **代价**：生产设置下跑 1 张图（上游和 tonesided 各一次），峰值显存要低于显卡的 95%，否则不跑 M1。
  - 两条都通过才进入 M1，同时打印 M1 的耗时估计。
- **显存处理**（不改变数值）：UNet 和 VAE 打开梯度检查点。diffusers 只在 train 模式下做检查点，代码会先确认模型里没有 dropout 和 batch norm。空提示的文本嵌入只算一次，之后释放文本编码器。CPU 上的小型管线中，这样做前后的输出逐位相同（`tests/test_guidance.py`）。
- 如果显存仍然不够，或者 M1 估计耗时太长，停下来报告，不自行降低分辨率或步数。

**M1：NYU 全部 30 张开发图 × 6 个变体**

- Holm 检验族（`configs/baselines/holm_marigold_nyu_v1.yaml`）：
  - 主族为 val，次族为 train 留出场景；
  - 比较 tmask 对 raw、tonesided 对 tmask、tsoft 对 tmask。
- **进入 M2 的门槛**：对每个引导变体 g，下面三条都要满足：
  1. val 上 C AbsRel(g) ≤ 0.9 × C AbsRel(tmask)；
  2. train 留出场景上 C AbsRel(g) ≤ C AbsRel(tmask)；
  3. val 上 g 对 tmask 的干净区域检查不是“明显变差”。

  val 只有 19 张，所以 M1 不要求显著性，显著性留到 M2 检验。
- **GT 标记上限**（只做描述）：1 − C(oracle_onesided) / C(oracle_mask)，在 val 上计算。
- **结论怎么读：**

  | 情况 | 结论 |
  |---|---|
  | 有变体过门槛 | 进入 M2 |
  | 没有变体过门槛，但上限 ≥ 10% | 提升空间存在，受限于 T 的标记 |
  | 都不满足 | 单边引导在这里没有额外收益；Marigold-DC 只作为第 7 个屏蔽基线；主线的模块 B 降为消融 |

**M2：HAMMER，10 个开发场景 × 3 个传感器 × 每组 3 帧（90 张）**

- 只跑过了门槛的变体，外加 raw、tmask、oracle_mask。T 用交叉拟合版本。
- 检验族（`configs/baselines/holm_marigold_cf_v1.yaml`）：tmask 对 raw，以及每个引导变体对 tmask。没跑的变体按 p = 1 留在族里。
- **确认的条件**：g 对 tmask 经 Holm 校正后 p < 0.05，差值 < 0，并且干净区域不是“明显变差”。

## 4. 运行

```
powershell -ExecutionPolicy Bypass -File scripts\windows\step12_marigold_guidance.ps1 -CheckOnly  # 只跑 M0
powershell -ExecutionPolicy Bypass -File scripts\windows\step12_marigold_guidance.ps1            # M0 + M1
powershell -ExecutionPolicy Bypass -File scripts\windows\step12_marigold_guidance.ps1 -Hammer    # M2，先看 M1 结果
```

- 首次运行会下载 Marigold-DC 权重：约 5.2 GB，Apache-2.0，固定 revision，并做 SHA256 校验。
- 总耗时由 M0 打印的单张耗时决定：M1 共 30 × 6 次推理，外加 6 张预检图。

## 5. M1 结果与判读（2026-10-19）

**按事先写定的规则：** 没有引导变体通过门槛。tonesided 在 val 上只比 T 屏蔽好 2.2%（门槛是 10%）；tsoft 在 val 和留出场景上都更差。GT 标记下的上限是 −1.9%，即用真值标记时单边约束也不比删除好。结论：在主范围 C AbsRel 上，单边引导没有额外收益，M2 不运行。

**同时得到的、预注册范围内的结果：**

- T 屏蔽对 Marigold-DC 同样有效：
  - val：0.579 → 0.157，Holm p = 0.003；
  - 留出场景：1.088 → 0.046，p = 0.0003。
- 干净区域都不是“明显变差”。
- Marigold-DC 的原始继承率 IR 为 0.62（val）。它比 OMNI-DC（0.93）少照抄一些，但仍有大量照抄。

**事后观察（不在门槛内，只能作为假设）：**

- tonesided 对比 T 屏蔽，干净区域的 AbsRel 在 val 上下降 16%（0.086 → 0.072，95% CI [−0.039, +0.0005]），在留出场景上下降 27%（CI 上界 +0.002）。
- 留出场景上 tonesided 的 C 也略好（−0.009，Holm p = 0.044）。
- 也就是说，把标记像素当上界而不是删掉，主要减轻的是屏蔽对周围干净像素的伤害，而不是改善错误像素本身。这与 GT 标记下上限约为 0 的结果一致：真值标记的像素几乎都是读远了，删掉后先验给出的填充本来就落在读数前面，单边约束很少被触发（`tests/test_onesided.py` 的观察相同）。
- 如果以后要检验“减轻过度屏蔽”，需要单独写一个以干净区域为主指标的预注册检验，不能沿用这次的数字作为证据。
