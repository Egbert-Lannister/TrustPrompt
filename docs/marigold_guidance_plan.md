# Marigold-DC 引导线：把 T 的标记写进引导目标（v1，2026-10-18）

状态：**代码已实现，小规模验证通过（CPU 上的小型随机管线），尚未在真实权重上运行。** 判据都在任何 Marigold-DC 预测之前写定，见 `configs/baselines/marigold_guidance_v1.yaml`。不需要训练，在本机运行。

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

**M0：重写的循环是否忠实，以及单张图的代价**（`scripts/check_marigold_guidance.py`）

- 用 3 张 NYU val 图比较：上游管线、上游再跑一次（噪声基线）、重写循环（等式 guide），再测一次 tonesided 的耗时和显存。
- 判定“忠实”的条件（和 LingBot 的检查一样）：
  - 逐图 |ΔC AbsRel| 的最大值 < 1e-3；
  - 合并后的像素相对差 p99 < 0.5%。
- 不忠实就不跑引导变体。显存不够（8 GB）时停下来报告，不自行降低分辨率。

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
powershell -ExecutionPolicy Bypass -File scripts\windows\step12_marigold_guidance.ps1            # M0 + M1
powershell -ExecutionPolicy Bypass -File scripts\windows\step12_marigold_guidance.ps1 -Hammer    # M2，先看 M1 结果
```

- 首次运行会下载 Marigold-DC 权重：约 5.2 GB，Apache-2.0，固定 revision，并做 SHA256 校验。
- 总耗时由 M0 打印的单张耗时决定：M1 共 30 × 6 次推理，外加 6 张预检图。
