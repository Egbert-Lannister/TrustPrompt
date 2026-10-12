"""DEPTHOR++-style learnable-parameter-free prompt anomaly detector (baseline, revision R14).

Reimplemented from the paper text (Xiang et al., "DEPTHOR++: Robust Depth
Enhancement from a Real-World Lightweight dToF and RGB Guidance",
arXiv:2509.26498v1, Sec. on anomaly detection, read 2026-10-12). No code was
released, so this is our reading of the equations:

  G_i = tanh(|G_i^abs - G_i^rel| / delta),  G^x_i = mean_j sgn(x_i - x_j)
  S_i = mean_j exp(-alpha ||p_i - p_j||) * | |d_i-d_j|/(d_i+d_j+eps) - |r_i-r_j|/(r_i+r_j+eps) |
  A_i = S_i + G_i
  gamma = Spearman(d, r);  t = +inf if gamma > 0.95,
          w t_stat + (1-w) t_otsu if 0.85 < gamma <= 0.95 (w = sigmoid(k (gamma - u))),
          t_otsu otherwise;  t_stat = TopK(A, floor(p N));  anomaly = A > t
  delta=0.5, eps=1e-6, alpha=15, p=0.04, k=40, u=0.9 (paper values).

Deviations (recorded in every run): the paper works on sparse dToF points with
all N^2 pairs. Our prompts are dense, so (1) points are taken on a strided grid
with at most `max_points` valid samples, (2) the means over j use a random
reference subset of `n_ref` grid points (Monte-Carlo estimate of the same mean),
(3) per-pixel flags/scores come from the nearest valid grid sample. The paper
uses Depth Anything V2-Small relative depth; we use the cached UniDepthV2-S depth
as r (its ordering matches depth, which the sign terms require; the paper does
not say whether its r is inverted). p_i = (x / W, y / H). Otsu uses 256 bins.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy import ndimage, stats

PAPER = dict(delta=0.5, eps=1e-6, alpha=15.0, p=0.04, k=40.0, u=0.9, hi=0.95, lo=0.85)


def otsu_threshold(x, bins=256):
    x = np.asarray(x, np.float64)
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers); m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[1:][int(np.argmax(between))])


def threshold(scores, gamma, p=PAPER["p"], k=PAPER["k"], u=PAPER["u"], hi=PAPER["hi"], lo=PAPER["lo"]):
    if gamma > hi:
        return np.inf, "gamma_gt_hi_no_anomaly"
    t_otsu = otsu_threshold(scores)
    if gamma <= lo:
        return t_otsu, "otsu"
    n = max(1, int(np.floor(p * scores.size)))
    t_stat = float(np.sort(scores)[::-1][n - 1])
    w = 1.0 / (1.0 + np.exp(-k * (gamma - u)))
    return w * t_stat + (1 - w) * t_otsu, "blend"


def point_scores(d, r, px, py, ref, delta=PAPER["delta"], alpha=PAPER["alpha"], eps=PAPER["eps"], chunk=1024):
    """A_i = S_i + G_i for points (d, r, px, py) against reference indices `ref`."""
    def rank(x):
        srt = np.sort(x[ref])
        less = np.searchsorted(srt, x, "left"); greater = ref.size - np.searchsorted(srt, x, "right")
        return (less - greater) / ref.size
    g = np.tanh(np.abs(rank(d) - rank(r)) / delta)
    dj, rj, xj, yj = (a[ref].astype(np.float32)[None, :] for a in (d, r, px, py))
    s = np.empty(d.size, np.float64)
    for a in range(0, d.size, chunk):
        b = min(d.size, a + chunk)
        di, ri, xi, yi = (z[a:b].astype(np.float32)[:, None] for z in (d, r, px, py))
        vabs = np.abs(di - dj) / (di + dj + eps)
        vrel = np.abs(ri - rj) / (ri + rj + eps)
        w = np.exp(-alpha * np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
        s[a:b] = (w * np.abs(vabs - vrel)).mean(1)
    return s + g, s, g


def sample_seed(base_seed, key):
    return (int(base_seed) * 1_000_003 + zlib.crc32(key.encode("utf-8"))) % (2 ** 32)


def detect(prompt, prompt_valid, rel_depth, *, seed=0, max_points=30000, n_ref=4096, min_points=100, **paper):
    """Return (anomaly bool HxW, score float HxW NaN outside valid, info dict)."""
    prompt = np.asarray(prompt, np.float64); rel = np.asarray(rel_depth, np.float64)
    h, w = prompt.shape
    valid = np.asarray(prompt_valid, bool) & np.isfinite(prompt) & (prompt > 0) & np.isfinite(rel) & (rel > 0)
    n_valid = int(valid.sum())
    stride = max(1, int(np.ceil(np.sqrt(n_valid / max_points)))) if n_valid else 1
    gy = np.arange(stride // 2, h, stride); gx = np.arange(stride // 2, w, stride)
    gv = valid[np.ix_(gy, gx)]
    info = dict(detector="depthor_pp_v1_reimpl", stride=stride, n_ref=n_ref, paper_params={**PAPER, **paper},
                valid_pixels=n_valid, grid_points=int(gv.sum()))
    anomaly = np.zeros((h, w), bool); score = np.full((h, w), np.nan, np.float32)
    if gv.sum() < min_points:
        info.update(status="too_few_points", threshold=None, gamma=None, anomaly_pixels=0)
        return anomaly, score, info
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    ys, xs = yy[gv], xx[gv]
    d = prompt[ys, xs]; r = rel[ys, xs]; px = xs / w; py = ys / h
    rng = np.random.default_rng(seed)
    ref = np.arange(d.size) if d.size <= n_ref else np.sort(rng.choice(d.size, n_ref, replace=False))
    a, s_term, g_term = point_scores(d, r, px, py, ref, **{k: v for k, v in paper.items() if k in ("delta", "alpha", "eps")})
    sub = np.arange(d.size) if d.size <= 20000 else rng.choice(d.size, 20000, replace=False)
    gamma = float(stats.spearmanr(d[sub], r[sub]).statistic)
    t, mode = threshold(a, gamma, **{k: v for k, v in paper.items() if k in ("p", "k", "u", "hi", "lo")})
    grid_a = np.full(gv.shape, np.nan); grid_a[gv] = a
    # nearest valid grid sample for every pixel
    _, (iy, ix) = ndimage.distance_transform_edt(~gv, return_indices=True)
    cy = np.clip(np.arange(h) // stride, 0, gv.shape[0] - 1); cx = np.clip(np.arange(w) // stride, 0, gv.shape[1] - 1)
    full = grid_a[iy[np.ix_(cy, cx)], ix[np.ix_(cy, cx)]]
    score[valid] = full[valid].astype(np.float32)
    anomaly = valid & (full > t)
    info.update(status="ok", gamma=gamma, threshold=None if not np.isfinite(t) else float(t), threshold_mode=mode,
                grid_anomaly_fraction=float((a > t).mean()), anomaly_pixels=int(anomaly.sum()),
                mean_s=float(s_term.mean()), mean_g=float(g_term.mean()))
    return anomaly, score, info


def trust_from_score(score):
    """Monotone map to [0,1] for trust-quality metrics (higher = more trusted)."""
    return np.where(np.isfinite(score), 1.0 / (1.0 + np.maximum(score, 0)), np.nan)


def detect_for_sample(sample, prior_depth, cfg=None):
    """Shared entry point for evaluation and PromptDA prompt building, so both see the
    identical mask: the prompt is taken on the GT grid (nearest), and the per-sample
    seed depends only on cfg['seed'] and the sample key, not on the consumer."""
    from tp.geom import resize_nearest
    cfg = dict(cfg or {})
    base_seed = cfg.pop("seed", 0)
    key = f"{sample.dataset}__{sample.sensor}__{sample.scene}__{sample.frame}"
    p = resize_nearest(sample.prompt, sample.gt.shape)
    pv = resize_nearest(sample.prompt_valid, sample.gt.shape)
    an, score, info = detect(p, pv, prior_depth, seed=sample_seed(base_seed, key), **cfg)
    info["sample_seed_base"] = base_seed
    return an, score, info
