"""T v0 training/inference arrays on the PromptDA prompt grid (192x256).

Label (per prompt pixel, method-independent, same rule as the evaluation regions):
  0 = clean        valid prompt, valid GT, |P - GT| / GT <= tau
  1 = contaminated valid prompt, valid GT, |P - GT| / GT >  tau
  255 = ignore     invalid prompt or invalid GT
Every array is sampled with tp.geom.resize_nearest (centre grid), so the low-res
prompt, label, prior and RGB refer to the same full-resolution pixels.
Simulated variants (train split only) replace the prompt by tp.sim.generate.simulate
output and recompute D_g on that prompt with the same robust global alignment.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from tp.geom import resize_nearest

GRID = (192, 256)
FEAT_GRID = (24, 32)


def labels_from(prompt, prompt_valid, gt, gt_valid, tau=0.05):
    lab = np.full(gt.shape, 255, np.uint8)
    ok = prompt_valid & gt_valid & np.isfinite(prompt) & (prompt > 0) & np.isfinite(gt) & (gt > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = np.abs(prompt - gt) / gt > tau
    lab[ok & ~bad] = 0
    lab[ok & bad] = 1
    return lab


def to_grid(prompt, prompt_valid, prior_aligned, rgb, label, hw=GRID):
    p = resize_nearest(np.where(prompt_valid, prompt, 0).astype(np.float32), hw)
    v = resize_nearest(prompt_valid.astype(bool), hw)
    dg = resize_nearest(np.asarray(prior_aligned, np.float32), hw)
    return dict(prompt=p.astype(np.float16), valid=v, dg=dg.astype(np.float16),
                rgb=resize_nearest(np.asarray(rgb, np.uint8), hw), label=resize_nearest(label, hw))


def upsample_prob(prob_lr, shape):
    """Bilinear, centre-aligned upsampling of a low-res map to `shape` (same grid convention
    as resize_nearest: low-res cell i covers full-res [i*s, (i+1)*s))."""
    prob_lr = np.asarray(prob_lr, np.float32)
    h, w = shape; lh, lw = prob_lr.shape
    yy = (np.arange(h) + 0.5) * lh / h - 0.5
    xx = (np.arange(w) + 0.5) * lw / w - 0.5
    gy, gx = np.meshgrid(np.clip(yy, 0, lh - 1), np.clip(xx, 0, lw - 1), indexing="ij")
    return map_coordinates(prob_lr, [gy, gx], order=1, mode="nearest").astype(np.float32)


def sim_labels(type_labels):
    """tp.sim type labels (0 clean / 1 far / 2 near / 3 hole / 255 ignore) -> T labels."""
    out = np.full(type_labels.shape, 255, np.uint8)
    out[type_labels == 0] = 0
    out[(type_labels == 1) | (type_labels == 2)] = 1
    return out


def scene_inner_holdout(dataset, scene):
    """Train-split scenes held out for T model selection (val/test are never used for it).
    HAMMER: scene 10. NYU: about one in five scenes, by a fixed hash of the scene name."""
    import zlib
    if dataset == "hammer":
        return scene in ("scene10", "10")
    return zlib.crc32(scene.encode("utf-8")) % 5 == 0
