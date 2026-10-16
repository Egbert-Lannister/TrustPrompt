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


def to_grid(prompt, prompt_valid, prior_aligned, rgb, label, hw=GRID, gt=None, gt_valid=None, focus=None):
    """v0 arrays; v1 adds gt (float16 m), gt_valid and focus (main-scope pixels: NYU mirror,
    HAMMER objects, plus simulated contamination regions) for error-based model selection."""
    p = resize_nearest(np.where(prompt_valid, prompt, 0).astype(np.float32), hw)
    v = resize_nearest(prompt_valid.astype(bool), hw)
    dg = resize_nearest(np.asarray(prior_aligned, np.float32), hw)
    out = dict(prompt=p.astype(np.float16), valid=v, dg=dg.astype(np.float16),
               rgb=resize_nearest(np.asarray(rgb, np.uint8), hw), label=resize_nearest(label, hw))
    if gt is not None:
        out.update(gt=resize_nearest(np.where(gt_valid, gt, 0).astype(np.float32), hw).astype(np.float16),
                   gt_valid=resize_nearest(np.asarray(gt_valid, bool), hw),
                   focus=resize_nearest(np.asarray(focus, bool), hw))
    return out


def replacement_error(trust, threshold, a, post=False):
    """Selection objective of T v1 on one low-res item `a` (needs gt/gt_valid/focus/dg):
    mean |D - GT| / GT where D keeps the prompt on trusted pixels and takes D_g on flagged
    ones (the k0 fill), over valid prompt & valid GT pixels; returned for the focus scope
    (None if empty) and the full image."""
    from tp.trust.post import postprocess
    p = a["prompt"].astype(np.float32); dg = a["dg"].astype(np.float32); gt = a["gt"].astype(np.float32)
    ok = a["valid"] & a["gt_valid"] & (p > 0) & (gt > 0) & np.isfinite(dg) & (dg > 0)
    flag = a["valid"] & (trust < threshold)
    if post:
        flag = postprocess(flag, a["valid"])
    d = np.where(flag, dg, p)
    err = np.abs(d - gt) / np.where(gt > 0, gt, 1)
    sc = ok & a["focus"]
    return (float(err[sc].mean()) if sc.any() else None), (float(err[ok].mean()) if ok.any() else None)


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


def labels3_from_arrays(a, tau=0.05):
    """T v2 labels from v1 arrays: 0 clean, 1 measured too far (P > (1+tau) GT; true surface is in front,
    so P is an upper bound), 2 measured too near (P < (1-tau) GT), 255 ignore."""
    p = a["prompt"].astype(np.float32); gt = a["gt"].astype(np.float32)
    ok = a["valid"] & a["gt_valid"] & (p > 0) & (gt > 0)
    lab = np.full(p.shape, 255, np.uint8)
    lab[ok & (np.abs(p - gt) <= tau * gt)] = 0
    lab[ok & (p > (1 + tau) * gt)] = 1
    lab[ok & (p < (1 - tau) * gt)] = 2
    return lab
