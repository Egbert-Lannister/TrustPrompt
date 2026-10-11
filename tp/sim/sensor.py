"""Base sensor models (plan section 4.1), v0. Applied to first-surface GT to make P_clean.

Not physically exact by design: real sensor data are the evaluation sets; these
models only widen the training distribution. All parameters are explicit and
recorded; defaults are [estimate] values to be fitted later (ARKitScenes for iPhone).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULTS = {
    "structured_light": dict(sigma_a=0.001, sigma_b=0.0015, edge_hole_px=(2, 6), edge_jump=0.06,
                             dark_hole_prob=0.3, dark_luma=0.08, max_range=8.0),
    "tof": dict(sigma_a=0.002, sigma_b=0.001, flying_prob=0.5, edge_jump=0.06, max_range=6.0),
    "iphone": dict(sigma_a=0.003, sigma_b=0.002, flying_prob=0.5, edge_jump=0.06, far_hole_start=5.0,
                   grid=(192, 256)),
}


def _jump_edges(depth, rel):
    """Pixels next to a relative depth discontinuity, plus which side is far."""
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    mx = ndimage.maximum_filter(np.nan_to_num(z, nan=0), size=3)
    mn = -ndimage.maximum_filter(-np.nan_to_num(z, nan=1e9), size=3)
    edge = np.isfinite(z) & ((mx - mn) / np.where(z > 0, z, 1) > rel)
    far_side = edge & (z >= (mx + mn) / 2)
    return edge, far_side, mn, mx


def apply_sensor(gt, gt_valid, rgb, kind, rng, **overrides):
    """Return (prompt float32 metres on the GT grid, valid bool, conf or None, info)."""
    p = dict(DEFAULTS[kind], **overrides)
    z = np.where(gt_valid, gt, np.nan).astype(np.float64)
    out = z + rng.normal(size=z.shape) * (p["sigma_a"] + p["sigma_b"] * np.nan_to_num(z) ** 2)
    valid = np.isfinite(out) & (out > 0)
    edge, far_side, mn, mx = _jump_edges(z, p["edge_jump"])
    info = dict(kind=kind, params=p)
    if kind == "structured_light":
        r = int(rng.integers(p["edge_hole_px"][0], p["edge_hole_px"][1] + 1))
        holes = ndimage.binary_dilation(far_side, iterations=r) & ~ndimage.binary_dilation(edge & ~far_side, iterations=1)
        valid &= ~holes
        luma = rgb.mean(-1) / 255.0
        valid &= ~((luma < p["dark_luma"]) & (rng.random(z.shape) < p["dark_hole_prob"]))
        info["edge_hole_px"] = r
    else:
        fly = edge & (rng.random(z.shape) < p["flying_prob"])
        a = rng.random(z.shape)
        out = np.where(fly, a * mn + (1 - a) * mx, out)             # mixed pixels between fore/background
        info["flying_pixels"] = int(fly.sum())
    if "max_range" in p:
        valid &= np.nan_to_num(out) <= p["max_range"]
    conf = None
    if kind == "iphone":
        far = np.nan_to_num(z) > p["far_hole_start"]
        valid &= ~(far & (rng.random(z.shape) < np.clip((np.nan_to_num(z) - p["far_hole_start"]) / 2, 0, 1)))
        conf = np.full(z.shape, 2, np.uint8); conf[edge] = 0; conf[np.nan_to_num(z) > 3.5] = 1
    out = np.where(valid, out, 0).astype(np.float32)
    return out, valid, conf, info
