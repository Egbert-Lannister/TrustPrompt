"""Non-Lambertian contamination (plan sections 4.2-4.4), v0.

Each function takes a clean prompt (metres, valid mask) and a region Omega and
returns a corrupted prompt, its validity, and a per-pixel `applied` code:
0 none, 1 penetration, 2 reflection, 3 metal. Labels for T are derived from the
corrupted prompt against GT (type_labels): 0 clean, 1 far, 2 near, 3 hole.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .mirror_render import fit_plane_from_depth, render_mirror

APPLIED = {"none": 0, "penetration": 1, "reflection": 2, "metal": 3}


def _ring(omega, inner=3, outer=10):
    a = ndimage.binary_dilation(omega, iterations=outer)
    b = ndimage.binary_dilation(omega, iterations=inner)
    return a & ~b


def _ring_plane_behind(gt, gt_valid, omega, rng):
    """Background behind Omega: 1/Z plane fitted on the outer ring, extrapolated inside;
    fallback to the 80th percentile ring depth when the ring is not coplanar."""
    ring = _ring(omega) & gt_valid & np.isfinite(gt) & (gt > 0)
    h, w = gt.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    if ring.sum() >= 30:
        A = np.stack([u[ring], v[ring], np.ones(ring.sum())], 1); y = 1.0 / gt[ring]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        rel = np.median(np.abs((A @ beta - y) / y))
        inv = beta[0] * u + beta[1] * v + beta[2]
        if rel < 0.02 and np.all(inv[omega] > 0):
            return 1.0 / inv, "ring_plane"
    if ring.sum() == 0:
        return None, "no_ring"
    return np.full(gt.shape, np.percentile(gt[ring], 80)), "ring_p80"


def penetration(prompt, valid, gt, gt_valid, omega, rng, behind=None, mix_prob=0.3):
    src = "naked_gt" if behind is not None else None
    if behind is None:
        behind, src = _ring_plane_behind(gt, gt_valid, omega, rng)
        if behind is None:
            return prompt, valid, np.zeros(gt.shape, np.uint8), dict(type="penetration", status="no_ring")
    p = prompt.copy(); vd = valid.copy()
    usable = omega & np.isfinite(behind) & (behind > 0)
    take = usable.copy()
    mixed = rng.random() < mix_prob
    if mixed:                                   # partial transmission: per-pixel mixture of surface and background
        alpha = rng.uniform(0.3, 0.8)
        take &= rng.random(gt.shape) < alpha
    p[take] = behind[take]; vd[take] = True
    applied = np.zeros(gt.shape, np.uint8); applied[take] = APPLIED["penetration"]
    return p, vd, applied, dict(type="penetration", behind_source=src, mixed=bool(mixed), pixels=int(take.sum()))


def reflection(prompt, valid, gt, gt_valid, omega, K, rng, unresolved="random"):
    n, d, resid = fit_plane_from_depth(gt, K, omega & gt_valid)
    virt, resolved = render_mirror(gt, gt_valid, K, omega, n, d)
    p = prompt.copy(); vd = valid.copy()
    p[resolved] = virt[resolved]; vd[resolved] = True
    rest = omega & ~resolved
    mode = rng.choice(["hole", "far"]) if unresolved == "random" else unresolved
    if mode == "hole":
        vd[rest] = False; p[rest] = 0
    else:
        far = float(np.nanmax(np.where(gt_valid, gt, np.nan))) * rng.uniform(1.2, 2.0)
        p[rest] = far; vd[rest] = True
    applied = np.zeros(gt.shape, np.uint8); applied[omega] = APPLIED["reflection"]
    return p, vd, applied, dict(type="reflection", plane_rmse=resid, resolved=int(resolved.sum()),
                                unresolved=int(rest.sum()), unresolved_mode=str(mode))


def metal(prompt, valid, omega, rng, drop=(0.2, 0.7), bias=(0.01, 0.10), speckle=0.01):
    p = prompt.copy(); vd = valid.copy()
    pd = rng.uniform(*drop); db = rng.uniform(*bias)
    dropped = omega & (rng.random(p.shape) < pd)
    vd[dropped] = False; p[dropped] = 0
    keep = omega & vd
    p[keep] = p[keep] * (1 + db) * (1 + speckle * rng.normal(size=int(keep.sum())))
    applied = np.zeros(p.shape, np.uint8); applied[omega] = APPLIED["metal"]
    return p, vd, applied, dict(type="metal", drop=pd, bias=db)


def type_labels(prompt, valid, gt, gt_valid, tau=0.05):
    """T labels: 0 clean, 1 far, 2 near, 3 hole; 255 = ignore (no GT)."""
    lab = np.full(gt.shape, 255, np.uint8)
    both = valid & gt_valid
    rel = np.zeros(gt.shape); rel[both] = (prompt[both] - gt[both]) / gt[both]
    lab[both & (np.abs(rel) <= tau)] = 0
    lab[both & (rel > tau)] = 1
    lab[both & (rel < -tau)] = 2
    lab[gt_valid & ~valid] = 3
    return lab
