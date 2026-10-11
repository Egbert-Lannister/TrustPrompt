"""Pair generator for T training (plan section 4.5), v0.

simulate(sample, rng) -> dict(P_clean, V_clean, P_corrupt, V_corrupt, applied, labels, info)

Region sources, in order of preference:
  * real masks: NYU mirror instances (material == 2) -> reflection;
    HAMMER object components (GT vs naked GT) -> penetration with exact `behind`, or metal;
  * planar patches found by RANSAC on the back-projected GT (area >= 2% of the image),
    cropped to a random rectangle -> reflection or penetration (ring-plane background).
With probability p_corrupt (0.6) 1-3 regions are corrupted; types are drawn
penetration 0.4 / reflection 0.4 / metal 0.2 (reflection requires a planar region;
metal / penetration may use any region). Region area is kept within 2-60% of the image,
and regions > 30% are kept with priority (plan: oversample large contamination).
Everything drawn is recorded in `info` so a training sample can be regenerated.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .corrupt import metal, penetration, reflection, type_labels
from .mirror_render import backproject
from .sensor import apply_sensor

SENSOR_OF = {"kinect1": "structured_light", "d435": "structured_light", "l515": "tof", "helios": "tof", "iphone": "iphone"}
TYPE_P = {"penetration": 0.4, "reflection": 0.4, "metal": 0.2}


def planar_regions(gt, gt_valid, K, rng, max_planes=3, iters=200, thr=0.01, min_frac=0.02, step=4):
    """Return a list of boolean masks, one per dominant plane (connected, >= min_frac of the image)."""
    h, w = gt.shape
    P = backproject(np.where(gt_valid, gt, np.nan), K)
    sub = P[::step, ::step].reshape(-1, 3); ok = np.isfinite(sub).all(1)
    pts = sub[ok]; free = np.ones(len(pts), bool); masks = []
    full = P.reshape(-1, 3); full_ok = np.isfinite(full).all(1)
    for _ in range(max_planes):
        idx = np.flatnonzero(free)
        if len(idx) < 100:
            break
        best, best_n = None, 0
        for _ in range(iters):
            a, b, c = pts[rng.choice(idx, 3, replace=False)]
            n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn; d = n @ a
            cnt = int((np.abs(pts[idx] @ n - d) < thr * np.abs(pts[idx, 2])).sum())
            if cnt > best_n:
                best, best_n = (n, d), cnt
        if best is None:
            break
        n, d = best
        free[idx[np.abs(pts[idx] @ n - d) < thr * np.abs(pts[idx, 2])]] = False
        inl = np.zeros(h * w, bool)
        inl[full_ok] = np.abs(full[full_ok] @ n - d) < thr * np.abs(full[full_ok, 2])
        inl = ndimage.binary_opening(inl.reshape(h, w), iterations=2)
        lab, nlab = ndimage.label(inl)
        if nlab == 0:
            continue
        sizes = ndimage.sum(inl, lab, range(1, nlab + 1))
        comp = lab == (1 + int(np.argmax(sizes)))
        if comp.mean() >= min_frac:
            masks.append(comp)
    return masks


def random_patch(mask, rng, min_frac=0.02, max_frac=0.6):
    """Random axis-aligned rectangle intersected with `mask`, area within [min_frac, max_frac]."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for _ in range(20):
        fy, fx = rng.uniform(0.3, 1.0, 2)
        hh, ww = max(4, int((ys.max() - ys.min() + 1) * fy)), max(4, int((xs.max() - xs.min() + 1) * fx))
        y0 = int(rng.integers(ys.min(), max(ys.min() + 1, ys.max() - hh + 2)))
        x0 = int(rng.integers(xs.min(), max(xs.min() + 1, xs.max() - ww + 2)))
        rect = np.zeros_like(mask); rect[y0:y0 + hh, x0:x0 + ww] = True
        r = rect & mask
        if min_frac <= r.mean() <= max_frac:
            return r
    return None


def simulate(sample, rng, p_corrupt=0.6, max_regions=3, tau=0.05, use_real_support=True):
    gt, gv, K = sample.gt, sample.gt_valid, sample.K_rgb
    kind = SENSOR_OF.get(sample.sensor, "structured_light")
    P0, V0, conf, sinfo = apply_sensor(gt, gv, sample.rgb, kind, rng)
    if use_real_support and sample.prompt.shape == gt.shape and not sample.metadata.get("synthetic", False):
        # Keep the real sensor's coverage (e.g. Helios ToF sees only the image centre) so that
        # P_clean has realistic support; real holes stay holes (label 3).
        V0 = V0 & sample.prompt_valid
        P0 = np.where(V0, P0, 0).astype(np.float32)
        sinfo["support"] = "model_holes_and_real_prompt_support"
    P, V = P0.copy(), V0.copy()
    applied = np.zeros(gt.shape, np.uint8)
    info = dict(sensor_model=sinfo, regions=[], corrupted=False)
    if rng.random() < p_corrupt:
        info["corrupted"] = True
        real = []
        if (sample.material == 2).any():
            lab, n = ndimage.label(sample.material == 2)
            real += [("mirror", lab == i) for i in range(1, n + 1)]
        if sample.behind is not None:
            obj = gv & (np.abs(gt - sample.behind) > 1e-3)
            lab, n = ndimage.label(ndimage.binary_opening(obj, iterations=1))
            real += [("object", lab == i) for i in range(1, n + 1) if (lab == i).mean() > 0.002]
        planes = planar_regions(gt, gv, K, rng)
        n_regions = int(rng.integers(1, max_regions + 1))
        used = np.zeros(gt.shape, bool)
        for _ in range(n_regions):
            t = rng.choice(list(TYPE_P), p=list(TYPE_P.values()))
            pool = [m for k, m in real if (k == "mirror") == (t == "reflection") or (k == "object" and t != "reflection")]
            if t == "reflection" or not pool:
                pool = pool + [m for m in (random_patch(pl, rng) for pl in planes) if m is not None]
            pool = [m & ~used for m in pool if (m & ~used).mean() >= 0.002]
            if not pool:
                continue
            big = [m for m in pool if m.mean() > 0.3]
            omega = (big or pool)[int(rng.integers(len(big or pool)))]
            try:
                if t == "penetration":
                    P, V, a, ri = penetration(P, V, gt, gv, omega, rng, behind=sample.behind)
                elif t == "reflection":
                    P, V, a, ri = reflection(P, V, gt, gv, omega, K, rng)
                else:
                    P, V, a, ri = metal(P, V, omega, rng)
            except ValueError as e:              # e.g. too few points for a mirror plane
                info["regions"].append(dict(type=str(t), status=f"skipped: {e}")); continue
            applied = np.where(a > 0, a, applied); used |= omega
            info["regions"].append(dict(ri, area_frac=float(omega.mean())))
    labels = type_labels(P, V, gt, gv, tau=tau)
    return dict(P_clean=P0, V_clean=V0, P_corrupt=P.astype(np.float32), V_corrupt=V, conf=conf,
                applied=applied, labels=labels, info=info)
