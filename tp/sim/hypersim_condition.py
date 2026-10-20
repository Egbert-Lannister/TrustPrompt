"""Mainline condition C (docs/plan_omnidc.md sec. 3): structured prompt errors on Hypersim frames.

The annotated surfaces themselves are corrupted, not random patches:
  mirror (NYU40 19)  every connected mirror region >= `min_area` of the image gets the reading of a depth sensor
                     looking at a mirror: the z-buffered virtual image of the visible scene reflected in the
                     region's fitted plane (tp.sim.corrupt.reflection). Most of what a mirror shows is behind the
                     camera and cannot be rendered from one view; with probability `p_backwall` those pixels see
                     the virtual image of a wall `backwall` metres behind the camera (mirror_render.backwall_depth),
                     otherwise they stay holes;
  window (NYU40 9)   the sensor sees through the glass: the reading is the surface behind it at `far_factor` x the
                     glass depth (outdoor / room beyond), then the sensor's maximum range removes what is too far
                     (tp.sim.corrupt.penetration with that background).
Readings beyond the sensor's maximum range become holes. The clean reading first comes from a structured-light or ToF sensor model on the GT (tp.sim.sensor), so holes and
noise look like the real sensors'. Labels follow T's definition (tp.sim.corrupt.type_labels: clean / far / near /
hole at tau). Every random draw comes from the caller's `rng`; `info` records what was applied.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from tp.data.hypersim import MIRROR, WINDOW
from .corrupt import penetration, reflection, type_labels
from .mirror_render import backwall_depth, fit_plane_from_depth
from .sensor import DEFAULTS, apply_sensor

SENSORS = ("structured_light", "tof")


def regions(semantic, label, valid, min_area):
    lab, n = ndimage.label(semantic == label)
    return [m for m in ((lab == i) & valid for i in range(1, n + 1)) if m.mean() >= min_area]


def condition(frame, rng, sensor=None, min_area=0.002, far_factor=(1.5, 4.0), p_backwall=0.8, backwall=(0.5, 3.0),
              tau=0.05):
    gt = np.where(frame.valid, frame.depth, 0).astype(np.float32)
    gv = frame.valid
    kind = sensor or str(rng.choice(SENSORS))
    P0, V0, _, sinfo = apply_sensor(gt, gv, frame.rgb, kind, rng)
    P, V = P0.copy(), V0.copy()
    applied = np.zeros(gt.shape, np.uint8)
    info = dict(sensor=kind, sensor_model=sinfo, regions=[])
    for omega in regions(frame.semantic, MIRROR, gv, min_area):
        try:
            P, V, a, ri = reflection(P, V, gt, gv, omega, frame.K, rng, unresolved="hole")
            n, d, _ = fit_plane_from_depth(gt, frame.K, omega & gv)
        except ValueError as e:                          # too few points for a plane
            info["regions"].append(dict(type="reflection", status=f"skipped: {e}")); continue
        ri["backwall_m"] = None
        if rng.random() < p_backwall:
            ri["backwall_m"] = b = float(rng.uniform(*backwall))
            zv = backwall_depth(frame.K, gt.shape, n, d, b)
            fill = omega & ~V & np.isfinite(zv)
            P = np.where(fill, zv, P).astype(np.float32); V = V | fill
            ri["backwall_px"] = int(fill.sum())
        applied = np.where(a > 0, a, applied)
        info["regions"].append(dict(ri, area_frac=float(omega.mean())))
    max_range = DEFAULTS[kind]["max_range"]
    for omega in regions(frame.semantic, WINDOW, gv, min_area):
        f = float(rng.uniform(*far_factor))
        P, V, a, ri = penetration(P, V, gt, gv, omega, rng, behind=np.where(omega, gt * f, np.nan).astype(np.float32))
        applied = np.where(a > 0, a, applied)
        info["regions"].append(dict(ri, type="window", behind_source="glass_depth_times_far_factor", far_factor=f,
                                    area_frac=float(omega.mean())))
    beyond = (applied > 0) & V & (P > max_range)
    V = V & ~beyond; P = np.where(beyond, 0, P).astype(np.float32)
    info["beyond_range_px"] = int(beyond.sum())
    labels = type_labels(P, V, gt, gv, tau=tau)
    return dict(P_clean=P0, V_clean=V0, P_corrupt=P.astype(np.float32), V_corrupt=V, applied=applied,
                labels=labels, info=info)

