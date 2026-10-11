"""Virtual-image depth of a planar mirror (plan section 4.3).

Given first-surface depth `gt` (axial Z, metres), intrinsics K and a mirror
region Omega on a plane n.X = d (camera frame, unit n), the sensor looking at
Omega sees the reflection of the rest of the visible scene. We back-project
every valid pixel outside Omega, reflect it across the plane, and z-buffer the
reflected points into the original camera. Only pixels in Omega are written.

A virtual point must lie behind the mirror along the pixel ray (Z_v greater
than the mirror depth); violators and pixels with no rendered point become
"unresolved": scene content behind the camera or outside the field of view
cannot be rendered from one view (documented limitation). The caller decides
whether unresolved pixels become holes or a far constant.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def backproject(depth, K):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1)


def reflect(points, n, d):
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
    dist = points @ n - d
    return points - 2.0 * dist[..., None] * n


def plane_depth_along_rays(K, shape, n, d):
    """Axial depth where each pixel ray meets the plane n.X = d (inf if parallel/behind)."""
    h, w = shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    denom = ray @ np.asarray(n, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d / denom
    z[~np.isfinite(z) | (z <= 0)] = np.inf
    return z


def fit_plane_from_depth(depth, K, mask):
    """Least-squares plane n.X = d through back-projected masked points (n points to camera side)."""
    P = backproject(depth, K)[mask & np.isfinite(depth) & (depth > 0)]
    if len(P) < 3:
        raise ValueError("need at least 3 points to fit a mirror plane")
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    d = float(n @ c)
    if d < 0:
        n, d = -n, -d
    resid = float(np.sqrt(np.mean(((P - c) @ vt[2]) ** 2)))
    return n, d, resid


def render_mirror(gt, gt_valid, K, omega, n, d, *, fill_small_holes=1):
    """Return (virtual_depth, resolved_mask) on the image grid; values only inside omega."""
    h, w = gt.shape
    pts = backproject(gt, K)
    src = gt_valid & ~omega & np.isfinite(gt) & (gt > 0)
    q = reflect(pts[src], n, d)
    front = q[:, 2] > 1e-6
    q = q[front]
    u = np.round(q[:, 0] / q[:, 2] * K[0, 0] + K[0, 2]).astype(np.int64)
    v = np.round(q[:, 1] / q[:, 2] * K[1, 1] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], q[inside, 2]
    keep = omega[v, u]
    u, v, z = u[keep], v[keep], z[keep]
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, v * w + u, z)           # nearest virtual surface wins
    zbuf = zbuf.reshape(h, w)
    if fill_small_holes:
        # Splatting gaps (magnified reflections): fill isolated holes from the 3x3 minimum.
        for _ in range(fill_small_holes):
            neigh = ndimage.minimum_filter(np.where(np.isfinite(zbuf), zbuf, np.inf), size=3)
            gap = omega & ~np.isfinite(zbuf) & np.isfinite(neigh)
            zbuf[gap] = neigh[gap]
    mirror_z = plane_depth_along_rays(K, (h, w), n, d)
    resolved = omega & np.isfinite(zbuf) & (zbuf > mirror_z * (1 + 1e-6))
    out = np.full((h, w), np.nan)
    out[resolved] = zbuf[resolved]
    return out.astype(np.float32), resolved
