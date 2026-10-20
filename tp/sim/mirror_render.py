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


def _ray_xy(K, u, v):
    """Normalised ray (X/Z, Y/Z) of pixel (u, v) for an upper-triangular K, skew K[0, 1] included (Hypersim
    tilt-shift scenes); with zero skew the arithmetic equals the plain pinhole formula exactly."""
    y = (v - K[1, 2]) / K[1, 1]
    x = (u - K[0, 2] - K[0, 1] * y) / K[0, 0]
    return x, y


def project(points, K):
    """Pixel coordinates (u, v) of camera points [..., 3] (Z > 0)."""
    x, y = points[..., 0] / points[..., 2], points[..., 1] / points[..., 2]
    return K[0, 0] * x + K[0, 1] * y + K[0, 2], K[1, 1] * y + K[1, 2]


def backproject(depth, K):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64)
    x, y = _ray_xy(K, u, v)
    return np.stack([x * z, y * z, z], -1)


def reflect(points, n, d):
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
    dist = points @ n - d
    return points - 2.0 * dist[..., None] * n


def plane_depth_along_rays(K, shape, n, d):
    """Axial depth where each pixel ray meets the plane n.X = d (inf if parallel/behind)."""
    h, w = shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    ray = np.stack([*_ray_xy(K, u, v), np.ones_like(u)], -1)
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
    pu, pv = project(q, K)
    u, v = np.round(pu).astype(np.int64), np.round(pv).astype(np.int64)
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


def backwall_depth(K, shape, n, d, b):
    """Axial depth of the virtual image of a wall behind the camera (plane Z = -b, b > 0) seen in the mirror n.X = d.

    One view cannot render what a mirror reflects from behind the camera (render_mirror leaves it unresolved).
    This closes that gap with the simplest room: the camera ray to the mirror (axial depth z_m) is reflected,
    r' = r - 2 (r.n) n, and travels to the plane Z = -b; the sensor reports the unfolded path length along the
    original ray, Z_v = z_m + (z_m + b) / |r'_z| (r normalised to r_z = 1). NaN where the reflected ray does not
    head back towards the camera side (r'_z >= 0) or the mirror is not hit in front of the camera.
    """
    h, w = shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    ray = np.stack([*_ray_xy(K, u, v), np.ones_like(u)], -1)
    n = np.asarray(n, np.float64); n = n / np.linalg.norm(n)
    z_m = plane_depth_along_rays(K, shape, n, d)
    r_out_z = ray[..., 2] - 2.0 * (ray @ n) * n[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_v = z_m + (z_m + b) / np.abs(r_out_z)
    z_v[~np.isfinite(z_m) | (r_out_z >= 0)] = np.nan
    return z_v.astype(np.float32)
