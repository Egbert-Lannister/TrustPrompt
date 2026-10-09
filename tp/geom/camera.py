"""NumPy pinhole geometry for metric camera-axis Z depth.

Pixels denote integer-indexed pixel centres. Camera axes are x right, y down,
z forward. Resizing uses the half-pixel/centre-grid convention throughout.

The pinhole backprojection was independently adapted from GLARE's
glare/geometry.py (SHA256 1f535ee03f50a25df8e8bcc6c345d98dfcee5081ff88e3bee0e8f7850042b416).
Unlike its scaled_intrinsics, resize_intrinsics accounts for half-pixel offsets.
No GLARE package, cache, or runtime dependency is required.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
        raise ValueError("image shape must contain two positive integers (H, W)")
    return int(shape[0]), int(shape[1])


def _intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0]):
        raise ValueError("intrinsics last row must be [0, 0, 1]")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or np.linalg.det(matrix) == 0:
        raise ValueError("intrinsics must be invertible with positive focal lengths")
    return matrix


def pixel_rays(intrinsics: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return HxWx3 rays with unit Z (not unit Euclidean norm)."""
    height, width = _shape(shape)
    y, x = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    rays = np.linalg.solve(_intrinsics(intrinsics), pixels.reshape(-1, 3).T).T
    return rays.reshape(height, width, 3)


def backproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Backproject HxW axial Z in metres to HxWx3 camera points.

    Nonfinite or nonpositive depths produce three NaN coordinates.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape HxW")
    rays = pixel_rays(intrinsics, depth.shape)
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    return rays * z[..., None]


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project ...x3 camera points to (...x2 pixels, ... axial Z).

    Points behind the camera or with any nonfinite coordinate have NaN pixels;
    returned Z is the original Z coordinate, to support validity diagnostics.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points must have final dimension 3")
    homogeneous = points @ _intrinsics(intrinsics).T
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    pixels = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    np.divide(homogeneous[..., :2], homogeneous[..., 2, None], out=pixels,
              where=valid[..., None])
    return pixels, points[..., 2].copy()


def resize_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> np.ndarray:
    """Resize K with x_new + .5 = scale_x * (x_old + .5).

    Thus cx_new = (cx_old + .5) * scale_x - .5, with the same rule
    for cy. Supports skew through a full affine left multiplication.
    """
    source_h, source_w = _shape(source_size)
    destination_h, destination_w = _shape(destination_size)
    sx, sy = destination_w / source_w, destination_h / source_h
    transform = np.array([[sx, 0, (sx - 1) / 2],
                          [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return transform @ _intrinsics(intrinsics)


def resize_nearest(array: np.ndarray, destination_size: tuple[int, int]) -> np.ndarray:
    """Nearest resize of HxW[x...] with centre-grid sampling, preserving dtype.

    Equal-distance ties choose the larger source index. Masks and depth must
    use this same function so original sensor support remains aligned.
    """
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("array must have at least H and W dimensions")
    source_h, source_w = _shape(array.shape[:2])
    destination_h, destination_w = _shape(destination_size)
    y = np.floor((np.arange(destination_h) + .5) * source_h / destination_h).astype(int)
    x = np.floor((np.arange(destination_w) + .5) * source_w / destination_w).astype(int)
    return array[y[:, None], x[None, :]].copy()


def nearest_fill(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Fill native-grid holes from nearest positive finite valid sensor depth.

    Distance is Euclidean in native-grid pixel units (SciPy EDT tie order).
    An entirely empty prompt stays all-NaN; it cannot invent sensor evidence.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape HxW")
    _shape(depth.shape)
    support = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != depth.shape:
            raise ValueError("valid mask must match depth shape")
        support &= valid
    if not support.any():
        return np.full(depth.shape, np.nan, dtype=np.float32)
    if support.all():
        return depth.copy()
    indices = distance_transform_edt(~support, return_distances=False, return_indices=True)
    return depth[tuple(indices)].copy()


def fit_inverse_depth_plane(
    depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares fit 1/Z = a*x_ray + b*y_ray + c.

    Returns [a,b,c] in inverse metres. Requires three independent ray locations;
    no robust fitting or extrapolation heuristics are implied by this primitive.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape HxW")
    rays = pixel_rays(intrinsics, depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("mask must match depth shape")
        valid &= mask
    if valid.sum() < 3:
        raise ValueError("at least three valid points are required to fit a plane")
    coefficients, _, rank, _ = np.linalg.lstsq(rays[valid], 1 / depth[valid], rcond=None)
    if rank < 3:
        raise ValueError("plane fitting requires non-collinear ray locations")
    return coefficients


def inverse_plane_depth(
    coefficients: np.ndarray, intrinsics: np.ndarray, shape: tuple[int, int],
) -> np.ndarray:
    """Evaluate 1/Z = [a,b,c] dot ray; nonpositive inverse depth is NaN."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise ValueError("inverse-plane coefficients must be finite [a,b,c]")
    inverse_depth = pixel_rays(intrinsics, shape) @ coefficients
    result = np.full(inverse_depth.shape, np.nan, dtype=np.float64)
    np.divide(1.0, inverse_depth, out=result, where=inverse_depth > 0)
    return result


def plane_depth(
    normal: np.ndarray, offset: float, intrinsics: np.ndarray, shape: tuple[int, int],
) -> np.ndarray:
    """Render an explicit camera-space plane normal dot X + offset = 0.

    Positive ray intersections only; a plane through the camera is undefined.
    """
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.any(normal):
        raise ValueError("plane normal must be a nonzero finite vector of length 3")
    if not np.isfinite(offset) or offset == 0:
        raise ValueError("plane offset must be finite and nonzero")
    return inverse_plane_depth(-normal / offset, intrinsics, shape)
