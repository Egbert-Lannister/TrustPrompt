"""Method-independent evaluation regions for metric camera-axis Z depth.

The plan does not specify boundary morphology. Protocol v1 defines the boundary
as the *inner* four-connected boundary of C (image exterior is False), followed
by square/Chebyshev dilation of radius ``band``. A radius of zero retains the
one-pixel boundary. The band may overlap C, hole and clean_all; C, clean and hole
are disjoint, but do not partition all because clean excludes the band.
"""
from __future__ import annotations

import numpy as np


def _check_arrays(prompt, prompt_valid, gt, gt_valid):
    prompt, gt = np.asarray(prompt), np.asarray(gt)
    prompt_valid, gt_valid = np.asarray(prompt_valid), np.asarray(gt_valid)
    if gt.ndim != 2 or any(a.shape != gt.shape for a in (prompt, prompt_valid, gt_valid)):
        raise ValueError("prompt, prompt_valid, gt and gt_valid must have the same 2D shape")
    if prompt_valid.dtype != np.bool_ or gt_valid.dtype != np.bool_:
        raise ValueError("validity masks must be boolean")
    if np.any(gt_valid & (~np.isfinite(gt) | (gt <= 0))):
        raise ValueError("gt_valid contains nonfinite or nonpositive ground truth")
    if np.any(prompt_valid & (~np.isfinite(prompt) | (prompt <= 0))):
        raise ValueError("prompt_valid contains nonfinite or nonpositive prompt depth")
    return prompt.astype(np.float64), prompt_valid, gt.astype(np.float64), gt_valid


def _square_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Separable square dilation with False outside the image."""
    out = mask.copy()
    for axis in (0, 1):
        padded = np.pad(out.astype(np.int64), [(radius, radius) if a == axis else (0, 0) for a in (0, 1)])
        sums = np.cumsum(padded, axis=axis)
        sums = np.concatenate([np.zeros_like(np.take(sums, [0], axis=axis)), sums], axis=axis)
        width = 2 * radius + 1
        upper = [slice(None), slice(None)]
        lower = [slice(None), slice(None)]
        upper[axis] = slice(width, None)
        lower[axis] = slice(None, -width)
        out = (sums[tuple(upper)] - sums[tuple(lower)]) > 0
    return out


def regions(prompt_up, prompt_valid_up, gt, gt_valid, tau=0.05, band=5):
    """Build fixed masks from unfilled, nearest-neighbor-upsampled sensor data.

    ``clean_all`` is the trust-label domain (relative error <= tau), before band
    exclusion. It resolves the plan's ambiguity between section 3.1 evaluation
    clean and section 5.2 trust labels. Invalid data under a True validity mask
    raises instead of changing the evaluation denominator.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)) or band < 0:
        raise ValueError("band must be a nonnegative integer radius")
    prompt, pv, gt, gv = _check_arrays(prompt_up, prompt_valid_up, gt, gt_valid)
    both = pv & gv
    rel = np.zeros(gt.shape, dtype=np.float64)
    rel[both] = np.abs(prompt[both] - gt[both]) / gt[both]
    corrupt = both & (rel > tau)
    padded = np.pad(corrupt, 1, constant_values=False)
    interior = corrupt & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary = corrupt & ~interior
    boundary_band = _square_dilate(boundary, int(band)) & gv
    clean_all = both & ~corrupt
    return {
        "C": corrupt,
        "Cfar": corrupt & (prompt > gt),
        "Cnear": corrupt & (prompt < gt),
        "clean": clean_all & ~boundary_band,
        "hole": ~pv & gv,
        "band": boundary_band,
        "all": gv.copy(),
        "clean_all": clean_all,
    }
