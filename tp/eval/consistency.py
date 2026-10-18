"""Shared helpers for checks that compare two prediction paths image by image (scripts/check_*.py)."""
from __future__ import annotations

import numpy as np

from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope


def quantize_cache(d):
    """The baseline caches' uint16 0.2 mm storage (scripts/cache_completion.store), invalid -> NaN, so both sides
    of a comparison carry the same rounding."""
    q = np.zeros(d.shape, np.float64)
    ok = np.isfinite(d) & (d > 0)
    q[ok] = np.clip(np.round(d[ok] / 2e-4), 1, 65535) * 2e-4
    q[~ok] = np.nan
    return q


def main_c_absrel(s, pred, tau=0.05):
    """Main-scope C AbsRel of one image; None for an empty region or an invalid prediction inside it."""
    c = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=tau, band=5)["C"] & focus_scope(s)[1]
    if not c.any() or not np.isfinite(pred[c]).all():
        return None
    return float(np.mean(np.abs(pred[c] - s.gt[c]) / s.gt[c]))


def relative_difference(a, b):
    """|a - b| / a over pixels finite in both (a = reference)."""
    ok = np.isfinite(a) & np.isfinite(b)
    return np.abs(a[ok] - b[ok]) / a[ok]
