"""Region-consistent post-processing of a binary low-trust mask (T v1).

Mirrors and glass are coherent regions, so isolated misses inside a flagged region are
filled: 3x3 binary closing, then hole filling per connected flagged region, restricted to
valid prompt pixels. Nothing outside the closed regions is added.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def postprocess(flag, valid, close_iter=1):
    flag = np.asarray(flag, bool)
    closed = ndimage.binary_closing(flag, structure=np.ones((3, 3), bool), iterations=close_iter) | flag
    filled = ndimage.binary_fill_holes(closed)
    return filled & np.asarray(valid, bool)


def postprocess_trust(trust, threshold, valid):
    """Trust map whose thresholded mask equals postprocess(trust < threshold): newly
    flagged pixels get min(trust, threshold / 2)."""
    trust = np.asarray(trust, np.float32)
    flag = np.asarray(valid, bool) & (trust < threshold)
    add = postprocess(flag, valid) & ~flag
    out = trust.copy(); out[add] = np.minimum(out[add], threshold / 2)
    return out
