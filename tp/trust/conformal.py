"""Conformal risk control for the low-trust threshold (R16 item 2; Angelopoulos et al., 2022).

Per image, loss(lambda) = share of *far-contaminated* main-scope prompt pixels (P > GT (1 + tau)) that
stay unflagged (trust >= lambda); 0 when the image has none. It lies in [0, 1] and does not increase
with lambda. With n calibration images, lambda_hat = min{lambda : n/(n+1) * mean_loss(lambda) + 1/(n+1)
<= alpha} gives E[loss_test] <= alpha for an exchangeable test image. Grid search over lambda.
"""
from __future__ import annotations

import numpy as np

GRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def far_mask(a, tau=0.05):
    p = a["prompt"].astype(np.float32); gt = a["gt"].astype(np.float32)
    ok = a["valid"] & a["gt_valid"] & (p > 0) & (gt > 0)
    return ok & (p > gt * (1 + tau)) & a["focus"], ok


def image_losses(trust, a, grid=GRID, tau=0.05):
    """(loss per lambda, false-flag rate per lambda) for one image; false flags = flagged clean pixels."""
    far, ok = far_mask(a, tau)
    p = a["prompt"].astype(np.float32); gt = a["gt"].astype(np.float32)
    clean = ok & (np.abs(p - gt) <= tau * gt)
    tf, tc = trust[far], trust[clean]
    loss = np.array([(tf >= lam).mean() if tf.size else 0.0 for lam in grid])
    ff = np.array([(tc < lam).mean() if tc.size else 0.0 for lam in grid])
    return loss, ff, int(far.sum())


def crc_threshold(losses, alpha, grid=GRID):
    """losses: (n, len(grid)). Smallest lambda meeting the CRC bound, or None if none does."""
    n = losses.shape[0]
    bound = n / (n + 1) * losses.mean(0) + 1.0 / (n + 1)
    ok = np.flatnonzero(bound <= alpha)
    return (float(grid[ok[0]]), float(bound[ok[0]])) if ok.size else (None, float(bound.min()))
