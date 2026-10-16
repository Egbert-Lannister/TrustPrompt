"""Conformal risk control: monotone loss, bound arithmetic, finite-sample behaviour on synthetic data."""
import numpy as np

from tp.trust.conformal import GRID, crc_threshold, image_losses


def _item(rng, far_frac=0.2):
    gt = rng.uniform(1, 3, (32, 32)).astype(np.float32); p = gt.copy()
    far = rng.random(gt.shape) < far_frac; p[far] *= 1.5
    a = dict(prompt=p.astype(np.float16), gt=gt.astype(np.float16), valid=np.ones_like(far), gt_valid=np.ones_like(far),
             focus=np.ones_like(far))
    trust = np.where(far, rng.uniform(0, 0.6, gt.shape), rng.uniform(0.3, 1, gt.shape)).astype(np.float32)
    return a, trust


def test_loss_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    a, t = _item(rng)
    loss, ff, n = image_losses(t, a)
    assert n > 0 and np.all(np.diff(loss) <= 1e-12) and loss[0] == 1.0 and loss[-1] == 0.0 and np.all(np.diff(ff) >= -1e-12)


def test_crc_threshold_controls_risk_on_exchangeable_data():
    rng = np.random.default_rng(1)
    def batch(n):
        return np.array([image_losses(t, a)[0] for a, t in (_item(rng) for _ in range(n))])
    cal = batch(60); test = batch(400)
    lam, bnd = crc_threshold(cal, 0.1)
    k = int(np.flatnonzero(GRID == lam)[0])
    assert bnd <= 0.1 and test[:, k].mean() <= 0.1 + 0.02
    k_prev = max(k - 1, 0)
    assert k == 0 or 60 / 61 * cal[:, k_prev].mean() + 1 / 61 > 0.1        # smallest such lambda


def test_unreachable_alpha_returns_none():
    losses = np.ones((5, GRID.size)); losses[:, -1] = 0.5
    lam, best = crc_threshold(losses, 0.05)
    assert lam is None and best > 0.05
