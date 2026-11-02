import numpy as np

from scripts.local_consistency import box_sum, local_fit_residual, verdict


def test_box_sum_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.random((9, 11))
    w, r = 5, 2
    s = box_sum(x, w)
    p = np.pad(x, r)
    brute = np.array([[p[i:i + w, j:j + w].sum() for j in range(11)] for i in range(9)])
    assert np.allclose(s, brute)


def test_local_fit_explains_affine_readings():
    rng = np.random.default_rng(1)
    D = 1 + rng.random((20, 20))
    P = 1.7 * D + 0.3
    valid = np.ones_like(D, bool)
    assert local_fit_residual(D, P, valid, 7).max() < 1e-9


def test_local_fit_falls_back_to_shift_on_flat_prior():
    D = np.full((10, 10), 2.0)
    P = np.full((10, 10), 3.0)
    r = local_fit_residual(D, P, np.ones_like(D, bool), 5)
    assert np.allclose(r, 0)


def test_verdict():
    auroc = {"15": 0.52, "31": 0.55, "63": 0.62, "127": 0.70, "inf": 0.84}
    res = dict(nyu=dict(auroc=auroc, delta_inf_31=(0.29, 0.2, 0.35)))
    assert verdict(res)["outcome"] == "supported"
    auroc2 = dict(auroc, **{"31": 0.80})
    assert verdict(dict(nyu=dict(auroc=auroc2, delta_inf_31=(0.04, 0.01, 0.06))))["outcome"] == "not supported"
