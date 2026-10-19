import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.completion import source_dir
from tp.omnidc.onesided import input_weights, integrate

ALPHA = 5.0


def chain_problem(n=40, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 0.05, n - 1)
    truth = np.r_[0, np.cumsum(g)]
    p = truth + rng.normal(0, 0.01, n)
    p[10:20] += 0.6                                        # a run of readings behind the surface
    t = lambda a: torch.tensor(a, dtype=torch.float64).reshape(1, 1, 1, -1)
    w_eq = np.ones(n); w_eq[10:20] = 0.02                  # low confidence where the readings are wrong
    far = np.zeros(n); far[8:22] = 1.0
    return g, t(p), torch.ones(1, 1, 1, n, dtype=torch.float64), t(w_eq), t(far)


def chain_solver(g):
    """Exact minimiser of sum (x_{i+1} - x_i - g_i)^2 + ALPHA * sum w_i (x_i - p_i)^2 for the given weights."""
    n = len(g) + 1
    D = np.diff(np.eye(n), axis=0)

    def solve(weights, x_init, p):
        w = weights[0, 0, 0].numpy()
        x = np.linalg.solve(D.T @ D + ALPHA * np.diag(w), D.T @ g + ALPHA * w * p[0, 0, 0].numpy())
        return torch.tensor(x).reshape(1, 1, 1, -1)
    return solve


def test_active_set_reaches_the_hinge_minimiser():
    from scipy.optimize import minimize
    g, p, valid, w_eq, far = chain_problem()
    lam = 4.0
    solve = chain_solver(g)
    x, active, rounds = integrate(lambda w, x0: solve(w, x0, p), p, valid, w_eq, far, lam=lam, max_rounds=10)
    pn, we, fn = p[0, 0, 0].numpy(), w_eq[0, 0, 0].numpy(), far[0, 0, 0].numpy()
    obj = lambda z: (np.sum((np.diff(z) - g) ** 2) + ALPHA * np.sum(we * (z - pn) ** 2)
                     + ALPHA * lam * np.sum(fn * np.maximum(z - pn, 0) ** 2))
    ref = minimize(obj, pn, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12)).x
    assert rounds < 10 and np.allclose(x[0, 0, 0].numpy(), ref, atol=1e-5)
    assert np.array_equal(active[0, 0, 0].numpy() > 0, x[0, 0, 0].numpy() > pn)      # KKT: S = {x > p} at the end


def test_hinge_bounds_a_downweighted_run_that_would_drift_behind_the_reading():
    """A down-weighted run at the image border (one anchored side only) whose predicted gradients drift backwards:
    without the hinge the estimate ends far behind the readings, with it it stays at or in front of them."""
    n = 40
    g = np.zeros(n - 1); g[29:] = 0.08
    p = np.zeros(n); p[30:] = 0.15
    t = lambda a: torch.tensor(a, dtype=torch.float64).reshape(1, 1, 1, -1)
    w_eq = np.ones(n); w_eq[30:] = 0.01
    far = np.zeros(n); far[30:] = 1.0
    solve = chain_solver(g)
    pt = t(p)
    plain = solve(t(w_eq), None, pt)
    x, _, _ = integrate(lambda w, x0: solve(w, x0, pt), pt, torch.ones_like(pt), t(w_eq), t(far), lam=50.0, max_rounds=10)
    behind_plain = float((plain - pt)[0, 0, 0, 30:].max())
    behind_hinge = float((x - pt)[0, 0, 0, 30:].max())
    assert behind_plain > 0.2 and behind_hinge < 0.05 * behind_plain


def test_without_far_weight_it_is_the_plain_solve():
    g, p, valid, w_eq, far = chain_problem(seed=1)
    solve = chain_solver(g)
    x, _, _ = integrate(lambda w, x0: solve(w, x0, p), p, valid, w_eq, torch.zeros_like(far), lam=4.0)
    assert torch.equal(x, solve(w_eq, None, p))


@pytest.fixture(scope="module")
def ddi():
    src = source_dir("omnidc") / "src" / "model"
    if not (src / "optim_layer" / "optim_layer.py").is_file():
        pytest.skip("OMNI-DC checkout missing")
    sys.path.insert(0, str(src))
    try:
        from optim_layer.optim_layer import DepthGradOptimLayer
    finally:
        sys.path.remove(str(src))
    return DepthGradOptimLayer


def test_gradients_through_omnidc_ddi_with_the_hinge(ddi):
    torch.manual_seed(0)
    B, H, W, R, dt = 1, 8, 8, 3, torch.float64
    grads = (torch.randn(B, 2 * R, H, W, dtype=dt) * 0.1).requires_grad_(True)
    valid = (torch.rand(B, 1, H, W) < 0.6).to(dt)
    logp = (torch.rand(B, 1, H, W, dtype=dt) * 0.5) * valid
    conf = torch.rand(B, 2 * R, H, W, dtype=dt) + 0.5
    w_eq = torch.cat([torch.rand(B, 1, H, W, dtype=dt) + 0.1, torch.zeros(B, R - 1, H, W, dtype=dt)], 1).requires_grad_(True)
    far = torch.rand(B, 1, H, W, dtype=dt).requires_grad_(True)
    b0 = torch.zeros(B, 1, H, W, dtype=dt, requires_grad=True)

    def run(gr, we, fa):
        solve = lambda w, x0: ddi.apply(gr, logp, valid, conf, w, R, x0, b0, ALPHA, 1e-12, 5000)[0]
        return integrate(solve, logp, valid, we, fa, lam=3.0)

    x, active, rounds = run(grads, w_eq, far)
    assert active.sum() > 0 and rounds >= 1
    solve = lambda w: ddi.apply(grads, logp, valid, conf, w, R, None, b0, ALPHA, 1e-12, 5000)[0]
    assert torch.allclose(solve(input_weights(w_eq, far, active, 3.0)), x, atol=1e-8)   # fixed point of the last set
    assert torch.autograd.gradcheck(lambda gr, we, fa: run(gr, we, fa)[0], (grads, w_eq, far), eps=1e-6, atol=1e-5)
