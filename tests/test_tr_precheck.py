import numpy as np

from scripts.tr_precheck import c_reduction, clean_retention, image_stats, trust_region, variants, verdict


def test_trust_region_clips_relative_to_uncertainty():
    Dp = np.array([2.0, 2.0, 2.0])
    Dc = np.array([2.1, 4.0, 1.0])
    u = np.array([0.05, 0.05, 0.5])
    out = trust_region(Dp, Dc, u, 2.0)                  # widths 0.2, 0.2, 2.0
    assert np.allclose(out, [2.1, 2.2, 1.0])


def test_constant_width_uses_image_median():
    Dp = np.full(3, 2.0)
    Dc = np.full(3, 3.0)
    u = np.array([0.01, 0.1, 1.0])
    v = variants(Dp, Dc, u)
    assert np.allclose(v["const2"], 2.0 + 2 * 0.1 * 2.0)
    assert np.allclose(v["tr2"], [2.04, 2.4, 3.0])


def _row(scene, C, clean):
    ab = {k: dict(C=[C[k], 1], clean=[clean[k], 1], C_full=[0, 0], clean_full=[0, 0], invalid=0) for k in C}
    return dict(scene=scene, stats=dict(absrel=ab))


def test_reduction_and_retention():
    C = dict(prior=0.1, capa=0.9, tr2=0.3, const2=0.6)
    clean = dict(prior=0.2, capa=0.05, tr2=0.08, const2=0.07)
    sc = [[_row("a", C, clean)]]
    assert abs(c_reduction(sc) - (1 - 0.3 / 0.9)) < 1e-9
    assert abs(clean_retention(sc) - (0.2 - 0.08) / (0.2 - 0.05)) < 1e-9
    assert abs(c_reduction(sc, "tr2", "const2") - 0.5) < 1e-9


def test_image_stats_shapes():
    G = np.full((8, 8), 2.0, np.float32)
    P = G.copy(); P[:3] = 4.0
    pv = np.ones_like(G, bool); scope = pv.copy()
    u = np.full_like(G, 0.1)
    st = image_stats(P, pv, G, pv, scope, variants(G.copy(), P.copy(), u), u, np.random.default_rng(0))
    assert st["counts"]["C"] == 24
    assert st["absrel"]["prior"]["C"][0] == 0 and st["absrel"]["capa"]["C"][1] == 24
    assert len(st["spearman_sample"]["u"]) == 64


def _res(sp, red, red_lo, ret, cv_lo, tr_clean, const_clean):
    return dict(spearman_u_err=sp, c_reduction={2.0: (red, red_lo, 1.0)}, clean_retention={2.0: ret},
                c_vs_const=(0.1, cv_lo, 0.2), absrel=dict(tr2=dict(clean=tr_clean), const2=dict(clean=const_clean)))


def test_verdict():
    good = _res(0.3, 0.5, 0.2, 0.7, 0.01, 0.10, 0.10)
    assert verdict(dict(nyu=good, hammer=good))["outcome"] == "go"
    fixed = _res(0.3, 0.5, 0.2, 0.7, -0.01, 0.10, 0.10)
    assert verdict(dict(nyu=good, hammer=fixed))["outcome"] == "fixed_region_enough"
    bad = _res(0.05, 0.5, 0.2, 0.7, 0.01, 0.10, 0.10)
    assert verdict(dict(nyu=good, hammer=bad))["outcome"] == "fails"
