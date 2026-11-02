from scripts.tta_tmask import reduction, verdict


def _x(scene, c):
    return dict(scene=scene, metrics={k: dict(C_sum=v, C_n=1) for k, v in c.items()})


def test_reduction():
    sc = [[_x("a", dict(capa_raw=0.8, tmask=0.2))], [_x("b", dict(capa_raw=1.0, tmask=0.4))]]
    assert abs(reduction(sc, "tmask") - (1 - 0.6 / 1.8)) < 1e-9


def test_verdict():
    r = lambda p, lo: dict(rho=dict(tmask=(p, lo, p + 0.1)))
    assert verdict(dict(nyu=r(0.6, 0.3), hammer=r(0.1, 0.01)))["outcome"] == "rescues"
    assert verdict(dict(nyu=r(0.6, 0.3), hammer=r(0.1, -0.01)))["outcome"] == "nyu_only"
    assert verdict(dict(nyu=r(0.4, 0.3), hammer=r(0.1, 0.01)))["outcome"] == "does_not_rescue"
