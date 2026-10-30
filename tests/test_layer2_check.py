import numpy as np

from scripts.layer2_check import CLASSES, classify, erode, verdict


def _scene():
    G = np.full((4, 5), 1.0, np.float32)
    B = np.full((4, 5), 2.0, np.float32)
    P = np.array([[1.0, 1.02, 0.5, 1.5, 2.0],      # correct, correct, front, between, layer2
                  [2.08, 3.0, 2.5, 0.0, 1.95],     # layer2, placebo (2B-G=3) + beyond, beyond, hole, layer2
                  [1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0]], np.float32)
    pv = P > 0
    gv = np.ones_like(pv)
    scope = np.zeros_like(pv)
    scope[:2] = True
    return P, pv, G, gv, B, scope


def test_classes_partition_c_on_separable_pixels():
    out = classify(*_scene())
    assert out["scope"] == 10 and out["hole"] == 1 and out["readings"] == 9
    assert out["correct"] == 2 and out["C"] == 7 and out["C_separable"] == 7
    assert (out["front"], out["between"], out["layer2"], out["beyond"]) == (1, 1, 3, 2)
    assert sum(out[k] for k in CLASSES) == out["C_separable"]
    assert out["placebo"] == 1
    assert sum(out["t_hist"]) == 7


def test_inseparable_layers_are_not_classified():
    P, pv, G, gv, B, scope = _scene()
    B = np.full_like(B, 1.08)                      # bands [0.95, 1.05] and [1.026, 1.134] overlap
    out = classify(P, pv, G, gv, B, scope)
    assert out["separable"] == 0 and out["C_separable"] == 0 and out["C_inseparable"] == out["C"]


def test_erode_removes_boundary():
    m = np.zeros((9, 9), bool)
    m[1:8, 1:8] = True
    e = erode(m, 2)
    assert e.sum() == 9 and e[3:6, 3:6].all()


def _obj(p, lo, placebo):
    return dict(layer2=(p, lo, p), placebo=placebo)


def test_verdict_rule():
    assert verdict(dict(pooled=_obj(0.7, 0.55, 0.05))) == "supported"
    assert verdict(dict(pooled=_obj(0.55, 0.40, 0.05))) == "leaning"
    assert verdict(dict(pooled=_obj(0.7, 0.55, 0.30))) == "not supported"          # placebo too close
    assert verdict(dict(pooled=_obj(0.3, 0.2, 0.02), helios=_obj(0.8, 0.6, 0.02),
                        d435=_obj(0.1, 0.05, 0.02))) == "sensor-specific (helios)"
