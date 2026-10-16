"""T v0 data utilities (numpy only)."""
import json

import numpy as np
import pytest

from tp.geom import resize_nearest
from tp.trust.data import labels_from, scene_inner_holdout, sim_labels, to_grid, upsample_prob


def test_labels_follow_evaluation_rule():
    gt = np.full((4, 4), 2.0, np.float32); gv = np.ones_like(gt, bool); gv[0, 0] = False
    p = gt.copy(); p[1, 1] = 2.2; p[2, 2] = 2.05; pv = np.ones_like(gv); pv[3, 3] = False
    lab = labels_from(p, pv, gt, gv, tau=0.05)
    assert lab[1, 1] == 1 and lab[2, 2] == 0 and lab[0, 0] == 255 and lab[3, 3] == 255 and lab[0, 1] == 0


def test_sim_label_mapping():
    t = np.array([0, 1, 2, 3, 255], np.uint8)
    assert sim_labels(t).tolist() == [0, 1, 1, 255, 255]


def test_grid_arrays_refer_to_the_same_pixels():
    h, w = 456, 608
    yy, xx = np.mgrid[0:h, 0:w]
    p = (1 + (yy * w + xx) / 1e5).astype(np.float32); pv = np.ones((h, w), bool)
    g = to_grid(p, pv, p, np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8))
    assert g["prompt"].shape == (192, 256)
    np.testing.assert_array_equal(g["valid"], resize_nearest(pv, (192, 256)))
    np.testing.assert_allclose(g["dg"].astype(np.float32), g["prompt"].astype(np.float32))


def test_upsample_prob_is_identity_on_constant_and_monotone_on_ramp():
    assert np.allclose(upsample_prob(np.full((192, 256), 0.3), (456, 608)), 0.3)
    ramp = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (192, 1))
    up = upsample_prob(ramp, (832, 1088))
    assert up.shape == (832, 1088) and np.all(np.diff(up[0]) >= -1e-7) and up.min() >= 0 and up.max() <= 1


def test_inner_holdout_is_train_scene_rule():
    assert scene_inner_holdout("hammer", "scene10") and not scene_inner_holdout("hammer", "scene9")
    nyu = [f"bathroom_{i:04d}" for i in range(200)]
    frac = np.mean([scene_inner_holdout("mirror3d_nyu", s) for s in nyu])
    assert 0.1 < frac < 0.3


def test_trust_cache_reads_and_checks_hashes(tmp_path):
    from tp.provenance import sha256, write_json
    from tp.trust.cache import TrustCache

    class S:  # minimal sample
        sensor, scene, frame = "d435", "scene11", "f0"
        gt = np.zeros((832, 1088), np.float32)
    (tmp_path / "predictions").mkdir()
    np.savez_compressed(tmp_path / "predictions" / "d435__scene11__f0.npz", trust=np.full((192, 256), 0.25, np.float16))
    write_json(tmp_path / "index.json", [dict(key="d435__scene11__f0", sample_sha256="abc", path="predictions/d435__scene11__f0.npz")])
    write_json(tmp_path / "receipt.json", dict(stage="trust_cache_verified", index_sha256=sha256(tmp_path / "index.json"), threshold=0.4))
    c = TrustCache(tmp_path)
    t = c.trust(S(), "abc")
    assert t.shape == (832, 1088) and np.allclose(t, 0.25) and c.threshold == 0.4
    with pytest.raises(ValueError):
        c.trust(S(), "other")


def test_postprocess_fills_holes_inside_flagged_regions_only():
    from tp.trust.post import postprocess, postprocess_trust
    flag = np.zeros((40, 40), bool); flag[10:30, 10:30] = True; flag[18:22, 18:22] = False   # a hole in a mirror
    flag[2, 2] = False
    valid = np.ones_like(flag); valid[0:5, 0:5] = False
    out = postprocess(flag, valid)
    assert out[10:30, 10:30].all() and not out[:9].any() and not out[:, 31:].any()
    trust = np.where(flag, 0.1, 0.9).astype(np.float32)
    t2 = postprocess_trust(trust, 0.4, valid)
    assert np.array_equal(t2 < 0.4, out) and np.allclose(t2[~out], 0.9)


def test_replacement_error_rewards_flagging_exactly_the_bad_pixels():
    from tp.trust.data import replacement_error
    gt = np.full((192, 256), 2.0, np.float32); p = gt.copy(); p[50:100, 50:100] = 6.0
    dg = gt * 1.1
    a = dict(prompt=p.astype(np.float16), dg=dg.astype(np.float16), gt=gt.astype(np.float16), valid=np.ones_like(gt, bool),
             gt_valid=np.ones_like(gt, bool), focus=np.zeros_like(gt, bool))
    a["focus"][40:110, 40:110] = True
    perfect = np.where(p > 3, 0.0, 1.0)
    keep_all, flag_all = np.ones_like(gt), np.zeros_like(gt)
    sp, fp = replacement_error(perfect, 0.5, a); sk, fk = replacement_error(keep_all, 0.5, a); sa, fa = replacement_error(flag_all, 0.5, a)
    assert sp < sa < sk and fp < fa < fk and abs(sa - 0.1) < 1e-3


def test_labels3_sign_and_ignore():
    from tp.trust.data import labels3_from_arrays
    gt = np.full((2, 3), 2.0, np.float16); p = np.array([[2.0, 2.5, 1.5], [2.05, 0, 2.0]], np.float16)
    valid = p > 0; gv = np.ones_like(valid); gv[1, 2] = False
    lab = labels3_from_arrays(dict(prompt=p, gt=gt, valid=valid, gt_valid=gv))
    assert lab.tolist() == [[0, 1, 2], [0, 255, 255]]
