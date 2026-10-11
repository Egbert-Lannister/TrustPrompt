"""Simulator v0: penetration, reflection integration, metal, labels, sensor models."""
import numpy as np

from tp.sim.corrupt import metal, penetration, reflection, type_labels
from tp.sim.sensor import apply_sensor

K = np.array([[200.0, 0, 100], [0, 200.0, 75], [0, 0, 1]])


def table_scene():
    v, u = np.mgrid[0:150, 0:200].astype(float)
    gt = 1.0 / (0.3 + 0.004 * v)                       # a receding floor/table plane
    obj = (u > 80) & (u < 120) & (v > 60) & (v < 100)
    gt_obj = gt.copy(); gt_obj[obj] = gt[obj] * 0.8      # object in front of the table
    return gt, gt_obj, obj


def test_penetration_with_and_without_behind_returns_background_depth():
    rng = np.random.default_rng(0)
    table, gt, obj = table_scene()
    valid = np.ones_like(obj)
    p, vd, applied, info = penetration(gt.copy(), valid, gt, valid, obj, rng, behind=table, mix_prob=0)
    np.testing.assert_allclose(p[obj], table[obj]); assert info["behind_source"] == "naked_gt"
    p2, _, _, info2 = penetration(gt.copy(), valid, gt, valid, obj, rng, behind=None, mix_prob=0)
    assert info2["behind_source"] == "ring_plane"
    np.testing.assert_allclose(p2[obj], table[obj], rtol=1e-6)      # ring is the table plane
    lab = type_labels(p2, valid, gt, valid)
    assert (lab[obj] == 1).all() and (lab[~obj] == 0).all()        # penetration = far


def test_reflection_on_a_wall_patch_renders_behind_the_mirror():
    rng = np.random.default_rng(1)
    v, u = np.mgrid[0:150, 0:200].astype(float)
    x = (u - 100) / 200
    gt = np.full((150, 200), 4.0)
    side = x > 0.25                                               # right side wall X = 1
    gt[side] = 1.0 / x[side]
    omega = side & (gt < 3.5) & (v > 40) & (v < 110)
    valid = np.ones_like(omega)
    p, vd, applied, info = reflection(gt.copy(), valid, gt, valid, omega, K, rng, unresolved="hole")
    res = omega & vd
    assert info["resolved"] > 0 and info["plane_rmse"] < 1e-9
    assert np.all(p[res] > gt[res])                               # virtual image is farther than the mirror
    assert (applied[omega] == 2).all()


def test_metal_drops_and_biases_inside_region_only():
    rng = np.random.default_rng(2)
    _, gt, obj = table_scene(); valid = np.ones_like(obj)
    p, vd, applied, info = metal(gt.copy(), valid, obj, rng, speckle=0)
    assert (~vd[obj]).mean() > 0.1 and vd[~obj].all()
    keep = obj & vd
    np.testing.assert_allclose(p[keep] / gt[keep], 1 + info["bias"])


def test_sensor_models_shapes_holes_and_noise_scale():
    rng = np.random.default_rng(3)
    _, gt, _ = table_scene(); valid = np.ones(gt.shape, bool)
    rgb = np.full(gt.shape + (3,), 128, np.uint8)
    for kind in ("structured_light", "tof", "iphone"):
        p, vd, conf, info = apply_sensor(gt, valid, rgb, kind, rng)
        assert p.shape == gt.shape and p.dtype == np.float32 and (p[~vd] == 0).all()
        rel = np.abs(p[vd] - gt[vd]) / gt[vd]
        assert np.median(rel) < 0.02
    assert conf is not None and conf.dtype == np.uint8


def test_simulate_on_synthetic_hammer_like_sample_is_reproducible():
    from tp.data.schema import Sample
    from tp.sim.generate import simulate
    table, gt, obj = table_scene()
    rgb = np.full(gt.shape + (3,), 120, np.uint8)
    s = Sample(rgb=rgb, prompt=gt.astype(np.float32), prompt_valid=np.ones(gt.shape, bool), gt=gt.astype(np.float32),
               gt_valid=np.ones(gt.shape, bool), material=np.zeros(gt.shape, np.uint8), K_rgb=K, K_prompt=K,
               sensor="l515", dataset="hammer", scene="scene3", frame="t", split="train", behind=table.astype(np.float32),
               metadata=dict(depth_unit="m", depth_convention="axial_z", synthetic=True))
    a = simulate(s, np.random.default_rng(7), p_corrupt=1.0)
    b = simulate(s, np.random.default_rng(7), p_corrupt=1.0)
    np.testing.assert_array_equal(a["P_corrupt"], b["P_corrupt"])
    assert a["info"]["corrupted"] and a["labels"].dtype == np.uint8
    assert set(np.unique(a["labels"])) <= {0, 1, 2, 3, 255}
