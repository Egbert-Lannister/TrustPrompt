"""Analytic checks for module C (fill candidates and rule v1)."""
import numpy as np

from tp.models.fill import fill_regions


def scene(h=80, w=100):
    v, u = np.mgrid[0:h, 0:w].astype(float)
    inv_plane = 0.4 + 0.002 * u + 0.001 * v          # a tilted wall: inverse depth affine in pixels
    return u, v, inv_plane


def test_mirror_like_region_is_refilled_exactly_by_plane_and_rule_picks_k3():
    u, v, inv = scene()
    gt = 1 / inv
    prompt = gt.copy()
    mirror = (u > 30) & (u < 70) & (v > 20) & (v < 60)
    prompt[mirror] = 6.0                              # virtual image: far and wrong
    trusted = ~mirror
    prior_inv = inv * (1 + 0.15 * (u / 100) ** 2)      # prior with a smooth shape error
    base = 1 / (prior_inv * np.median(inv / prior_inv))
    out, rec = fill_regions(base, prior_inv, prompt, trusted, mirror, grow=0)
    assert rec[0]["choice"] == "k3"
    np.testing.assert_allclose(out[mirror], gt[mirror], rtol=1e-6)
    out3, _ = fill_regions(base, prior_inv, prompt, trusted, mirror, grow=0, rule="k0")
    assert np.abs(out3[mirror] - gt[mirror]).max() > 1e-3          # k0 alone is not exact


def test_curved_object_is_not_flattened_k3_rejected_k2_recovers_shape():
    u, v, inv = scene()
    bump = np.exp(-(((u - 50) / 8) ** 2 + ((v - 40) / 8) ** 2))
    true_inv = inv + 0.8 * bump                        # an object in front of the wall
    gt = 1 / true_inv
    obj = bump > 0.05
    prompt = gt.copy()
    prompt[obj] = 1 / inv[obj]                         # sensor sees through the object (penetration)
    trusted = ~obj
    prior_inv = 2.0 * true_inv + 0.1                   # relative prior with the right shape
    base = 1 / ((prior_inv - 0.1) / 2.0)
    out, rec = fill_regions(base * 1.07, prior_inv, prompt, trusted, obj, grow=0)
    assert rec[0]["candidates"].get("k3") == "rejected_interior_not_planar"
    assert rec[0]["choice"] == "k2"
    np.testing.assert_allclose(out[obj], gt[obj], rtol=1e-4)


def test_small_or_ringless_regions_fall_back_to_k0_and_trusted_pixels_keep_prompt():
    u, v, inv = scene()
    gt = 1 / inv
    low = np.zeros_like(gt, bool); low[5:8, 5:8] = True             # 9 px < min_area
    base = gt * 1.01
    out, rec = fill_regions(base, inv, gt, ~low, low, grow=0)
    assert rec[0]["reason"] == "below_min_area" and rec[0]["choice"] == "k0"
    np.testing.assert_allclose(out[low], base[low]); np.testing.assert_allclose(out[~low], gt[~low])


def test_oracle_best_selects_lowest_error_candidate():
    u, v, inv = scene()
    gt = 1 / inv
    mirror = (u > 30) & (u < 70) & (v > 20) & (v < 60)
    prompt = gt.copy(); prompt[mirror] = 6.0
    prior_inv = inv * 1.3
    out, rec = fill_regions(gt * 1.2, prior_inv, prompt, ~mirror, mirror, grow=0, rule="oracle_best", gt=gt)
    errs = rec[0]["oracle_errors"]
    assert rec[0]["choice"] == min(errs, key=errs.get) and errs[rec[0]["choice"]] < 1e-5
