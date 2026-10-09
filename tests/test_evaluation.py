"""Hand-checkable protocol regressions (no learned models or external data)."""
import json
import math

import numpy as np
import pytest

from tp.eval.aggregate import aggregate
from tp.eval.bootstrap import holm_adjust, paired_scene_bootstrap
from tp.eval.metrics import evaluate_depth, trust_quality
from tp.eval.regions import regions


def test_regions_hand_computed_4_by_4():
    gt = np.full((4, 4), 2.0)
    prompt = gt.copy()
    prompt[1, 1] = 4.0  # Far corruption.
    prompt[1, 2] = 1.0  # Near corruption.
    prompt[0, 3] = 0.0  # Missing prompt, valid GT.
    pv, gv = prompt > 0, np.ones((4, 4), bool)
    gv[3, 3] = False
    masks = regions(prompt, pv, gt, gv, band=1)
    expected_c = np.zeros((4, 4), bool)
    expected_c[1, 1:3] = True
    expected_band = np.zeros((4, 4), bool)
    expected_band[:3, :] = True
    expected_clean = np.zeros((4, 4), bool)
    expected_clean[3, :3] = True
    np.testing.assert_array_equal(masks["C"], expected_c)
    np.testing.assert_array_equal(masks["band"], expected_band)
    np.testing.assert_array_equal(masks["clean"], expected_clean)
    assert masks["Cfar"].sum() == masks["Cnear"].sum() == masks["hole"].sum() == 1
    assert masks["all"].sum() == 15
    assert not np.any(masks["C"] & masks["clean"])
    assert not np.any(masks["C"] & masks["hole"])
    assert masks["clean_all"].sum() == 12


def test_boundary_is_inner_four_connected_and_zero_radius_is_boundary():
    gt = np.ones((5, 5))
    prompt = np.full((5, 5), 2.0)
    valid = np.ones((5, 5), bool)
    mask = regions(prompt, valid, gt, valid, band=0)["band"]
    expected = np.ones((5, 5), bool)
    expected[1:-1, 1:-1] = False
    np.testing.assert_array_equal(mask, expected)


def test_hand_computed_metric_formulas_and_strict_thresholds():
    gt = np.array([[2.0, 4.0]])
    pred = np.array([[1.0, 8.0]])
    valid = np.ones_like(gt, dtype=bool)
    result = evaluate_depth(pred, gt, pred, regions(pred, valid, gt, valid, band=0))
    all_ = result["regions"]["all"]
    assert all_["absrel"] == pytest.approx(0.75)
    assert all_["rmse"] == pytest.approx(math.sqrt(8.5))
    assert all_["log_rmse"] == pytest.approx(math.log(2))
    assert all_["delta_1_25"] == 0
    assert result["IR"]["value"] == 1
    assert result["DOR"]["value"] == 0.5
    boundary_gt = np.ones((1, 3))
    boundary_pred = np.array([[1.05, 1.10, 1.25]])
    valid = np.ones((1, 3), bool)
    result = evaluate_depth(boundary_pred, boundary_gt, boundary_pred,
                            regions(boundary_pred, valid, boundary_gt, valid, band=0))
    assert result["regions"]["all"]["delta_1_05"] == 0
    assert result["regions"]["all"]["delta_1_10"] == pytest.approx(1 / 3)
    assert result["regions"]["all"]["delta_1_25"] == pytest.approx(2 / 3)
    assert result["DOR"]["value"] == pytest.approx(1 / 3)  # Equality at 1.10 is not dangerous.


def test_invalid_predictions_never_shrink_denominator_or_turn_into_good_errors():
    gt = np.ones((1, 4))
    prompt = np.array([[2.0, 2.0, 1.0, 1.0]])
    valid = np.ones((1, 4), bool)
    pred = np.array([[2.0, np.nan, 1.0, 0.0]])
    masks = regions(prompt, valid, gt, valid, band=0)
    result = evaluate_depth(pred, gt, prompt, masks)
    all_ = result["regions"]["all"]
    assert all_["count"] == 4 and all_["invalid_count"] == 2
    assert all_["invalid_rate"] == 0.5 and all_["status"] == "invalid_predictions"
    assert all_["absrel"] is all_["rmse"] is all_["log_rmse"] is None
    assert all_["delta_1_25"] == 0.25
    assert result["IR"]["value"] is result["DOR"]["value"] is None
    assert result["IR"]["lower_bound"] == result["DOR"]["lower_bound"] == 0.5
    assert result["IR"]["upper_bound"] == result["DOR"]["upper_bound"] == 1.0
    assert result["regions"]["hole"]["status"] == "empty"
    assert result["regions"]["hole"]["absrel"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_malformed_validity_masks_fail_loudly(invalid):
    gt = np.ones((1, 1))
    with pytest.raises(ValueError, match="prompt_valid"):
        regions(np.array([[invalid]]), np.ones((1, 1), bool), gt, np.ones((1, 1), bool))


def _record(scene, frame, prediction, prompt=1.0):
    gt = np.ones((1, 1))
    p = np.full_like(gt, prompt)
    valid = np.ones((1, 1), bool)
    return {"dataset": "toy", "scene": scene, "frame": frame, "split": "test",
            "metrics": evaluate_depth(np.full_like(gt, prediction), gt, p,
                                      regions(p, valid, gt, valid, band=0))}


def test_aggregation_balances_scenes_and_propagates_invalid_nonempty_values():
    rows = [_record("a", "0", 1.0), _record("a", "1", 1.0), _record("b", "0", 2.0)]
    combined = aggregate(rows)["datasets"]["toy"]
    assert combined["scene_count"] == 2 and combined["image_count"] == 3
    assert combined["metrics"]["regions"]["all"]["absrel"] == 0.5  # Not 1/3.
    rows[0] = _record("a", "0", float("nan"))
    combined = aggregate(rows)["datasets"]["toy"]["metrics"]["regions"]["all"]
    assert combined["absrel"] is None
    assert combined["status"] == "invalid_predictions"
    assert combined["invalid_count"] == 1
    assert combined["delta_1_25"] == 0.25


def test_aggregation_excludes_only_explicit_empty_regions():
    rows = [_record("a", "0", 1.0), _record("b", "0", 2.0, prompt=2.0)]
    c = aggregate(rows)["datasets"]["toy"]["metrics"]["regions"]["C"]
    assert c["absrel"] == 1.0 and c["empty_unit_count"] == 1
    rows[1]["scene"] = "a"
    rows[1]["frame"] = "1"
    rows[1]["split"] = "train"
    with pytest.raises(ValueError, match="scene crosses"):
        aggregate(rows)


def test_confidence_ties_are_grouped_and_permutation_invariant():
    gt = np.ones((1, 4))
    prompt = np.array([[1.0, 2.0, 1.0, 3.0]])
    valid = np.ones((1, 4), bool)
    equal_scores = np.full((1, 4), 0.5)
    tied = trust_quality(equal_scores, prompt, gt, valid, valid)
    assert tied["auroc"] == tied["aupr"] == 0.5
    assert tied["risk_coverage"] == [{"coverage": 1.0, "retained_count": 4,
                                      "risk_absrel": 0.75, "threshold": 0.5}]
    scores = np.array([[0.9, 0.9, 0.1, 0.1]])
    grouped = trust_quality(scores, prompt, gt, valid, valid)
    perm = [1, 0, 3, 2]
    permuted = trust_quality(scores[:, perm], prompt[:, perm], gt, valid, valid)
    assert grouped == permuted
    assert grouped["auroc"] == grouped["aupr"] == 0.5
    assert grouped["risk_coverage"][0]["coverage"] == 0.5
    assert grouped["risk_coverage"][0]["risk_absrel"] == 0.5


def test_confidence_perfect_order_and_invalid_scores():
    gt = np.ones((1, 4))
    prompt = np.array([[1.0, 1.0, 2.0, 2.0]])
    valid = np.ones((1, 4), bool)
    scores = np.array([[0.9, 0.8, 0.2, 0.1]])
    result = trust_quality(scores, prompt, gt, valid, valid)
    assert result["auroc"] == result["aupr"] == 1.0
    scores[0, 0] = np.nan
    result = trust_quality(scores, prompt, gt, valid, valid)
    assert result["status"] == "invalid_confidence" and result["auroc"] is None
    assert result["count"] == 4


def test_paired_bootstrap_reproducibility_constant_difference_and_holm():
    reference = {"a": 2.0, "b": 4.0, "c": 9.0}
    candidate = {"a": 1.0, "b": 3.0, "c": 8.0}
    result = paired_scene_bootstrap(reference, candidate, seed=43)
    assert result == paired_scene_bootstrap(reference, candidate, seed=43)
    assert result["n_resamples"] == 10000
    assert result["mean_difference"] == result["ci_low"] == result["ci_high"] == -1.0
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "b": 0.06, "c": 0.06}
    with pytest.raises(ValueError, match="scene sets differ"):
        paired_scene_bootstrap(reference, {"a": 1.0})
    with pytest.raises(ValueError, match="cannot be silently removed"):
        paired_scene_bootstrap({"a": None}, {"a": 1.0})
