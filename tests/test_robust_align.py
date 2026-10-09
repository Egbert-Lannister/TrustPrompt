"""Analytic synthetic checks only; these are not real-data experiment results."""

import json

import numpy as np
import pytest

from tp.align.robust import align_depth, align_disparity


def synthetic(shape=(48, 64), seed=17):
    rng = np.random.default_rng(seed)
    m = rng.uniform(.08, .9, size=shape)
    expected = 1 / (1.7 * m + .12)
    return rng, m, expected


def test_global_ransac_recovers_affine_with_outliers_and_holes():
    rng, m, expected = synthetic()
    prompt = expected.copy()
    corrupted = rng.random(m.shape) < .30
    prompt[corrupted] *= rng.uniform(2., 4., size=corrupted.sum())
    valid = rng.random(m.shape) > .10
    prompt[~valid] = 0.
    result = align_depth(1 / m, prompt, valid, seed=8)
    np.testing.assert_allclose(result["global_depth"], expected, rtol=1e-6)
    assert result["diagnostics"]["status"] == "ok"
    assert .6 < result["diagnostics"]["global_inlier_ratio"] < .8
    assert result["diagnostics"]["config"]["ransac_iterations"] == 1000
    assert np.isfinite(result["local_depth"]).all()
    assert result["mad"].shape == m.shape
    assert np.isnan(result["residual"][~valid]).all()
    json.dumps(result["diagnostics"], allow_nan=False)


def test_deterministic_sampling_caps_every_fit_and_preserves_inputs():
    _, m, prompt = synthetic()
    valid = np.ones(m.shape, dtype=bool)
    originals = (m.copy(), prompt.copy(), valid.copy())
    kwargs = dict(seed=42, ransac_iterations=80, max_fit_points=128)
    first = align_disparity(m, prompt, valid, **kwargs)
    second = align_disparity(m, prompt, valid, **kwargs)
    for name in first:
        if name == "diagnostics":
            assert first[name] == second[name]
        else:
            np.testing.assert_array_equal(first[name], second[name])
    assert first["diagnostics"]["global_fit_sample_count"] == 128
    assert all(tile["fit_sample_count"] <= 128 for tile in first["diagnostics"]["tiles"])
    for current, original in zip((m, prompt, valid), originals):
        np.testing.assert_array_equal(current, original)


def test_insufficient_and_degenerate_global_models_are_explicit_nan_failures():
    m = np.ones((16, 16))
    prompt = np.full_like(m, 2.)
    no_support = align_disparity(m, prompt, np.zeros(m.shape, dtype=bool))
    assert no_support["diagnostics"]["status"] == "insufficient_global_support"
    assert np.isnan(no_support["global_depth"]).all()
    degenerate = align_disparity(m, prompt, np.ones(m.shape, dtype=bool))
    assert degenerate["diagnostics"]["status"] == "degenerate_prior_disparity"
    assert np.isnan(degenerate["local_depth"]).all()


def test_sparse_neighbourhoods_fall_back_to_global_without_inventing_support():
    _, m, prompt = synthetic((64, 64))
    valid = np.zeros(m.shape, dtype=bool)
    valid[:8, :8] = True
    result = align_disparity(m, prompt, valid, seed=0, ransac_iterations=80)
    assert result["diagnostics"]["local_fallback_count"] > 0
    np.testing.assert_allclose(result["local_depth"], prompt, rtol=1e-6)
    empty_tiles = [tile for tile in result["diagnostics"]["tiles"] if tile["support_count"] == 0]
    assert empty_tiles and all(tile["inlier_ratio"] is None for tile in empty_tiles)
    assert all(tile["fallback_global"] for tile in empty_tiles)


def test_invalid_metric_prior_and_marked_valid_prompt_are_counted():
    _, m, prompt = synthetic()
    prior = 1 / m
    prior[0, :4] = [0., -1., np.nan, np.inf]
    prompt[1, :3] = [0., np.nan, -1.]
    result = align_depth(prior, prompt, np.ones(m.shape, dtype=bool), ransac_iterations=80)
    assert np.isnan(result["global_depth"][0, :4]).all()
    assert np.isnan(result["local_depth"][0, :4]).all()
    assert result["diagnostics"]["invalid_prior_count"] == 4
    assert result["diagnostics"]["invalid_prompt_marked_valid_count"] == 3
    assert result["diagnostics"]["invalid_global_prediction_count"] == 4


def test_constant_local_disparity_falls_back_to_identifiable_global_model():
    rng = np.random.default_rng(8)
    m = rng.uniform(.2, .8, (64, 64))
    m[:48] = .4
    expected = 1 / (1.7 * m + .12)
    result = align_disparity(m, expected, np.ones(m.shape, dtype=bool), ransac_iterations=150)
    assert result["diagnostics"]["status"] == "ok"
    assert any(tile["status"] == "degenerate_or_nonpositive_local_model"
               for tile in result["diagnostics"]["tiles"])
    np.testing.assert_allclose(result["local_depth"], expected, rtol=1e-6)


def test_nonpositive_extrapolation_stays_nan_instead_of_clipping():
    rng = np.random.default_rng(2)
    m = rng.uniform(.3, .9, (48, 64))
    prompt = 1 / (m - .2)
    m[0, :8] = .1
    valid = np.ones(m.shape, dtype=bool)
    valid[0, :8] = False
    result = align_disparity(m, prompt, valid, ransac_iterations=80)
    assert result["diagnostics"]["status"] == "ok"
    assert np.isnan(result["global_depth"][0, :8]).all()
    assert np.isnan(result["local_depth"][0, :8]).all()
    assert result["diagnostics"]["nonpositive_or_nonfinite_global_on_valid_prior_count"] == 8


def test_negative_scale_is_rejected_and_relative_disparity_can_be_negative():
    rng = np.random.default_rng(4)
    m = rng.uniform(-.5, .5, (32, 32))
    valid = np.ones(m.shape, dtype=bool)
    failed = align_disparity(m, 1 / (1 - m), valid, ransac_iterations=100)
    assert failed["diagnostics"]["status"] == "no_positive_affine_consensus"
    assert np.isnan(failed["global_depth"]).all()
    result = align_disparity(m, 1 / (1 + m), valid, ransac_iterations=100)
    np.testing.assert_allclose(result["global_depth"], 1 / (1 + m), rtol=1e-6)


def test_local_irls_reduces_smooth_spatial_bias_and_exposes_disagreement():
    rng, m, _ = synthetic((64, 64))
    x = np.linspace(-1, 1, m.shape[1])[None, :]
    expected = 1 / (1.7 * m + .12 + .02 * x)
    result = align_disparity(m, expected, np.ones(m.shape, dtype=bool), ransac_iterations=200)
    global_error = np.mean(np.abs(result["global_depth"] - expected))
    local_error = np.mean(np.abs(result["local_depth"] - expected))
    assert local_error < global_error * .4
    assert np.max(result["disagreement"]) > .001
    assert result["diagnostics"]["local_fallback_count"] == 0


def test_masks_and_fit_configuration_are_validated():
    _, m, prompt = synthetic()
    with pytest.raises(ValueError, match="boolean"):
        align_disparity(m, prompt, np.ones(m.shape))
    with pytest.raises(ValueError, match="min_support"):
        align_disparity(m, prompt, np.ones(m.shape, dtype=bool), max_fit_points=49)
