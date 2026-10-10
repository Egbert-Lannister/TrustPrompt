"""Analytic checks for plan revision R4 (local v2 fallback) and R5 (gate)."""
import json

import numpy as np
import pytest

from tp.align.robust import _local_v2, align_depth, gate_prompt


def test_v2_falls_back_to_global_only_where_local_inverse_is_nonpositive():
    m = np.full((4, 4), 0.5)
    scale = np.full((4, 4), 2.0)
    shift = np.full((4, 4), 0.0)
    scale[0, :2] = -3.0          # interpolated local model extrapolates negative
    shift[1, 0] = np.nan          # nonfinite local coefficient
    prior_valid = np.ones((4, 4), dtype=bool)
    prior_valid[3, 3] = False
    m[3, 3] = np.nan
    v1 = np.where(scale * m + shift > 0, 1 / (scale * m + shift), np.nan)
    depth, diag = _local_v2(m, scale, shift, np.array([1.0, 0.5]), prior_valid,
                            np.array([0.9, 1.0, 1.1]), v1)
    np.testing.assert_allclose(depth[0, :2], 1.0)            # 1/(1*0.5+0.5)
    np.testing.assert_allclose(depth[1, 0], 1.0)
    np.testing.assert_allclose(depth[2, 2], 1.0)             # 1/(2*0.5)
    assert np.isnan(depth[3, 3])
    assert diag["fallback_pixel_count"] == 3
    assert diag["v1_invalid_pixel_count"] == 3
    assert diag["invalid_on_valid_prior_count"] == 0
    json.dumps(diag, allow_nan=False)


def test_v2_keeps_nan_when_global_is_also_nonpositive():
    m = np.array([[0.5, -10.0]])
    scale = np.array([[-1.0, 1.0]])         # both local inverses are nonpositive
    shift = np.zeros((1, 2))
    depth, diag = _local_v2(m, scale, shift, np.array([1.0, 0.1]), np.ones((1, 2), bool),
                            np.array([1.0]), np.full((1, 2), np.nan))
    np.testing.assert_allclose(depth[0, 0], 1 / 0.6, rtol=1e-6)
    assert np.isnan(depth[0, 1])
    assert diag["invalid_on_valid_prior_count"] == 1


def test_v2_equals_v1_wherever_v1_is_valid_on_clean_affine_data():
    rng = np.random.default_rng(3)
    m = rng.uniform(.1, .9, size=(48, 64))
    prompt = 1 / (1.7 * m + .12)
    result = align_depth(1 / m, prompt, np.ones(m.shape, bool), seed=1)
    v1, v2 = result["local_depth"], result["local_depth_v2"]
    assert np.isfinite(v1).all()
    np.testing.assert_array_equal(v1, v2)
    assert result["diagnostics"]["local_v2"]["fallback_pixel_count"] == 0


def test_gate_keeps_agreeing_prompt_and_replaces_disagreeing_or_missing():
    aligned = np.full((2, 3), 2.0, dtype=np.float32)
    prompt = np.array([[2.05, 3.0, 0.0], [1.0, 2.0, np.nan]])
    valid = np.array([[True, True, False], [True, True, True]])
    depth, w = gate_prompt(prompt, valid, aligned, tau_gate=0.05)
    np.testing.assert_allclose(depth, [[2.05, 2.0, 2.0], [2.0, 2.0, 2.0]], rtol=1e-6)
    np.testing.assert_array_equal(w[0, :2], [1, 0])
    assert np.isnan(w[0, 2]) and np.isnan(w[1, 2])     # no prompt -> no trust statement
    np.testing.assert_array_equal(w[1, :2], [0, 1])


def test_gate_propagates_invalid_aligned_prior_where_prompt_not_kept():
    aligned = np.array([[np.nan, 2.0]], dtype=np.float32)
    prompt = np.array([[5.0, 2.0]])
    depth, w = gate_prompt(prompt, np.ones((1, 2), bool), aligned)
    assert np.isnan(depth[0, 0]) and depth[0, 1] == pytest.approx(2.0)
    assert w[0, 0] == 0 and w[0, 1] == 1


def test_gate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        gate_prompt(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        gate_prompt(np.ones((2, 2)), np.ones((2, 2), bool), np.ones((2, 3)))
    with pytest.raises(ValueError):
        gate_prompt(np.ones((2, 2)), np.ones((2, 2), bool), np.ones((2, 2)), tau_gate=0)


def test_skipping_local_leaves_global_bit_identical():
    rng = np.random.default_rng(5)
    m = rng.uniform(.1, .9, size=(40, 56))
    prompt = 1 / (1.3 * m + .2)
    prompt[rng.random(m.shape) < .2] *= 3
    full = align_depth(1 / m, prompt, np.ones(m.shape, bool), seed=4)
    quick = align_depth(1 / m, prompt, np.ones(m.shape, bool), seed=4, compute_local=False)
    np.testing.assert_array_equal(full["global_depth"], quick["global_depth"])
    np.testing.assert_array_equal(full["global_residual"], quick["global_residual"])
    assert np.isnan(quick["local_depth"]).all() and quick["diagnostics"]["local_skipped"]
