"""Sensor-only robust affine alignment in inverse depth (plan section 5.1).

For y=1/prompt and m=prior disparity, fit y=s*m+t. The 5% inlier test is
abs(s*m+t-y)/y <= 0.05 (relative *inverse-depth* error, denominator y).
Huber IRLS minimizes the corresponding relative residual. MAD is the unscaled
median absolute deviation of that signed relative residual, hence unitless.

Each fit uses at most 4096 uniformly sampled support pixels. The global model
uses 1000 seeded RANSAC trials followed by IRLS on its consensus. Local models
use all available valid supports in a 3x3 tile neighbourhood, subsampled to the
same cap, and IRLS initialized by the global model. Insufficient consensus or
degenerate local fits inherit the global solution. Coefficient interpolation
is bilinear between tile centres and constant beyond the outermost centres.

No RGB, GT, scene labels, or test-derived hyperparameters enter this module.
Invalid priors and nonpositive aligned inverse depths remain NaN; no clamping,
hole filling, or implicit return of the unaligned prior hides failed fits.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _residual(x, y, model):
    return (model[0] * x + model[1] - y) / y


def _model_valid(model):
    return model is not None and np.isfinite(model).all() and model[0] > 0


def _weighted_fit(x, y, weights):
    """Centred two-parameter WLS; reject unidentifiable scale/shift."""
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return None
    x_mean = np.sum(weights * x) / weight_sum
    y_mean = np.sum(weights * y) / weight_sum
    xc = x - x_mean
    variance = np.sum(weights * xc * xc) / weight_sum
    if variance <= 1e-12 * max(np.mean(x * x), 1e-12):
        return None
    slope = np.sum(weights * xc * (y - y_mean)) / (weight_sum * variance)
    model = np.array([slope, y_mean - slope * x_mean])
    return model if _model_valid(model) else None


def _irls(x, y, initial, huber_delta, iterations):
    model = np.asarray(initial, dtype=np.float64)
    # Normalization by a common constant improves conditioning, not the fit.
    inverse_y = np.min(y) / y
    relative_weights = inverse_y * inverse_y
    for _ in range(iterations):
        error = np.abs(_residual(x, y, model))
        huber_weights = np.ones_like(error)
        tail = error > huber_delta
        huber_weights[tail] = huber_delta / error[tail]
        updated = _weighted_fit(x, y, relative_weights * huber_weights)
        if updated is None:
            return None
        converged = np.allclose(updated, model, rtol=1e-8, atol=1e-10)
        model = updated
        if converged:
            break
    return model


def _take_support(x, y, rng, max_fit_points):
    if x.size <= max_fit_points:
        return x, y
    selected = rng.choice(x.size, size=max_fit_points, replace=False)
    return x[selected], y[selected]


def _ransac(x, y, rng, iterations, threshold, min_support):
    if np.var(x) <= 1e-12 * max(np.mean(x * x), 1e-12):
        return None, None, "degenerate_prior_disparity"
    best_model, best_inliers = None, None
    best_count, best_median = 0, np.inf
    pairs = rng.integers(0, x.size, size=(iterations, 2))
    tolerance = np.sqrt(1e-12 * max(np.mean(x * x), 1e-12))
    for first, second in pairs:
        dx = x[second] - x[first]
        if abs(dx) <= tolerance:
            continue
        scale = (y[second] - y[first]) / dx
        model = np.array([scale, y[first] - scale * x[first]])
        if not _model_valid(model):
            continue
        residual = np.abs(_residual(x, y, model))
        inliers = residual <= threshold
        count = int(inliers.sum())
        if count < min_support or count < best_count:
            continue
        median = float(np.median(residual[inliers]))
        if count > best_count or median < best_median:
            best_model, best_inliers = model, inliers
            best_count, best_median = count, median
    if best_model is None:
        return None, None, "no_positive_affine_consensus"
    return best_model, best_inliers, "ok"


def _depth(m, model_scale, model_shift, prior_valid):
    with np.errstate(over="ignore", invalid="ignore"):
        inverse_depth = model_scale * m + model_shift
    valid = prior_valid & np.isfinite(inverse_depth) & (inverse_depth > 0)
    depth = np.full(m.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        np.divide(1.0, inverse_depth, out=depth, where=valid)
    # The public result is float32: unrepresentable predictions are invalid too.
    valid &= np.isfinite(depth) & (depth <= np.finfo(np.float32).max)
    depth[~valid] = np.nan
    with np.errstate(under="ignore"):
        output = depth.astype(np.float32)
    valid &= np.isfinite(output) & (output > 0)
    output[~valid] = np.nan
    return output, int(np.count_nonzero(prior_valid & ~valid))


V2_EPSILON_FACTOR = 1e-6


def _local_v2(m, scale_map, shift_map, global_model, prior_valid, support_inverse_prompt, local_v1):
    """Plan revision R4: per-pixel global fallback where the interpolated local
    inverse depth is <= eps or nonfinite. eps is fixed relative to the median
    support inverse prompt (sensor-only, method-independent of GT).

    v1 is unchanged and returned separately by the caller; pixels whose global
    inverse depth is also nonpositive stay NaN and count as invalid.
    """
    epsilon = V2_EPSILON_FACTOR * float(np.median(support_inverse_prompt))
    with np.errstate(over="ignore", invalid="ignore"):
        local_inverse = scale_map * m + shift_map
    bad_local = prior_valid & ~(np.isfinite(local_inverse) & (local_inverse > epsilon))
    scale = np.where(bad_local, global_model[0], scale_map)
    shift = np.where(bad_local, global_model[1], shift_map)
    depth, invalid = _depth(m, scale, shift, prior_valid)
    # Local pixels that were valid but became unrepresentable as float32 also
    # remain NaN through _depth; they are counted below, never filtered.
    fallback = bad_local & np.isfinite(depth)
    still_invalid = prior_valid & ~np.isfinite(depth)
    diagnostics = {
        "rule": "global_fallback_where_local_inverse_depth_le_eps_or_nonfinite",
        "epsilon": epsilon, "epsilon_factor": V2_EPSILON_FACTOR,
        "epsilon_reference": "median_support_inverse_prompt",
        "fallback_pixel_count": int(fallback.sum()),
        "fallback_pixel_ratio": float(fallback.sum() / max(int(prior_valid.sum()), 1)),
        "v1_invalid_pixel_count": int((prior_valid & ~np.isfinite(local_v1)).sum()),
        "invalid_prediction_count": int((~np.isfinite(depth)).sum()),
        "invalid_on_valid_prior_count": int(still_invalid.sum()),
    }
    return depth, diagnostics


def gate_prompt(prompt_up, prompt_valid_up, aligned_depth, tau_gate=0.05):
    """RobustAlign-Gate (plan revision R5).

    Keep the sensor prompt where it is valid and agrees with the aligned prior
    within tau_gate relative depth; elsewhere output the aligned prior.
    Returns (depth, w) where w is the binary agreement map (NaN where the
    prompt is invalid, i.e. no trust statement). Invalid aligned pixels where
    the prompt is not kept stay NaN and are counted downstream.
    """
    p = np.asarray(prompt_up, dtype=np.float64)
    pv = np.asarray(prompt_valid_up)
    a = np.asarray(aligned_depth, dtype=np.float64)
    if p.shape != a.shape or pv.shape != a.shape or pv.dtype != np.bool_:
        raise ValueError("prompt, validity and aligned depth must share an HxW grid; validity boolean")
    if not (np.isfinite(tau_gate) and tau_gate > 0):
        raise ValueError("tau_gate must be positive and finite")
    sensor = pv & np.isfinite(p) & (p > 0)
    comparable = sensor & np.isfinite(a) & (a > 0)
    agree = np.zeros(a.shape, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        agree[comparable] = np.abs(p[comparable] - a[comparable]) / a[comparable] <= tau_gate
    depth = np.where(agree, p, a).astype(np.float32)
    depth[~(np.isfinite(depth) & (depth > 0))] = np.nan
    w = np.full(a.shape, np.nan, dtype=np.float32)
    w[sensor] = agree[sensor].astype(np.float32)
    return depth, w


def _tile_map(tile_values, y_edges, x_edges, shape):
    y_centres = (y_edges[:-1] + y_edges[1:] - 1) / 2
    x_centres = (x_edges[:-1] + x_edges[1:] - 1) / 2
    y = np.interp(np.arange(shape[0]), y_centres, np.arange(tile_values.shape[0]))
    x = np.interp(np.arange(shape[1]), x_centres, np.arange(tile_values.shape[1]))
    coordinates = np.meshgrid(y, x, indexing="ij")
    return map_coordinates(tile_values, coordinates, order=1, mode="nearest", prefilter=False)


def align_disparity(
    prior_disparity,
    prompt,
    prompt_valid,
    *,
    seed=0,
    ransac_iterations=1000,
    inlier_threshold=0.05,
    huber_delta=0.05,
    irls_iterations=20,
    min_support=50,
    max_fit_points=4096,
    grid=(8, 8),
):
    """Return global/local aligned depth plus maps and JSON-safe diagnostics.

    Inputs must share a 2D grid. The caller explicitly resamples original
    unfilled prompt and validity when necessary. Relative prior disparity may
    be zero or negative if finite; only the aligned inverse depth must be
    positive. A metric prior should use :func:`align_depth` instead.

    ``inlier_ratio`` and ``mad`` are bilinearly interpolated tile statistics,
    measured over full available neighbourhood support (not just fit samples).
    They describe the selected local/global-fallback model. ``disagreement``
    is abs(log(local_depth)-log(global_depth)); invalid comparisons stay NaN.
    """
    m = np.asarray(prior_disparity, dtype=np.float64)
    prompt = np.asarray(prompt, dtype=np.float64)
    pv = np.asarray(prompt_valid)
    if m.ndim != 2 or min(m.shape) == 0 or prompt.shape != m.shape or pv.shape != m.shape:
        raise ValueError("prior, prompt and validity must have matching nonempty HxW shapes")
    if pv.dtype != np.bool_:
        raise ValueError("prompt_valid must be boolean")
    integer_options = (ransac_iterations, irls_iterations, min_support, max_fit_points, *grid)
    if len(grid) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
                             for v in integer_options):
        raise ValueError("iteration counts, support limits and grid sizes must be positive integers")
    if min_support < 2 or max_fit_points < min_support:
        raise ValueError("require 2 <= min_support <= max_fit_points")
    if not (np.isfinite(inlier_threshold) and 0 < inlier_threshold < 1):
        raise ValueError("inlier_threshold must lie strictly between zero and one")
    if not (np.isfinite(huber_delta) and huber_delta > 0):
        raise ValueError("huber_delta must be positive and finite")
    rng = np.random.default_rng(seed)
    prior_valid = np.isfinite(m)
    sensor_valid = pv & np.isfinite(prompt) & (prompt > 0)
    inverse_prompt = np.full(m.shape, np.nan, dtype=np.float64)
    np.divide(1.0, prompt, out=inverse_prompt, where=sensor_valid)
    sensor_valid &= np.isfinite(inverse_prompt) & (inverse_prompt > 0)
    support = prior_valid & sensor_valid
    height, width = m.shape
    tile_h, tile_w = min(grid[0], height), min(grid[1], width)
    diagnostics = {
        "status": "pending", "seed": int(seed),
        "config": {"ransac_iterations": int(ransac_iterations),
                   "inlier_threshold": float(inlier_threshold), "huber_delta": float(huber_delta),
                   "irls_iterations": int(irls_iterations), "min_support": int(min_support),
                   "max_fit_points": int(max_fit_points), "grid": list(grid),
                   "effective_grid": [tile_h, tile_w], "neighbourhood": [3, 3],
                   "residual": "(s*m+t-1/P)/(1/P)", "mad": "unscaled_relative_inverse_depth",
                   "coefficient_interpolation": "bilinear_tile_centres_border_constant"},
        "support_count": int(support.sum()),
        "invalid_prior_count": int((~prior_valid).sum()),
        "invalid_prompt_marked_valid_count": int((pv & ~sensor_valid).sum()),
        "global_fit_sample_count": 0, "global_inlier_count": 0,
        "global_inlier_ratio": None, "global_scale": None, "global_shift": None,
        "invalid_global_prediction_count": int(m.size),
        "invalid_local_prediction_count": int(m.size),
        "tiles": [], "local_fallback_count": 0,
    }
    names = ("global_depth", "local_depth", "local_depth_v2", "scale_map", "shift_map", "inlier_ratio",
             "mad", "disagreement", "residual", "global_residual")
    result = {name: np.full(m.shape, np.nan, dtype=np.float32) for name in names}
    result["diagnostics"] = diagnostics
    if support.sum() < min_support:
        diagnostics["status"] = "insufficient_global_support"
        return result
    x, y = _take_support(m[support], inverse_prompt[support], rng, max_fit_points)
    diagnostics["global_fit_sample_count"] = int(x.size)
    model, consensus, status = _ransac(x, y, rng, ransac_iterations, inlier_threshold, min_support)
    if model is None:
        diagnostics["status"] = status
        return result
    model = _irls(x[consensus], y[consensus], model, huber_delta, irls_iterations)
    if not _model_valid(model):
        diagnostics["status"] = "invalid_global_refinement"
        return result
    global_residual = _residual(m[support], inverse_prompt[support], model)
    global_inliers = np.abs(global_residual) <= inlier_threshold
    if global_inliers.sum() < min_support:
        diagnostics["status"] = "insufficient_refined_global_consensus"
        return result
    diagnostics.update(status="ok", global_scale=float(model[0]), global_shift=float(model[1]),
                       global_inlier_count=int(global_inliers.sum()),
                       global_inlier_ratio=float(global_inliers.mean()))
    global_depth, invalid_global = _depth(m, model[0], model[1], prior_valid)
    diagnostics["nonpositive_or_nonfinite_global_on_valid_prior_count"] = invalid_global
    diagnostics["invalid_global_prediction_count"] = int((~np.isfinite(global_depth)).sum())
    scales = np.full((tile_h, tile_w), model[0])
    shifts = np.full((tile_h, tile_w), model[1])
    ratios = np.full((tile_h, tile_w), np.nan)
    mads = np.full((tile_h, tile_w), np.nan)
    y_edges = np.linspace(0, height, tile_h + 1).astype(int)
    x_edges = np.linspace(0, width, tile_w + 1).astype(int)
    for row in range(tile_h):
        for col in range(tile_w):
            region = (slice(y_edges[max(0, row - 1)], y_edges[min(tile_h, row + 2)]),
                      slice(x_edges[max(0, col - 1)], x_edges[min(tile_w, col + 2)]))
            region_support = support[region]
            all_x, all_y = m[region][region_support], inverse_prompt[region][region_support]
            local, reason, fit_count = model, "insufficient_support", 0
            if all_x.size >= min_support:
                local_x, local_y = _take_support(all_x, all_y, rng, max_fit_points)
                fit_count = int(local_x.size)
                candidate = _irls(local_x, local_y, model, huber_delta, irls_iterations)
                if _model_valid(candidate):
                    local_inliers = np.abs(_residual(all_x, all_y, candidate)) <= inlier_threshold
                    if local_inliers.sum() >= min_support:
                        local, reason = candidate, "local_irls"
                    else:
                        reason = "insufficient_local_consensus"
                else:
                    reason = "degenerate_or_nonpositive_local_model"
            fallback = reason != "local_irls"
            diagnostics["local_fallback_count"] += int(fallback)
            scales[row, col], shifts[row, col] = local
            if all_x.size:
                residual = _residual(all_x, all_y, local)
                ratios[row, col] = np.mean(np.abs(residual) <= inlier_threshold)
                mads[row, col] = np.median(np.abs(residual - np.median(residual)))
            diagnostics["tiles"].append({
                "row": row, "col": col, "support_count": int(all_x.size),
                "fit_sample_count": fit_count, "status": reason, "fallback_global": fallback,
                "scale": float(local[0]), "shift": float(local[1]),
                "inlier_ratio": float(ratios[row, col]) if all_x.size else None,
                "mad": float(mads[row, col]) if all_x.size else None,
            })
    scale_map = _tile_map(scales, y_edges, x_edges, m.shape)
    shift_map = _tile_map(shifts, y_edges, x_edges, m.shape)
    local_depth, invalid_local = _depth(m, scale_map, shift_map, prior_valid)
    diagnostics["nonpositive_or_nonfinite_local_on_valid_prior_count"] = invalid_local
    diagnostics["invalid_local_prediction_count"] = int((~np.isfinite(local_depth)).sum())
    local_v2, v2_diagnostics = _local_v2(m, scale_map, shift_map, model, prior_valid,
                                         inverse_prompt[support], local_depth)
    diagnostics["local_v2"] = v2_diagnostics
    result.update(global_depth=global_depth, local_depth=local_depth, local_depth_v2=local_v2,
                  scale_map=scale_map.astype(np.float32), shift_map=shift_map.astype(np.float32),
                  inlier_ratio=_tile_map(ratios, y_edges, x_edges, m.shape).astype(np.float32),
                  mad=_tile_map(mads, y_edges, x_edges, m.shape).astype(np.float32))
    both_valid = np.isfinite(global_depth) & np.isfinite(local_depth)
    result["disagreement"][both_valid] = np.abs(np.log(local_depth[both_valid]) - np.log(global_depth[both_valid]))
    for name, depth in (("residual", local_depth), ("global_residual", global_depth)):
        valid = sensor_valid & np.isfinite(depth)
        result[name][valid] = np.log(prompt[valid]) - np.log(depth[valid])
    return result


def align_depth(prior_depth, prompt, prompt_valid, **kwargs):
    """Metric-prior wrapper; invert only finite positive depth, preserving holes."""
    prior_depth = np.asarray(prior_depth, dtype=np.float64)
    disparity = np.full(prior_depth.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(prior_depth) & (prior_depth > 0)
    np.divide(1.0, prior_depth, out=disparity, where=valid)
    result = align_disparity(disparity, prompt, prompt_valid, **kwargs)
    result["diagnostics"]["input_prior"] = "metric_depth_inverted_at_finite_positive_pixels"
    return result
