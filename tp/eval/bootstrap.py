"""Paired scene bootstrap and Holm multiple-testing adjustment."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def paired_scene_bootstrap(reference: Mapping, candidate: Mapping, n_resamples=10_000,
                           seed=0, confidence=0.95):
    """Bootstrap the mean paired difference (candidate minus reference).

    Each mapping has one finite scalar per *scene*. Exact keys must match;
    absent, empty and invalid metrics must be resolved explicitly upstream,
    never silently dropped here. Percentile CI uses paired scene resampling.
    The two-sided p-value is a centered/null bootstrap approximation with a
    +1 correction. One-scene intervals are descriptive and p is None.
    """
    if set(reference) != set(candidate):
        raise ValueError("paired scene sets differ; do not silently drop unmatched scenes")
    if not reference:
        raise ValueError("no scenes for paired bootstrap")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    scenes = sorted(reference)
    if any(reference[s] is None or candidate[s] is None for s in scenes):
        raise ValueError("empty/invalid scene metric cannot be silently removed")
    a = np.asarray([reference[s] for s in scenes], dtype=np.float64)
    b = np.asarray([candidate[s] for s in scenes], dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("bootstrap requires one finite scalar per scene")
    difference = b - a
    observed = float(np.mean(difference))
    rng = np.random.default_rng(seed)
    sampled = np.empty(int(n_resamples), dtype=np.float64)
    # Bound memory even when many scenes are present.
    for start in range(0, int(n_resamples), 1000):
        stop = min(start + 1000, int(n_resamples))
        indices = rng.integers(0, len(scenes), size=(stop - start, len(scenes)))
        sampled[start:stop] = np.mean(difference[indices], axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    centered = sampled - observed
    p_value = (float((1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (n_resamples + 1))
               if len(scenes) > 1 else None)
    return {
        "difference": "candidate_minus_reference", "mean_difference": observed,
        "ci_low": float(low), "ci_high": float(high), "confidence": float(confidence),
        "n_resamples": int(n_resamples), "seed": int(seed), "scene_count": len(scenes),
        "scenes": scenes, "p_value": p_value,
        "p_value_method": "two_sided_centered_paired_bootstrap_plus_one",
        "status": "ok" if len(scenes) > 1 else "descriptive_only_single_scene",
    }


def holm_adjust(p_values: Mapping[str, float]):
    """Return Holm-Bonferroni adjusted p-values with original hypothesis keys."""
    keys = list(p_values)
    if any(p_values[key] is None for key in keys):
        raise ValueError("missing p-values must be explicitly resolved before Holm correction")
    values = np.asarray([p_values[key] for key in keys], dtype=np.float64)
    if np.any(~np.isfinite(values) | (values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(keys), dtype=np.float64)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(keys) - rank) * values[index]))
        adjusted[index] = previous
    return {key: float(adjusted[index]) for index, key in enumerate(keys)}
