"""Checks for the DEPTHOR++-style detector reimplementation."""
import numpy as np

from tp.models.depthor_det import detect, otsu_threshold, threshold


def room(h=120, w=160):
    v, u = np.mgrid[0:h, 0:w].astype(float)
    depth = 2.0 + 0.01 * u + 0.005 * v
    return u, v, depth


def test_clean_prompt_with_monotone_prior_flags_nothing():
    u, v, depth = room()
    rel = 0.3 * depth + 0.1                      # any increasing relation keeps ranks
    an, score, info = detect(depth, np.ones_like(depth, bool), rel, seed=1)
    assert info["gamma"] > 0.95 and info["threshold_mode"] == "gamma_gt_hi_no_anomaly"
    assert an.sum() == 0 and np.isfinite(score).all()


def test_medium_virtual_depth_region_is_flagged():
    u, v, depth = room()
    mirror = (u > 60) & (u < 110) & (v > 35) & (v < 75)           # ~10% of the image
    prompt = depth.copy(); prompt[mirror] = 7.0 + 0.002 * u[mirror]     # far virtual image
    an, score, info = detect(prompt, np.ones_like(depth, bool), depth, seed=1)
    assert info["status"] == "ok" and info["gamma"] <= 0.95
    recall = an[mirror].mean(); false_pos = an[~mirror].mean()
    assert recall > 0.9 and false_pos < 0.1
    assert np.nanmean(score[mirror]) > np.nanmean(score[~mirror])


def test_deterministic_and_invalid_pixels_never_flagged():
    u, v, depth = room()
    prompt = depth.copy(); prompt[40:80, 60:100] = 9.0
    valid = np.ones_like(depth, bool); valid[:, :20] = False
    a1, s1, _ = detect(prompt, valid, depth, seed=3, max_points=2000, n_ref=500)
    a2, s2, _ = detect(prompt, valid, depth, seed=3, max_points=2000, n_ref=500)
    assert np.array_equal(a1, a2) and np.array_equal(np.isnan(s1), np.isnan(s2))
    assert not a1[~valid].any() and np.isnan(s1[~valid]).all()


def test_threshold_regimes_and_otsu():
    x = np.r_[np.zeros(900), np.ones(100)]
    assert 0 < otsu_threshold(x) <= 1
    assert threshold(x, 0.99)[0] == np.inf
    assert threshold(x, 0.5)[1] == "otsu" and threshold(x, 0.9)[1] == "blend"


def test_large_region_is_still_found_but_rank_term_raises_false_positives():
    """Documents a property of the method (not a bug): when ~19% of the prompt is
    replaced by far virtual depth, every clean point's global rank shifts too."""
    u, v, depth = room()
    mirror = (u > 50) & (u < 110) & (v > 30) & (v < 90)
    prompt = depth.copy(); prompt[mirror] = 7.0 + 0.002 * u[mirror]
    an, _, _ = detect(prompt, np.ones_like(depth, bool), depth, seed=1)
    assert an[mirror].mean() > 0.9 and an[~mirror].mean() > 0.1


def test_small_region_is_skipped_by_the_spearman_gate_although_scores_separate():
    """Paper rule: gamma > 0.95 -> threshold +inf. A 3% virtual-depth patch keeps
    gamma above 0.95, so nothing is removed even though A separates the patch."""
    u, v, depth = room()
    mirror = (u > 70) & (u < 100) & (v > 40) & (v < 60)
    prompt = depth.copy(); prompt[mirror] = 7.0 + 0.002 * u[mirror]
    an, score, info = detect(prompt, np.ones_like(depth, bool), depth, seed=1)
    assert info["threshold_mode"] == "gamma_gt_hi_no_anomaly" and an.sum() == 0
    assert np.nanmean(score[mirror]) > 5 * np.nanmean(score[~mirror])


def test_numpy_spearman_matches_scipy_including_ties():
    from scipy import stats
    from tp.models.depthor_det import average_ranks, spearman
    rng = np.random.default_rng(0)
    a = rng.integers(0, 20, 500).astype(float); b = a + rng.normal(0, 5, 500)
    np.testing.assert_allclose(average_ranks(a), stats.rankdata(a))
    assert abs(spearman(a, b) - stats.spearmanr(a, b).statistic) < 1e-12


def test_detector_module_does_not_import_scipy_stats():
    import subprocess, sys
    code = "import sys, tp.models.depthor_det; sys.exit(1 if 'scipy.stats' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_quantised_detector_trust_keeps_risk_coverage_small():
    from tp.eval.metrics import trust_quality
    from tp.models.depthor_det import trust_from_score
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 5, (200, 300)); prompt = gt * (1 + rng.normal(0, 0.05, gt.shape))
    valid = np.ones_like(gt, bool)
    tw = np.round(trust_from_score(rng.uniform(0, 2, gt.shape)), 3)
    tq = trust_quality(tw, prompt, gt, valid, valid)
    assert tq["status"] == "ok" and len(tq["risk_coverage"]) <= 1001
