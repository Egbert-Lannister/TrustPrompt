"""Fixed-denominator, unaligned metric depth and confidence evaluation."""
from __future__ import annotations

import numpy as np

from .regions import _check_arrays

CONTINUOUS = ("absrel", "rmse", "log_rmse")
DELTAS = {"delta_1_05": 1.05, "delta_1_10": 1.10, "delta_1_25": 1.25}
METRIC_NAMES = CONTINUOUS + tuple(DELTAS)


def _region_metrics(pred, gt, mask):
    count = int(np.count_nonzero(mask))
    p, g = pred[mask], gt[mask]
    valid = np.isfinite(p) & (p > 0)
    invalid_count = int(count - np.count_nonzero(valid))
    result = {
        "count": count, "invalid_count": invalid_count,
        "invalid_rate": invalid_count / count if count else None,
        "status": "empty" if not count else "invalid_predictions" if invalid_count else "ok",
        **{name: None for name in METRIC_NAMES},
    }
    if not count:
        return result
    # Invalid predictions are failures under every delta threshold. The
    # denominator remains the input mask count, never count(valid).
    ratio = np.maximum(p[valid] / g[valid], g[valid] / p[valid])
    for key, threshold in DELTAS.items():
        result[key] = float(np.count_nonzero(ratio < threshold) / count)
    if invalid_count:
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        values = (np.mean(np.abs(p - g) / g), np.sqrt(np.mean((p - g) ** 2)),
                  np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))
    if not all(np.isfinite(value) for value in values):
        result["status"] = "numerical_overflow"
        return result
    result.update({key: float(value) for key, value in zip(CONTINUOUS, values)})
    return result


def _rate(mask, valid_prediction, success):
    count = int(np.count_nonzero(mask))
    invalid_count = int(np.count_nonzero(mask & ~valid_prediction))
    successes = int(np.count_nonzero(mask & valid_prediction & success))
    return {
        "count": count, "invalid_count": invalid_count,
        "invalid_rate": invalid_count / count if count else None,
        "status": "empty" if not count else "invalid_predictions" if invalid_count else "ok",
        "value": successes / count if count and not invalid_count else None,
        "lower_bound": successes / count if count else None,
        "upper_bound": (successes + invalid_count) / count if count else None,
    }


def evaluate_depth(pred, gt, prompt, region_masks):
    """Return JSON-safe region metrics, IR(C), and DOR(C union band).

    Inputs are already at RGB resolution in metres/camera-axis Z. No clipping,
    alignment, scale fitting, or resampling is performed. Strict delta, IR and
    DOR comparisons follow the plan. Continuous metrics become None if *any*
    prediction is invalid in their fixed domain; status explains why. IR and
    DOR similarly become None and include conservative rate bounds.
    """
    pred, gt, prompt = (np.asarray(a, dtype=np.float64) for a in (pred, gt, prompt))
    if gt.ndim != 2 or pred.shape != gt.shape or prompt.shape != gt.shape:
        raise ValueError("pred, gt and prompt must have the same 2D shape")
    if any(key not in region_masks for key in ("all", "C", "band")):
        raise ValueError("region_masks must contain all, C and band")
    masks = {key: np.asarray(value) for key, value in region_masks.items()}
    for name, mask in masks.items():
        if mask.dtype != np.bool_ or mask.shape != gt.shape:
            raise ValueError(f"region {name} must be boolean with the ground-truth shape")
        if np.any(mask & (~np.isfinite(gt) | (gt <= 0))):
            raise ValueError(f"region {name} includes invalid ground truth")
        if np.any(mask & ~masks["all"]):
            raise ValueError(f"region {name} is outside the fixed all domain")
    if np.any(masks["C"] & (~np.isfinite(prompt) | (prompt <= 0))):
        raise ValueError("C includes invalid original sensor prompts")
    valid = np.isfinite(pred) & (pred > 0)
    inherited = np.zeros(gt.shape, dtype=bool)
    c_valid = masks["C"] & valid
    inherited[c_valid] = ((np.abs(pred[c_valid] - prompt[c_valid]) / prompt[c_valid] < 0.05)
                         & (np.abs(pred[c_valid] - gt[c_valid]) / gt[c_valid] > 0.05))
    dangerous = np.zeros(gt.shape, dtype=bool)
    dor_domain = masks["C"] | masks["band"]
    dor_valid = dor_domain & valid
    dangerous[dor_valid] = pred[dor_valid] > 1.10 * gt[dor_valid]
    return {
        "protocol": "metric_z_m_v1_no_alignment",
        "regions": {key: _region_metrics(pred, gt, mask) for key, mask in masks.items()},
        "IR": _rate(masks["C"], valid, inherited),
        "DOR": _rate(dor_domain, valid, dangerous),
    }


def trust_quality(w, prompt, gt, prompt_valid, gt_valid, tau=0.05):
    """Confidence quality on original valid prompts with valid ground truth.

    Positive = abs(P-GT)/GT <= tau, including clean boundary pixels. AUROC uses
    half credit for ties. AUPR is non-interpolated average precision evaluated
    at whole tied-score groups. Risk-coverage only reports attainable coverage
    at complete score groups; no arbitrary index ordering breaks ties.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    prompt, pv, gt, gv = _check_arrays(prompt, prompt_valid, gt, gt_valid)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != gt.shape:
        raise ValueError("w must have the same shape as ground truth")
    domain = pv & gv
    scores = w[domain]
    count = int(scores.size)
    invalid = ~np.isfinite(scores) | (scores < 0) | (scores > 1)
    result = {
        "count": count, "invalid_count": int(np.count_nonzero(invalid)),
        "status": "empty" if not count else "invalid_confidence" if np.any(invalid) else "ok",
        "label": "clean_all_relative_error_le_tau", "tau": float(tau),
        "auroc": None, "aupr": None, "positive_count": None,
        "negative_count": None, "risk_coverage": [],
        "aupr_definition": "noninterpolated_average_precision_grouped_ties",
    }
    if not count or np.any(invalid):
        return result
    risk = np.abs(prompt[domain] - gt[domain]) / gt[domain]
    label = risk <= tau
    n_pos, n_neg = int(np.count_nonzero(label)), int(np.count_nonzero(~label))
    result.update(positive_count=n_pos, negative_count=n_neg)
    order = np.argsort(-scores, kind="stable")
    scores, label, risk = scores[order], label[order], risk[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]) + 1, count]
    tp = np.cumsum(label, dtype=np.int64)[ends - 1]
    fp = ends - tp
    if n_pos and n_neg:
        tpr, fpr = np.r_[0, tp / n_pos], np.r_[0, fp / n_neg]
        result["auroc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    if n_pos:
        result["aupr"] = float(np.sum(np.diff(np.r_[0, tp / n_pos]) * tp / ends))
    if not n_pos or not n_neg:
        result["status"] = "single_class"
    cumulative_risk = np.cumsum(risk, dtype=np.float64)
    result["risk_coverage"] = [
        {"coverage": float(end / count), "retained_count": int(end),
         "risk_absrel": float(cumulative_risk[end - 1] / end),
         "threshold": float(scores[end - 1])}
        for end in ends
    ]
    return result
