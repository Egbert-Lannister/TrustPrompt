"""Pre-registered comparison families with Holm adjustment (plan revision v2 sec. 3.7.5).

One comparison = candidate run vs reference run on the same fixed cohort (same sample hashes, same
method-independent region populations), one dataset, one population (val / train-holdout scenes), one primary
metric (main-scope C AbsRel by default). Difference = candidate - reference; negative = candidate better.

Resampling unit
  scene         >= 2 scenes in the population: paired scene bootstrap on scene means (tp.eval.bootstrap), the
                same image -> scene -> dataset weighting as every table.
  frame_block   one scene only (HAMMER val = scene11, HAMMER holdout = scene10): circular block bootstrap of
                frames within each trajectory (all sensors of a frame move together), block length L. This
                is inference about frames of *these* scenes, not about new scenes, and is labelled so.
Every comparison that cannot produce a p-value (missing / invalid values) enters the Holm family with p = 1,
listed, so the family size never shrinks after seeing results.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .bootstrap import holm_adjust, paired_scene_bootstrap


def _key(r):
    return (r["dataset"], r.get("sensor"), r["scene"], r["frame"], r["split"])


def per_image(rows, region="C", metric="absrel"):
    """{key: value | 'empty' | None}; None = invalid prediction in a non-empty region."""
    out = {}
    for r in rows:
        reg = r["metrics"]["regions"][region]
        if reg["status"] == "empty" or reg["count"] == 0:
            out[_key(r)] = "empty"
        else:
            v = reg.get(metric)
            out[_key(r)] = None if v is None or not math.isfinite(v) else float(v)
    return out


def frame_parts(dataset, scene, frame):
    """(trajectory id, frame index) used for frame blocks. HAMMER frames are '<traj>_<index>'."""
    if "_" in frame and frame.rsplit("_", 1)[1].isdigit():
        traj, idx = frame.rsplit("_", 1)
        return f"{scene}/{traj}", int(idx)
    return f"{scene}/-", int(frame) if frame.isdigit() else frame


def circular_block_bootstrap(diff_by_frame, block_length=8, n_resamples=10_000, seed=0, confidence=0.95):
    """diff_by_frame: {(traj, index): [per-image differences of that frame (one per sensor)]}.
    Resample each trajectory's frame sequence with circular blocks of `block_length` consecutive frames; the
    statistic is the mean over all resampled images (the single-scene dataset mean)."""
    by_traj = defaultdict(list)
    for (traj, idx), vals in diff_by_frame.items():
        by_traj[traj].append((idx, vals))
    trajs = sorted(by_traj)
    seqs = [[v for _, v in sorted(by_traj[t], key=lambda x: x[0])] for t in trajs]
    all_vals = [x for s in seqs for vals in s for x in vals]
    if not all_vals:
        raise ValueError("no paired images")
    observed = float(np.mean(all_vals))
    rng = np.random.default_rng(seed)
    # Pre-sum each frame so a resample is a sum over chosen frames / number of chosen images.
    sums = [np.asarray([float(np.sum(v)) for v in s]) for s in seqs]
    cnts = [np.asarray([len(v) for v in s]) for s in seqs]
    stats = np.empty(int(n_resamples))
    L = int(block_length)
    for b in range(int(n_resamples)):
        tot = 0.0; n = 0
        for sm, ct in zip(sums, cnts):
            m = len(sm)
            nb = int(math.ceil(m / L))
            starts = rng.integers(0, m, size=nb)
            idx = ((starts[:, None] + np.arange(L)[None, :]) % m).ravel()[:m]
            tot += sm[idx].sum(); n += ct[idx].sum()
        stats[b] = tot / n
    a = (1 - confidence) / 2
    low, high = np.quantile(stats, [a, 1 - a])
    centered = stats - observed
    p = float((1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (n_resamples + 1))
    return dict(difference="candidate_minus_reference", mean_difference=observed, ci_low=float(low), ci_high=float(high),
                confidence=confidence, n_resamples=int(n_resamples), seed=int(seed), block_length=L,
                trajectories=trajs, frame_count=int(sum(len(s) for s in seqs)), image_count=len(all_vals),
                p_value=p, p_value_method="two_sided_centered_circular_block_bootstrap_plus_one",
                status="ok_within_scene_frames",
                interpretation="frames of this one scene; does not generalise to new scenes")


def compare(ref_rows, cand_rows, *, dataset, region="C", metric="absrel", n_resamples=10_000, seed=0,
            block_length=8, unit=None):
    """unit None: paired scene bootstrap with >= 2 scenes, else frame blocks; unit "frame_block" forces circular
    frame blocks (per sequence) also with several scenes (inference about the frames of these scenes only)."""
    if unit not in (None, "scene", "frame_block"):
        raise ValueError(f"unknown bootstrap unit {unit}")
    ref = {k: v for k, v in per_image(ref_rows, region, metric).items() if k[0] == dataset}
    cand = {k: v for k, v in per_image(cand_rows, region, metric).items() if k[0] == dataset}
    if set(ref) != set(cand):
        raise ValueError("cohort mismatch between reference and candidate")
    empty = sorted(k for k, v in ref.items() if v == "empty")
    if any(cand[k] != "empty" for k in empty) or any(v == "empty" for k, v in cand.items() if ref[k] != "empty"):
        raise ValueError("empty-region mismatch: populations must be method-independent")
    keys = [k for k in ref if ref[k] != "empty"]
    bad = [k for k in keys if ref[k] is None or cand[k] is None]
    base = dict(dataset=dataset, region=region, metric=metric, images=len(keys), excluded_empty_images=len(empty))
    if not keys or bad:
        return dict(base, status="NA_invalid_or_missing" if keys else "NA_all_empty", p_value=None,
                    invalid_images=[list(map(str, k)) for k in bad][:20], n_invalid=len(bad))
    scenes = sorted({k[2] for k in keys})
    smean = lambda vals: float(np.mean([np.mean([vals[k] for k in keys if k[2] == sc]) for sc in scenes]))
    base.update(reference_mean=smean(ref), candidate_mean=smean(cand), scene_count=len(scenes))
    if (len(scenes) >= 2 and unit != "frame_block") or unit == "scene":
        sr, sc = defaultdict(list), defaultdict(list)
        for k in keys:
            sr[k[2]].append(ref[k]); sc[k[2]].append(cand[k])
        res = paired_scene_bootstrap({s: float(np.mean(v)) for s, v in sr.items()},
                                     {s: float(np.mean(v)) for s, v in sc.items()}, n_resamples=n_resamples, seed=seed)
        res["unit"] = "scene"
    else:
        d = defaultdict(list)
        for k in keys:
            d[frame_parts(dataset, k[2], k[3])].append(cand[k] - ref[k])
        res = circular_block_bootstrap(d, block_length=block_length, n_resamples=n_resamples, seed=seed)
        res["unit"] = "frame_block"
    res.update(base)
    return res


def holm_family(results, alpha=0.05):
    """results: {comparison id: compare(...) output}. Adds p_holm and reject; NA -> p = 1 (kept in the family)."""
    ps = {cid: (r["p_value"] if r.get("p_value") is not None else 1.0) for cid, r in results.items()}
    adj = holm_adjust(ps)
    for cid, r in results.items():
        r["p_for_holm"] = ps[cid]; r["p_holm"] = adj[cid]; r["reject_at_alpha"] = bool(adj[cid] <= alpha)
        r["holm_note"] = None if r.get("p_value") is not None else "no p-value (NA or single unit); entered as p = 1"
    return results
