"""Deterministic record subsets shared by the caching and evaluation scripts (both must pick the same frames)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def per_group_subsample(records, n):
    """n evenly spaced frames (by frame name) of every (dataset, scene, sensor) group; groups with <= n keep all.
    Order of the returned records follows the input order."""
    if n is None:
        return list(records)
    if int(n) < 1:
        raise ValueError("per-group subsample size must be >= 1")
    groups = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["scene"], r.get("sensor"))].append(r["frame"])
    keep = set()
    for g, frames in groups.items():
        frames = sorted(frames)
        idx = np.unique(np.linspace(0, len(frames) - 1, min(int(n), len(frames))).round().astype(int))
        keep.update((*g, frames[i]) for i in idx)
    return [r for r in records if (r["dataset"], r["scene"], r.get("sensor"), r["frame"]) in keep]
