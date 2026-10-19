"""AUROC over very many pixels from fixed-bin score histograms (scores in [0, 1]); ties within a bin count 1/2."""
from __future__ import annotations

import numpy as np


class ScoreHistogram:
    """Accumulates positive / negative score counts; `auroc` = P(score_pos > score_neg) + 1/2 P(same bin)."""

    def __init__(self, bins=1000):
        self.bins = int(bins)
        self.pos = np.zeros(self.bins, np.int64)
        self.neg = np.zeros(self.bins, np.int64)

    def _index(self, scores):
        s = np.asarray(scores, np.float64)
        if s.size and (not np.all(np.isfinite(s)) or s.min() < 0 or s.max() > 1):
            raise ValueError("scores must be finite and in [0, 1]")
        return np.minimum((s * self.bins).astype(np.int64), self.bins - 1)

    def add(self, scores, positive):
        positive = np.asarray(positive, bool)
        idx = self._index(scores)
        self.pos += np.bincount(idx[positive], minlength=self.bins)
        self.neg += np.bincount(idx[~positive], minlength=self.bins)

    def merge(self, other):
        self.pos += other.pos; self.neg += other.neg
        return self

    @property
    def auroc(self):
        n_pos, n_neg = self.pos.sum(), self.neg.sum()
        if n_pos == 0 or n_neg == 0:
            return None
        neg_below = np.cumsum(self.neg) - self.neg            # negatives in strictly lower bins
        return float((self.pos * (neg_below + 0.5 * self.neg)).sum() / (n_pos * n_neg))

    def counts(self):
        return dict(positive=int(self.pos.sum()), negative=int(self.neg.sum()))
