import numpy as np

from tp.eval.auroc import ScoreHistogram


def exact_auroc(pos, neg):
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def test_histogram_auroc_matches_pairwise_on_binned_scores():
    rng = np.random.default_rng(0)
    pos, neg = rng.beta(3, 2, 2000), rng.beta(2, 3, 3000)
    h = ScoreHistogram(bins=1000)
    h.add(np.r_[pos, neg], np.r_[np.ones(2000, bool), np.zeros(3000, bool)])
    binned = lambda x: np.minimum((x * 1000).astype(int), 999)
    assert abs(h.auroc - exact_auroc(binned(pos), binned(neg))) < 1e-12
    assert abs(h.auroc - exact_auroc(pos, neg)) < 2e-3


def test_histogram_edge_cases_and_merge():
    h = ScoreHistogram(bins=10)
    assert h.auroc is None
    h.add([0.0, 1.0], [False, True])
    assert h.auroc == 1.0
    g = ScoreHistogram(bins=10); g.add([0.95, 0.05], [False, True])
    assert h.merge(g).auroc == 0.5 and h.counts() == dict(positive=2, negative=2)
    try:
        h.add([1.2], [True]); raise AssertionError("out of range accepted")
    except ValueError:
        pass
