import json

import numpy as np
import pytest

from scripts.train_trust import parse_fold
from tp.provenance import sha256, write_json
from tp.trust.cache import TrustCache


def test_parse_fold():
    assert parse_fold("test=scene2,scene3;hold=scene4") == ({"scene2", "scene3"}, {"scene4"})
    with pytest.raises(ValueError):
        parse_fold("test=scene2")


class S:
    sensor, scene, frame = "d435", "scene2", "traj1_1_000000"


def make_cache(root, per_item, with_thr=True):
    (root / "predictions").mkdir(parents=True)
    np.savez_compressed(root / "predictions" / "a.npz", trust=np.ones((192, 256), np.float16))
    e = dict(key="d435__scene2__traj1_1_000000", sample_sha256="x", path="predictions/a.npz")
    if with_thr:
        e["threshold"] = 0.37
    write_json(root / "index.json", [e])
    rec = dict(stage="trust_cache_verified", index_sha256=sha256(root / "index.json"))
    rec.update(per_item_threshold=True, threshold=None) if per_item else rec.update(threshold=0.25)
    write_json(root / "receipt.json", rec)
    return root


def test_trust_cache_per_item_threshold(tmp_path):
    c = TrustCache(make_cache(tmp_path / "cf", True))
    assert c.per_item and c.threshold_for(S) == pytest.approx(0.37)
    with pytest.raises(ValueError):
        c.threshold                       # a cross-fitted cache has no global threshold
    g = TrustCache(make_cache(tmp_path / "global", False))
    assert g.threshold == 0.25 and g.threshold_for(S) == 0.25
    with pytest.raises(ValueError):
        TrustCache(make_cache(tmp_path / "bad", True, with_thr=False))
