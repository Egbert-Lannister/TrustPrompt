import json

import numpy as np
import pytest

from baselines.completion import exact_knn, registry, weight_file
from scripts.cache_completion import model_input, select_records
from scripts.make_synthetic import make_sample


def brute_knn(points, queries, k):
    d = np.sqrt(((queries[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    idx = np.arange(len(points))[None, :].repeat(len(queries), 0)
    order = np.lexsort((idx, d), axis=-1)            # (distance, index): torch_cluster CUDA tie rule
    return order[:, :k]


def test_exact_knn_matches_bruteforce_with_grid_ties():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:30, 0:40]
    grid = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    keep = rng.random(len(grid)) < 0.6
    pts, qs = grid[keep], grid[~keep]
    for k in (1, 5, 9):
        assert np.array_equal(exact_knn(pts, qs, k), brute_knn(pts, qs, k))


def test_exact_knn_wide_ties_need_requery():
    # 40 points on a circle of radius 5 around the query (integer lattice points at distance 5 plus duplicates)
    pts = np.array([[5, 0], [0, 5], [-5, 0], [0, -5], [3, 4], [4, 3], [-3, 4], [-4, 3], [3, -4], [4, -3], [-3, -4], [-4, -3]] * 3, float)
    q = np.zeros((1, 2))
    assert np.array_equal(exact_knn(pts, q, 5), brute_knn(pts, q, 5))
    assert list(exact_knn(pts, q, 5)[0]) == [0, 1, 2, 3, 4]


def test_torch_cluster_shim_layout_and_batches():
    torch = pytest.importorskip("torch")
    from baselines.completion import torch_cluster_shim
    knn = torch_cluster_shim().knn
    x = torch.tensor([[0., 0.], [1., 0.], [5., 5.], [0., 0.], [2., 2.]])
    bx = torch.tensor([0, 0, 0, 1, 1])
    y = torch.tensor([[0.2, 0.], [4., 4.], [1., 1.]])
    by = torch.tensor([0, 0, 1])
    out = knn(x=x, y=y, k=2, batch_x=bx, batch_y=by)
    assert out.shape == (2, 6)
    assert out[0].tolist() == [0, 0, 1, 1, 2, 2]
    assert out[1].tolist() == [0, 1, 2, 1, 3, 4]       # global x indices, (distance, index) order; ties -> lower index


def test_registry_is_complete_and_sensor_weights_resolve():
    reg = registry()
    assert set(reg["models"]) == {"priorda", "omnidc", "cdm", "lingbot", "any2full", "marigold_dc"}
    labels = [m["label"] for m in reg["models"].values()]
    assert len(set(labels)) == len(labels)
    for key, m in reg["models"].items():
        assert len(m["commit"]) == 40 and m["repo"].startswith("https://github.com/")
        for h in m["hf"]:
            assert len(h["revision"]) == 40
            for name, (size, digest) in h["files"].items():
                assert size > 0 and (digest is None or len(digest) == 64)
    files = {f"{h['repo']}/{n}" for h in reg["models"]["cdm"]["hf"] for n in h["files"]}
    for sensor, ref in reg["models"]["cdm"]["settings"]["sensor_weights"].items():
        assert ref in files, sensor
        assert weight_file(ref).name == ref.rsplit("/", 1)[1]


def same_grid_sample():
    from tp.geom import resize_nearest
    s = make_sample()
    hw = s.rgb.shape[:2]
    s.prompt = resize_nearest(s.prompt, hw).astype(np.float32); s.prompt_valid = resize_nearest(s.prompt_valid, hw)
    return s


def test_model_input_rejects_prompt_off_the_rgb_grid():
    with pytest.raises(ValueError):
        model_input(make_sample(), "raw", tau=0.05)


def test_model_input_variants_never_read_gt_unless_oracle():
    s = same_grid_sample()
    raw, info = model_input(s, "raw", tau=0.05)
    assert info["removed"] == 0 and np.all(raw[~s.prompt_valid] == 0) and np.all(raw[s.prompt_valid] == s.prompt[s.prompt_valid])
    trust = np.ones(s.prompt.shape, np.float32); trust[:5] = 0.1
    t, ti = model_input(s, "tmask", tau=0.05, trust=trust, threshold=0.5)
    assert ti["removed"] == int((s.prompt_valid[:5]).sum()) and np.all(t[:5] == 0)
    s2 = same_grid_sample(); s2.gt = s2.gt * 3        # changing GT must not change deployable inputs
    assert np.array_equal(model_input(s2, "raw", tau=0.05)[0], raw)
    assert np.array_equal(model_input(s2, "tmask", tau=0.05, trust=trust, threshold=0.5)[0], t)
    o, oi = model_input(s, "oracle_mask", tau=0.05)
    assert oi["removed"] > 0 and o.dtype == np.float32 and np.all(o >= 0)
    assert np.all(o[(raw > 0) & (o == 0)] == 0) and oi["kept"] + oi["removed"] == info["kept"]


def test_select_records_holdout_and_test_lock():
    recs = [dict(dataset="hammer", scene=s, split=sp, frame="f") for s, sp in
            (("scene10", "train"), ("scene3", "train"), ("scene11", "val"), ("scene12", "test"))]
    got = select_records(recs, {"train", "val"}, "holdout")
    assert [r["scene"] for r in got] == ["scene10", "scene11"]
    assert len(select_records(recs, {"train", "val"}, "all")) == 3
    with pytest.raises(ValueError):
        select_records(recs, {"train", "test"}, "holdout")
