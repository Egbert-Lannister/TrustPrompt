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
    raw, g, info = model_input(s, "raw", tau=0.05)
    assert g is None and info["removed"] == 0
    assert np.all(raw[~s.prompt_valid] == 0) and np.all(raw[s.prompt_valid] == s.prompt[s.prompt_valid])
    trust = np.ones(s.prompt.shape, np.float32); trust[:5] = 0.1
    t, _, ti = model_input(s, "tmask", tau=0.05, trust=trust, threshold=0.5)
    assert ti["removed"] == int((s.prompt_valid[:5]).sum()) and np.all(t[:5] == 0)
    s2 = same_grid_sample(); s2.gt = s2.gt * 3        # changing GT must not change deployable inputs
    for variant in ("raw", "tmask", "tonesided", "tsoft"):
        a = model_input(s, variant, tau=0.05, trust=trust, threshold=0.5)
        b = model_input(s2, variant, tau=0.05, trust=trust, threshold=0.5)
        assert np.array_equal(a[0], b[0]) and (a[1] is None or a[1].sha256() == b[1].sha256())
    o, _, oi = model_input(s, "oracle_mask", tau=0.05)
    assert oi["removed"] > 0 and o.dtype == np.float32 and np.all(o >= 0)
    assert np.all(o[(raw > 0) & (o == 0)] == 0) and oi["kept"] + oi["removed"] == info["kept"]


def test_guided_variants_keep_the_prompt_and_move_flagged_pixels_to_the_hinge():
    s = same_grid_sample()
    valid = s.prompt_valid
    trust = np.ones(s.prompt.shape, np.float32); trust[:5] = 0.1
    raw = model_input(s, "raw", tau=0.05)[0]
    d, g, info = model_input(s, "tonesided", tau=0.05, trust=trust, threshold=0.5)
    flag = valid.copy(); flag[5:] = False
    assert np.array_equal(d, raw) and info["removed"] == 0 and info["flagged"] == int(flag.sum())
    assert np.array_equal(g.hinge > 0, flag) and np.array_equal(g.weight > 0, valid & ~flag)
    assert np.array_equal(g.anchor, valid & ~flag) and info["kept"] == int(g.anchor.sum())
    g.check(valid)
    d, g, _ = model_input(s, "tsoft", tau=0.05, trust=trust, threshold=0.5)
    assert np.allclose(g.weight + g.hinge, valid.astype(np.float32)) and np.array_equal(g.anchor, valid & ~flag)
    d, g, info = model_input(s, "oracle_onesided", tau=0.05)
    c = model_input(s, "oracle_mask", tau=0.05)[2]
    assert np.array_equal(d, raw) and info["flagged"] == c["removed"] and int((g.hinge > 0).sum()) == c["removed"]


def test_guided_variant_needs_a_guidance_model():
    from scripts.cache_completion import run
    with pytest.raises(ValueError, match="no guidance objective"):
        run("priorda", "nyu", "tonesided", output="unused")


def test_per_group_subsample_is_even_and_shared():
    from tp.data.select import per_group_subsample
    recs = [dict(dataset="hammer", scene=sc, sensor=se, frame=f"traj1_1_{i:06d}", split="val")
            for sc in ("scene2", "scene3") for se in ("d435", "tof") for i in range(10)]
    recs += [dict(dataset="hammer", scene="scene4", sensor="d435", frame="traj1_1_000000", split="val")]
    got = per_group_subsample(recs, 3)
    assert len(got) == 4 * 3 + 1
    assert [r["frame"][-2:] for r in got[:3]] == ["00", "04", "09"]      # linspace 0, 4.5, 9; numpy rounds half to even
    assert per_group_subsample(list(reversed(recs)), 3) == list(reversed(got))      # order-independent choice
    assert per_group_subsample(recs, None) == recs


def test_select_records_holdout_and_test_lock():
    recs = [dict(dataset="hammer", scene=s, split=sp, frame="f") for s, sp in
            (("scene10", "train"), ("scene3", "train"), ("scene11", "val"), ("scene12", "test"))]
    got = select_records(recs, {"train", "val"}, "holdout")
    assert [r["scene"] for r in got] == ["scene10", "scene11"]
    assert len(select_records(recs, {"train", "val"}, "all")) == 3
    with pytest.raises(ValueError):
        select_records(recs, {"train", "test"}, "holdout")


def test_marigold_gate_needs_all_three_conditions():
    import yaml
    from scripts.summarize_guidance import CONFIG, gate_verdict
    g = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["gate"]
    assert gate_verdict(0.08, 0.10, 0.09, 0.10, False, g)["passes"]                  # 20% gain, holdout better
    assert not gate_verdict(0.095, 0.10, 0.09, 0.10, False, g)["passes"]             # 5% gain < 10%
    assert not gate_verdict(0.08, 0.10, 0.11, 0.10, False, g)["passes"]              # holdout worse
    assert not gate_verdict(0.08, 0.10, 0.09, 0.10, True, g)["passes"]               # clean clearly worse
    assert not gate_verdict(None, 0.10, 0.09, 0.10, False, g)["passes"]              # missing run fails
    assert not gate_verdict(0.08, 0.10, 0.09, 0.10, None, g)["passes"]               # clean check NA fails
