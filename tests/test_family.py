import numpy as np
import pytest

from tp.eval.family import circular_block_bootstrap, compare, frame_parts, holm_family


def rows(values, dataset="hammer", scene="scene11", sensors=("d435", "l515"), split="val", empty=()):
    out = []
    for i, v in enumerate(values):
        for s in sensors:
            frame = f"traj{1 + i % 2}_1_{i:06d}"
            st = "empty" if i in empty else "ok"
            out.append(dict(dataset=dataset, sensor=s, scene=scene if not isinstance(scene, list) else scene[i], frame=frame,
                            split=split, metrics=dict(regions=dict(C=dict(status=st, count=0 if st == "empty" else 10,
                                                                         absrel=None if st == "empty" else v)))))
    return out


def test_frame_parts_hammer_and_nyu():
    assert frame_parts("hammer", "scene11", "traj2_1_000040") == ("scene11/traj2_1", 40)
    assert frame_parts("mirror3d_nyu", "bathroom_0002", "43") == ("bathroom_0002/-", 43)


def test_block_bootstrap_null_and_shift():
    rng = np.random.default_rng(1)
    base = {("t", i): [float(x)] for i, x in enumerate(rng.normal(0, 1e-3, 60))}
    r = circular_block_bootstrap(base, block_length=8, n_resamples=2000, seed=0)
    assert r["p_value"] > 0.05
    shifted = {k: [v[0] - 0.01] for k, v in base.items()}
    r2 = circular_block_bootstrap(shifted, block_length=8, n_resamples=2000, seed=0)
    assert r2["p_value"] < 0.01 and r2["ci_high"] < 0


def test_compare_single_scene_uses_frame_blocks_and_excludes_empty():
    ref = rows([0.05] * 40, empty=(3,))
    cand = rows([0.04] * 40, empty=(3,))
    r = compare(ref, cand, dataset="hammer", n_resamples=500, seed=0)
    assert r["unit"] == "frame_block" and r["excluded_empty_images"] == 2 and r["images"] == 78
    assert r["mean_difference"] == pytest.approx(-0.01) and r["reference_mean"] == pytest.approx(0.05)


def test_compare_multi_scene_uses_scene_bootstrap_and_na_on_invalid():
    scenes = [f"s{i % 6}" for i in range(24)]
    ref = rows([0.1] * 24, dataset="mirror3d_nyu", scene=scenes, sensors=("kinect1",))
    cand = rows([0.08] * 24, dataset="mirror3d_nyu", scene=scenes, sensors=("kinect1",))
    r = compare(ref, cand, dataset="mirror3d_nyu", n_resamples=500)
    assert r["unit"] == "scene" and r["scene_count"] == 6 and r["mean_difference"] == pytest.approx(-0.02)
    cand[0]["metrics"]["regions"]["C"]["absrel"] = None
    assert compare(ref, cand, dataset="mirror3d_nyu")["status"] == "NA_invalid_or_missing"


def test_empty_region_must_be_method_independent():
    with pytest.raises(ValueError):
        compare(rows([0.05] * 10, empty=(1,)), rows([0.05] * 10), dataset="hammer")


def test_holm_family_keeps_na_with_p_one():
    res = {"a": dict(p_value=0.001), "b": dict(p_value=0.04), "c": dict(p_value=None, status="NA_missing_run")}
    holm_family(res, alpha=0.05)
    assert res["a"]["p_holm"] == pytest.approx(0.003) and res["a"]["reject_at_alpha"]
    assert res["b"]["p_holm"] == pytest.approx(0.08) and not res["b"]["reject_at_alpha"]
    assert res["c"]["p_holm"] == 1.0 and res["c"]["holm_note"]
