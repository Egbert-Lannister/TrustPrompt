"""Local synthetic cache integrity tests, with no model/data dependencies."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from baselines.cached_prior import CachedPrior
from scripts.make_synthetic import make_sample
from tp.data.schema import save_sample
from tp.eval.metrics import evaluate_depth
from tp.eval.regions import regions
from tp.provenance import sha256, write_json


def _cache(tmp_path, *, depth=None):
    sample = make_sample()
    sample_path = tmp_path / "sample.npz"
    save_sample(sample, sample_path)
    source_hash = sha256(sample_path)
    root = tmp_path / "cache"
    (root / "predictions").mkdir(parents=True)
    path = root / "predictions" / "0.npz"
    if depth is None:
        depth = np.full(sample.rgb.shape[:2], 2.0, dtype=np.float32)
    np.savez_compressed(path, depth=depth)
    row = {key: getattr(sample, key) for key in ("dataset", "sensor", "scene", "frame", "split")}
    row.update(sample_npz_sha256=source_hash,
               rgb_sha256=hashlib.sha256(sample.rgb.tobytes()).hexdigest(),
               rgb_shape=list(sample.rgb.shape),
               prediction_path="Q:/old_machine/cache/predictions/0.npz",
               prediction_sha256=sha256(path),
               seconds=0.25,
               validation=dict(depth_unit="m", depth_convention="axial_z", points_z_exactly_equal_depth=True))
    write_json(root / "images.json", [row])
    receipt = {
        "schema": "trustprompt.run.v1", "stage": "real_train_smoke_verified", "frame_count": 1,
        "model_version": {"name": "UniDepthV2-ViT-S14", "source_hashes": {"example.py": "a" * 64},
                          "weight_hashes": {"model.safetensors": "b" * 64}},
        "config": {"input": "rgb_only", "camera_input": "none", "split": "train", "resolution_level": 0},
        "all_points_z_equal_depth": True,
        "inputs": [{key: row[key] for key in ("dataset", "sensor", "scene", "frame", "split",
                                               "rgb_sha256", "sample_npz_sha256")}],
        "images_sha256": sha256(root / "images.json"),
    }
    write_json(root / "receipt.json", receipt)
    return root, sample, source_hash, row, receipt


def _write_index(root, rows, receipt):
    write_json(root / "images.json", rows)
    receipt = deepcopy(receipt)
    receipt["images_sha256"] = sha256(root / "images.json")
    write_json(root / "receipt.json", receipt)


def test_valid_cache_is_portable_and_does_not_require_original_prediction_path(tmp_path):
    root, sample, source_hash, _, _ = _cache(tmp_path)
    prediction = CachedPrior(root).predict(sample, source_hash)
    np.testing.assert_array_equal(prediction, np.full(sample.rgb.shape[:2], 2, np.float32))


@pytest.mark.parametrize("stage", ["started", "failed", "small_scale_validation_passed"])
def test_unverified_or_incomplete_cache_is_rejected(tmp_path, stage):
    root, _, _, _, receipt = _cache(tmp_path)
    receipt["stage"] = stage
    write_json(root / "receipt.json", receipt)
    with pytest.raises(ValueError, match="verified|complete|stage"):
        CachedPrior(root)


def test_duplicate_sample_sensor_identity_rejected(tmp_path):
    root, _, _, row, receipt = _cache(tmp_path)
    _write_index(root, [row, row], receipt)
    with pytest.raises(ValueError, match="duplicate"):
        CachedPrior(root)


def test_stale_sample_hash_is_rejected_even_when_rgb_is_unchanged(tmp_path):
    root, sample, source_hash, _, _ = _cache(tmp_path)
    assert CachedPrior(root).predict(sample, source_hash).shape == sample.gt.shape
    with pytest.raises(ValueError, match="sample hash"):
        CachedPrior(root).predict(sample, "f" * 64)


def test_rgb_mutation_cannot_reuse_old_preprocessed_cache(tmp_path):
    root, sample, source_hash, _, _ = _cache(tmp_path)
    sample.rgb[0, 0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="RGB preprocessing"):
        CachedPrior(root).predict(sample, source_hash)


def test_split_mismatch_is_rejected(tmp_path):
    root, sample, source_hash, _, _ = _cache(tmp_path)
    sample.split = "test"
    with pytest.raises(ValueError, match="split"):
        CachedPrior(root).predict(sample, source_hash)


@pytest.mark.parametrize("field", ["dataset", "sensor", "scene", "frame"])
def test_other_source_identity_never_falls_back_to_same_frame(tmp_path, field):
    root, sample, source_hash, _, _ = _cache(tmp_path)
    setattr(sample, field, "other")
    with pytest.raises((KeyError, ValueError)):
        CachedPrior(root).predict(sample, source_hash)


def test_prediction_corruption_detected_before_numpy_decode(tmp_path):
    root, sample, source_hash, _, _ = _cache(tmp_path)
    path = root / "predictions" / "0.npz"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="corrupted prior prediction"):
        CachedPrior(root).predict(sample, source_hash)


def test_wrong_prediction_grid_is_rejected_even_with_matching_payload_hash(tmp_path):
    root, sample, source_hash, _, _ = _cache(tmp_path, depth=np.ones((4, 4), np.float32))
    with pytest.raises(ValueError, match="grid mismatch"):
        CachedPrior(root).predict(sample, source_hash)


def test_invalid_cached_depths_survive_to_fixed_denominator_metrics(tmp_path):
    depth = np.ones((96, 128), dtype=np.float32)
    depth[0, :4] = [np.nan, np.inf, 0.0, -1.0]
    root, sample, source_hash, _, _ = _cache(tmp_path, depth=depth)
    prediction = CachedPrior(root).predict(sample, source_hash)
    np.testing.assert_array_equal(prediction, depth)
    gt = np.ones_like(prediction)
    valid = np.ones(gt.shape, dtype=bool)
    result = evaluate_depth(prediction, gt, gt, regions(gt, valid, gt, valid, band=0))
    metrics = result["regions"]["all"]
    assert metrics["count"] == gt.size
    assert metrics["invalid_count"] == 4
    assert metrics["status"] == "invalid_predictions"
    assert metrics["absrel"] is metrics["rmse"] is metrics["log_rmse"] is None
    assert metrics["delta_1_25"] == pytest.approx((gt.size - 4) / gt.size)
    json.dumps(result, allow_nan=False)


def test_wrong_model_receipt_cannot_be_reported_as_unidepth_v2_small(tmp_path):
    root, _, _, _, receipt = _cache(tmp_path)
    receipt["model_version"]["name"] = "UniDepthV2-ViT-L14"
    write_json(root / "receipt.json", receipt)
    with pytest.raises(ValueError):
        CachedPrior(root)


def test_index_bytes_are_bound_to_new_receipts(tmp_path):
    root, _, _, _, _ = _cache(tmp_path)
    with (root / "images.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError):
        CachedPrior(root)


@pytest.mark.parametrize("field,value", [
    ("sample_npz_sha256", "e" * 64), ("rgb_sha256", "d" * 64),
    ("split", "test"), ("frame", "other"), ("scene", "other"),
    ("dataset", "other"), ("sensor", "other"),
])
def test_index_row_cannot_disagree_with_receipt_even_if_index_hash_updated(tmp_path, field, value):
    root, _, _, row, receipt = _cache(tmp_path)
    row[field] = value
    _write_index(root, [row], receipt)
    with pytest.raises(ValueError):
        CachedPrior(root)


def test_incomplete_index_cannot_inherit_complete_receipt(tmp_path):
    root, _, _, _, receipt = _cache(tmp_path)
    receipt["frame_count"] = 2
    write_json(root / "receipt.json", receipt)
    with pytest.raises(ValueError):
        CachedPrior(root)


def test_legacy_receipt_without_index_digest_still_verifies_bound_source(tmp_path):
    root, sample, source_hash, _, receipt = _cache(tmp_path)
    del receipt["images_sha256"]
    write_json(root / "receipt.json", receipt)
    np.testing.assert_array_equal(CachedPrior(root).predict(sample, source_hash),
                                  np.full(sample.rgb.shape[:2], 2, np.float32))


def test_windows_prediction_path_basename_resolves_on_any_host():
    from pathlib import PureWindowsPath
    assert PureWindowsPath(r"D:\x\cache\predictions\45.npz").name == "45.npz"
    assert PureWindowsPath("/sessions/x/predictions/45.npz").name == "45.npz"


def test_fixed_point_prediction_decodes_within_0p1mm_and_keeps_invalid(tmp_path):
    import numpy as np
    depth = np.array([[0.3501234, 1.23456], [np.nan, 13.0]], np.float32)
    finite = np.isfinite(depth) & (depth > 0)
    q = np.zeros(depth.shape, np.uint16); q[finite] = np.round(depth[finite] / 2e-4).astype(np.uint16)
    np.savez_compressed(tmp_path / "p.npz", depth_q=q, scale_m=np.float64(2e-4))
    with np.load(tmp_path / "p.npz") as z:
        d = (z["depth_q"].astype(np.float64) * float(z["scale_m"])).astype(np.float32); d[z["depth_q"] == 0] = np.nan
    assert np.isnan(d[1, 0]) and np.nanmax(np.abs(d - depth)) <= 1e-4 + 1e-7
