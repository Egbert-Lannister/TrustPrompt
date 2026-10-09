"""Meaningful checks on RGB-only access and axial-Z output qualification."""
import json
from pathlib import Path
import numpy as np
import pytest
from baselines.unidepth import normalize_prediction, UniDepthPredictor
from scripts.cache_unidepth import load_rgb_training_npz


def test_raw_npz_gate_does_not_decode_ground_truth(tmp_path):
    path = tmp_path / "rgb_only.npz"
    meta = {"dataset": "mirror3d_nyu", "sensor": "kinect1", "scene": "scene_a", "frame": "1",
            "split": "train", "metadata": {"synthetic": False}}
    rgb = np.full((2, 3, 3), 127, dtype=np.uint8)
    # allow_pickle=False would fail if the prediction driver attempted to decode GT.
    np.savez(path, rgb=rgb, metadata_json=np.asarray(json.dumps(meta)), gt=np.array([object()], dtype=object))
    decoded, ids = load_rgb_training_npz(path)
    np.testing.assert_array_equal(decoded, rgb)
    assert ids["split"] == "train"
    meta["split"] = "test"
    np.savez(path, rgb=rgb, metadata_json=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError, match="training"):
        load_rgb_training_npz(path)


def test_model_boundary_receives_only_uint8_rgb():
    torch = pytest.importorskip("torch")
    class Model:
        def infer(self, *args, **kwargs):
            assert len(args) == 1 and kwargs == {}
            tensor = args[0]
            assert tensor.shape == (1, 3, 2, 3) and tensor.dtype == torch.uint8
            points = torch.ones(1, 3, 2, 3)
            return {"depth": points[:, 2:3], "points": points,
                    "intrinsics": torch.eye(3)[None], "confidence": torch.ones(1, 1, 2, 3)}
    predictor = UniDepthPredictor.__new__(UniDepthPredictor)
    predictor.device = torch.device("cpu")
    predictor.model = Model()
    predictor.last_encoder_shape = None
    arrays, report = predictor.predict_rgb(np.full((2, 3, 3), 127, np.uint8))
    assert arrays["depth"].shape == (2, 3)
    assert report["points_z_exactly_equal_depth"]


def test_range_depth_cannot_masquerade_as_axial_z_and_invalids_remain():
    points = np.ones((1, 3, 2, 2), np.float32)
    outputs = {"depth": np.linalg.norm(points, axis=1, keepdims=True), "points": points,
               "intrinsics": np.eye(3, dtype=np.float32)[None], "confidence": np.ones((1, 1, 2, 2), np.float32)}
    with pytest.raises(ValueError, match="point-cloud Z"):
        normalize_prediction(outputs, (2, 2))
    points[0, 2, 0, 0] = np.nan
    outputs["depth"] = points[:, 2:3]
    arrays, report = normalize_prediction(outputs, (2, 2))
    assert np.isnan(arrays["depth"][0, 0])
    assert report["invalid_depth_count"] == 1


def test_dev_mode_admits_val_but_never_test(tmp_path):
    path = tmp_path / "val.npz"
    meta = {"dataset": "mirror3d_nyu", "sensor": "kinect1", "scene": "s", "frame": "2",
            "split": "val", "metadata": {"synthetic": False}}
    np.savez(path, rgb=np.zeros((2, 2, 3), np.uint8), metadata_json=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError):
        load_rgb_training_npz(path)
    _, ids = load_rgb_training_npz(path, frozenset({"train", "val"}))
    assert ids["split"] == "val"
    meta["split"] = "test"
    np.savez(path, rgb=np.zeros((2, 2, 3), np.uint8), metadata_json=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError, match="test refused"):
        load_rgb_training_npz(path, frozenset({"train", "val"}))
