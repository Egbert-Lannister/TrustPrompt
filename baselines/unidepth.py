"""Local RGB-only UniDepthV2 native inference. Never imports GLARE.

Import boundary selectively adapted from GLARE's reviewed
experiments/generate_i3_transphy3d_unidepth_cache.py (see run provenance).
UniDepth itself remains a configurable, read-only third-party dependency.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


@contextmanager
def _wandb_import_stub():
    """Upstream visualization imports need names, not an active logger."""
    previous = sys.modules.get("wandb")
    stub = ModuleType("wandb")
    stub.Image = lambda value: value
    stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous


def _within_source(module, source: Path) -> bool:
    paths = [getattr(module, "__file__", None)]
    if paths[0] is None:
        paths = list(getattr(module, "__path__", []))
    return bool(paths) and all(p is not None and Path(p).resolve().is_relative_to(source) for p in paths)


def import_local_unidepth(source: Path):
    # Persistent for this inference process: later lazy imports must also be read-only.
    sys.dont_write_bytecode = True
    source = source.resolve()
    if not (source / "unidepth/models/unidepthv2/unidepthv2.py").is_file():
        raise FileNotFoundError("Expected a local UniDepth checkout")
    existing = sys.modules.get("unidepth")
    if existing is not None and not _within_source(existing, source):
        raise RuntimeError("Ambient UniDepth differs from the configured source")
    sys.path.insert(0, str(source))
    try:
        with _wandb_import_stub():
            models = importlib.import_module("unidepth.models")
    finally:
        sys.path.remove(str(source))
    if not _within_source(sys.modules["unidepth"], source) or not _within_source(models, source):
        raise RuntimeError("UniDepth import escaped the configured checkout")
    return models.UniDepthV2


def normalize_prediction(outputs: dict, shape: tuple[int, int]) -> tuple[dict, dict]:
    """Retain invalid values and verify the returned depth really is axial Z."""
    def array(key):
        value = outputs[key]
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)
    depth = array("depth")
    points = array("points")
    intrinsics = array("intrinsics")
    confidence = array("confidence")
    if depth.shape != (1, 1, *shape) or points.shape != (1, 3, *shape):
        raise ValueError("Native inference must return depth and points on the input RGB grid")
    if intrinsics.shape != (1, 3, 3) or confidence.shape != depth.shape:
        raise ValueError("Unexpected UniDepth camera/confidence layout")
    z = points[:, 2:3]
    if not np.array_equal(depth, z, equal_nan=True):
        raise ValueError("UniDepth depth differs from returned point-cloud Z")
    valid = np.isfinite(depth) & (depth > 0)
    report = {"shape": list(shape), "depth_unit": "m", "depth_convention": "axial_z",
              "points_z_exactly_equal_depth": True, "invalid_depth_count": int((~valid).sum()),
              "nonfinite_points_count": int((~np.isfinite(points)).sum()),
              "depth_min_positive_m": float(depth[valid].min()) if valid.any() else None,
              "depth_max_positive_m": float(depth[valid].max()) if valid.any() else None}
    return {"depth": depth[0, 0], "points": points[0], "intrinsics": intrinsics[0],
            "confidence": confidence[0, 0]}, report


class UniDepthPredictor:
    def __init__(self, source: str | Path, weights: str | Path, device="cuda", resolution_level=0):
        import torch
        if isinstance(resolution_level, bool) or not isinstance(resolution_level, int) or not 0 <= resolution_level < 10:
            raise ValueError("resolution_level must be an integer in [0, 10)")
        weights = Path(weights).resolve()
        if not (weights / "model.safetensors").is_file() or not (weights / "config.json").is_file():
            raise FileNotFoundError("Local model.safetensors and config.json are required")
        model_class = import_local_unidepth(Path(source))
        self.device = torch.device(device)
        self.model = model_class.from_pretrained(str(weights), local_files_only=True).to(self.device).eval()
        self.model.resolution_level = resolution_level
        self.last_encoder_shape = None
        self.model.pixel_encoder.register_forward_pre_hook(self._record_encoder_shape)

    def _record_encoder_shape(self, module, inputs):
        self.last_encoder_shape = list(inputs[0].shape)

    def predict_rgb(self, rgb: np.ndarray) -> tuple[dict, dict]:
        """Only RGB crosses the model boundary; no K/prompt/GT/material argument."""
        import torch
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("RGB must be uint8 [H,W,3]")
        if min(rgb.shape[:2]) < 1:
            raise ValueError("Empty RGB")
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model.infer(tensor)
        arrays, report = normalize_prediction(outputs, rgb.shape[:2])
        report["encoder_input_shape"] = self.last_encoder_shape
        return arrays, report
