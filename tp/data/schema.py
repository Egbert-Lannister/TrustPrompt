"""Versioned, pickle-free sample contract. All depths are metric camera-axis Z.

Invalid raw prompt values are preserved; validity masks must mark every
non-finite/nonpositive/out-of-range value invalid. No implicit filling occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import numpy as np

SCHEMA = "trustprompt.sample.v1"


@dataclass
class Sample:
    rgb: np.ndarray
    prompt: np.ndarray
    prompt_valid: np.ndarray
    gt: np.ndarray
    gt_valid: np.ndarray
    material: np.ndarray
    K_rgb: np.ndarray
    K_prompt: np.ndarray
    sensor: str
    dataset: str
    scene: str
    frame: str
    split: str
    prompt_conf: np.ndarray | None = None
    behind: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> "Sample":
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 [H,W,3]")
        shape = self.rgb.shape[:2]
        if min(shape) < 1 or self.prompt.ndim != 2 or min(self.prompt.shape) < 1:
            raise ValueError("empty or non-2D depth")
        for name, expected in (("prompt", self.prompt.shape), ("gt", shape)):
            a = getattr(self, name)
            m = getattr(self, name + "_valid")
            if a.dtype != np.float32 or a.shape != expected:
                raise ValueError(f"{name} must be float32 {expected}")
            if m.dtype != np.bool_ or m.shape != expected:
                raise ValueError(f"{name}_valid must be bool {expected}")
            if np.any(m & (~np.isfinite(a) | (a <= 0))):
                raise ValueError(f"{name}_valid includes nonfinite/nonpositive depth")
        if self.material.dtype != np.uint8 or self.material.shape != shape or np.any(self.material > 4):
            raise ValueError("material must be uint8 [H,W], codes 0..4")
        for name in ("K_rgb", "K_prompt"):
            k = getattr(self, name)
            if k.shape != (3, 3) or k.dtype.kind != "f" or not np.isfinite(k).all():
                raise ValueError(f"{name} must be finite float 3x3")
            if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
                raise ValueError(f"{name} invalid pinhole intrinsics")
            if abs(np.linalg.det(k)) < 1e-12:
                raise ValueError(f"{name} singular intrinsics")
        for name in ("dataset", "sensor", "scene", "frame", "split"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v or any(x in v for x in ("/", "\\", "..")):
                raise ValueError(f"invalid identifier: {name}")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train/val/test")
        if self.prompt_conf is not None:
            if self.prompt_conf.shape != self.prompt.shape or self.prompt_conf.dtype.kind not in "uif":
                raise ValueError("prompt_conf must be numeric on prompt grid")
            if not np.isfinite(self.prompt_conf).all():
                raise ValueError("nonfinite prompt_conf")
        if self.behind is not None and (self.behind.dtype != np.float32 or self.behind.shape != shape):
            raise ValueError("behind must be float32 [H,W]")
        if self.metadata.get("depth_convention") != "axial_z" or self.metadata.get("depth_unit") != "m":
            raise ValueError("metadata must explicitly declare depth_unit=m, depth_convention=axial_z")
        if not isinstance(self.metadata.get("synthetic"), bool):
            raise ValueError("metadata.synthetic must explicitly be true or false")
        json.dumps(self.metadata, allow_nan=False)
        return self

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.sensor}/{self.scene}/{self.frame}"


def save_sample(sample: Sample, path: str | Path) -> None:
    sample.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f.name: getattr(sample, f.name) for f in fields(sample)
               if isinstance(getattr(sample, f.name), np.ndarray)}
    meta = {f.name: getattr(sample, f.name) for f in fields(sample)
            if isinstance(getattr(sample, f.name), str)}
    meta.update(schema=SCHEMA, metadata=sample.metadata)
    payload["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False, allow_nan=False))
    # Exclusive create: processed samples are immutable, never overwrite old work.
    with path.open("xb") as f:
        np.savez_compressed(f, **payload)


def load_sample(path: str | Path) -> Sample:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        if meta.pop("schema") != SCHEMA:
            raise ValueError("unsupported sample schema")
        values = {k: archive[k].copy() for k in archive.files if k != "metadata_json"}
    return Sample(**values, **meta).validate()


def validate_scene_splits(records: list[dict]) -> None:
    seen, keys = {}, set()
    for row in records:
        key = (row["dataset"], row["scene"])
        if key in seen and seen[key] != row["split"]:
            raise ValueError(f"scene leakage: {key}")
        seen[key] = row["split"]
        frame = (*key, row["frame"], row.get("sensor"))
        if frame in keys:
            raise ValueError(f"duplicate sample: {frame}")
        keys.add(frame)
