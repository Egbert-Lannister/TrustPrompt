"""Sensor-only baseline: native-grid nearest fill, then nearest resize."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tp.geom.camera import nearest_fill, resize_nearest

if TYPE_CHECKING:
    from tp.data.schema import Sample


def predict(sample: Sample) -> np.ndarray:
    """Return float32 metric Z at RGB resolution without reading ground truth.

    Hole filling occurs before resizing, per the implementation plan. The
    original prompt and validity are unchanged, so evaluation can construct
    method-independent regions from the original unfilled sensor evidence.
    Entirely empty prompts result in all-NaN invalid predictions.
    """
    filled = nearest_fill(sample.prompt, sample.prompt_valid)
    return resize_nearest(filled, sample.rgb.shape[:2]).astype(np.float32, copy=False)
