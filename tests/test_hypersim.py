import json

import numpy as np
import pytest

from tp.data.hypersim import axial_depth, pinhole
from tp.provenance import ROOT
from tp.sim.mirror_render import backproject, project

SEL = json.loads((ROOT / "configs/hypersim/selection_v1.json").read_text(encoding="utf-8"))
H, W = 48, 64                          # small grid, same normalised camera


@pytest.mark.parametrize("scene", ["ai_001_001", "ai_042_003", "ai_035_010", "ai_041_003"])
def test_axial_depth_gives_exact_points_and_pixels(scene):
    M = np.array(SEL["scenes"][scene]["camera"]["M_cam_from_uv"])
    K, R, kinv = pinhole(M, H, W)
    assert np.allclose(R @ R.T, np.eye(3)) and K[1, 0] == K[2, 0] == K[2, 1] == 0 and K[0, 0] > 0 and K[1, 1] > 0
    dist = np.random.default_rng(0).uniform(0.5, 8.0, (H, W))
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([x, y, np.ones_like(x)], -1) @ kinv.T
    points = (dist[..., None] * rays / np.linalg.norm(rays, axis=-1, keepdims=True)) @ R.T     # rotated frame
    z = axial_depth(dist, kinv, R)
    assert np.allclose(backproject(z, K), points, rtol=1e-5, atol=1e-5)
    u, v = project(points, K)
    assert np.allclose(u, x, atol=1e-6) and np.allclose(v, y, atol=1e-6)


def test_standard_camera_is_the_plain_pinhole():
    M = np.array(SEL["scenes"]["ai_001_001"]["camera"]["M_cam_from_uv"])
    K, R, _ = pinhole(M, 768, 1024)
    assert np.allclose(R, np.eye(3)) and abs(K[0, 1]) < 1e-9
    assert np.isclose(K[0, 0], 1024 / (2 * M[0, 0])) and np.isclose(K[0, 2], 511.5) and np.isclose(K[1, 2], 383.5)
