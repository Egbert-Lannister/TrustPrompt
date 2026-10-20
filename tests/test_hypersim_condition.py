import numpy as np

from tp.data.hypersim import MIRROR, WINDOW, Frame
from tp.sim.hypersim_condition import condition


def room(h=96, w=128):
    """Camera looking at a back wall 2 m away, floor below; a mirror on the left of the wall, a window on the right."""
    K = np.array([[100.0, 0, (w - 1) / 2], [0, 100.0, (h - 1) / 2], [0, 0, 1]])
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    ry = (y - K[1, 2]) / K[1, 1]
    depth = np.full((h, w), 2.0)
    floor = ry > 0.375                                    # floor plane Y = 0.75 m below the camera
    depth[floor] = np.minimum(2.0, 0.75 / ry[floor])
    sem = np.ones((h, w), np.int16)
    sem[20:55, 10:50] = MIRROR
    sem[20:55, 80:118] = WINDOW
    rgb = np.full((h, w, 3), 128, np.uint8)
    return Frame("s", "cam_00", 0, rgb, depth.astype(np.float32), np.ones((h, w), bool), sem, K, np.eye(3))


def test_mirror_and_window_readings_are_behind_the_surface():
    f = room()
    out = condition(f, np.random.default_rng(0), sensor="structured_light", p_backwall=1.0)
    lab, sem = out["labels"], f.semantic
    mirror_far = (lab[sem == MIRROR] == 1).mean()             # reflected room within the 8 m range
    window = lab[sem == WINDOW]
    assert mirror_far > 0.9                                # every reading on the mirror lies behind its plane
    assert np.isin(window, (0, 1, 3)).all() and (window != 0).mean() > 0.25   # seen through (far / beyond range);
    # partial transmission (penetration's mixed mode) keeps some glass-surface readings
    assert (lab[(sem == 1) & (f.depth == 2.0)] == 0).mean() > 0.9     # the rest of the wall stays clean
    types = [r.get("type") for r in out["info"]["regions"]]
    assert types == ["reflection", "window"]


def test_same_seed_same_result():
    f = room()
    a, b = condition(f, np.random.default_rng(5)), condition(f, np.random.default_rng(5))
    assert np.array_equal(a["P_corrupt"], b["P_corrupt"]) and a["info"]["sensor"] == b["info"]["sensor"]


def test_without_back_wall_unresolved_mirror_pixels_are_holes():
    f = room()
    out = condition(f, np.random.default_rng(0), sensor="structured_light", p_backwall=0.0)
    lab = out["labels"][f.semantic == MIRROR]
    assert np.isin(lab, (1, 3)).all() and (lab == 3).mean() > 0.5
