import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.p1_trust_floor import GRID, crc_max_lambda, false_distrust_curve, floor_fn  # noqa: E402


def test_floor_keeps_only_low_trust():
    c = torch.tensor([0.05, 0.3, 0.6, 0.95])
    assert torch.allclose(floor_fn(0.4)(c), torch.tensor([0.05, 0.3, 1.0, 1.0]))
    assert torch.all(floor_fn(0.0)(c) == 1)


def test_false_distrust_curve_and_crc():
    gt = np.full((2, 4), 2.0)
    prompt = gt.copy(); prompt[0, 0] = 3.0                      # one wrong reading
    s = types.SimpleNamespace(prompt=prompt, gt=gt, prompt_valid=np.ones((2, 4), bool), gt_valid=np.ones((2, 4), bool))
    conf = np.array([[0.1, 0.2, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]])
    curve, n = false_distrust_curve(conf, s)
    assert n == 7 and curve[0] == 0                              # lambda 0 distrusts nothing
    assert curve[list(GRID).index(0.5)] == pytest.approx(1 / 7)    # the correct reading at c = 0.2 is distrusted
    assert np.all(np.diff(curve) >= 0)
    losses = np.tile(curve, (99, 1))
    lam, bound = crc_max_lambda(losses, 0.2)
    assert lam == pytest.approx(0.9) and bound <= 0.2            # 1/7 + slack fits below 0.9, not at it
    assert crc_max_lambda(np.ones((5, len(GRID))), 0.1)[0] == 0.0
