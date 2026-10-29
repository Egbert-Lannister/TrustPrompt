import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.p1_coupling import cell_masks, override_confidence  # noqa: E402


def test_cell_masks_pad_and_pool():
    valid = np.zeros((6, 7), bool); valid[0, 0] = valid[5, 6] = True
    wrong = np.zeros((6, 7), bool); wrong[5, 6] = True
    has, bad = cell_masks(valid, wrong, (8, 8))
    assert has.shape == (2, 2) and has[0, 0] and has[1, 1] and not has[0, 1]
    assert bad[1, 1] and not bad[0, 0]


def test_override_replaces_only_confidence_and_restores():
    class Backbone:
        def forward(self, *a):
            return ("a", "b", "c", "d", torch.full((1, 1, 2, 2), 0.3), "f")
    net = types.SimpleNamespace(backbone=Backbone())
    with override_confidence(net, lambda c: c * 0 + 1):
        out = net.backbone.forward()
        assert torch.all(out[4] == 1) and out[:4] == ("a", "b", "c", "d") and out[5] == "f"
    assert torch.allclose(net.backbone.forward()[4], torch.full((1, 1, 2, 2), 0.3))
