import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tp.omnidc.gate import Gate, gate_inputs, gate_targets, metric_prior  # noqa: E402
from tp.omnidc.gates import extend_depth_input, soft_gated  # noqa: E402
from tp.trust.model import make_inputs  # noqa: E402


def test_metric_prior_recovers_the_affine_disparity_despite_outliers():
    g = torch.Generator().manual_seed(0)
    z = torch.rand(1, 1, 32, 40, generator=g) * 4 + 1
    rel = (1 / z - 0.1) / 0.5                               # disparity = 0.5 * rel + 0.1
    prompt = z.clone()
    bad = torch.rand(z.shape, generator=g) < 0.2
    prompt[bad] = prompt[bad] * 2.5                        # 20 % readings far behind (mirror-like)
    prompt[torch.rand(z.shape, generator=g) < 0.1] = 0     # holes
    prior = metric_prior(rel, prompt)
    assert torch.allclose(prior, z, rtol=1e-3)


def test_gate_inputs_match_T_inputs():
    rng = np.random.default_rng(0)
    prompt = rng.uniform(0.5, 5, (24, 32)).astype(np.float32); prompt[rng.random((24, 32)) < 0.2] = 0
    prior = rng.uniform(0.5, 5, (24, 32)).astype(np.float32); prior[rng.random((24, 32)) < 0.1] = np.nan
    rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    ref = make_inputs(prompt, prompt > 0, prior, rgb)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
    got = gate_inputs(t(prompt)[None], t(prior)[None], t(rgb.transpose(2, 0, 1).astype(np.float32) / 255))
    assert np.allclose(got[0].numpy(), ref, atol=1e-5)


def test_gate_shapes_and_targets():
    logits = Gate(width=8)(torch.rand(2, 1, 32, 48) + 0.5, torch.rand(2, 1, 32, 48) + 0.5, torch.rand(2, 3, 32, 48))
    assert logits.shape == (2, 3, 32, 48)
    lab = torch.tensor([[0, 1, 2, 3, 255]], dtype=torch.uint8)
    assert gate_targets(lab, torch.tensor([[True, True, True, False, True]])).tolist() == [[0, 1, 2, -100, -100]]


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_dep = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1, bias=False))

    def forward(self, rgb, depth, pattern):
        f = self.conv1_dep(depth)
        return None, None, None, None, torch.ones(1, 1, depth.shape[2] // 4, depth.shape[3] // 4), f


class Spn(torch.nn.Module):
    def forward(self, feat, guide, sp_dep, confidence):
        return torch.sigmoid(confidence)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.backbone, self.prop_layer = Backbone(), Spn()


def test_soft_gate_on_all_three_paths_and_open_gate_changes_nothing():
    torch.manual_seed(0)
    net = Net()
    depth = torch.randn(1, 2, 8, 8)
    before = net.backbone(None, depth, None)[5]
    extend_depth_input(net.backbone)
    g = torch.ones(1, 1, 8, 8)
    with soft_gated(net, g, 4):
        assert torch.equal(net.backbone(None, depth, None)[5], before)        # zero-initialised gate channel
    g = torch.rand(1, 1, 8, 8); g[0, 0, 0, 0] = 0.01
    conf = torch.randn(1, 1, 8, 8)
    with soft_gated(net, g, 4):
        out = net.backbone(None, depth, None)
        assert out[4][0, 0, 0, 0] == pytest.approx(g[0, 0, :4, :4].min().item())   # DDI: least trusted pixel of the cell
        assert torch.allclose(net.prop_layer(None, None, None, conf), g * torch.sigmoid(conf), atol=1e-6)
    with torch.no_grad():
        net.backbone.conv1_dep[0].weight[:, 2] = 1.0
    with soft_gated(net, g, 4, paths=("input",)):
        expect = torch.nn.functional.conv2d(torch.cat([depth[:, :1] * g, depth[:, 1:], g], 1),
                                            net.backbone.conv1_dep[0].weight, padding=1)
        assert torch.allclose(net.backbone(None, depth, None)[5], expect, atol=1e-6)
        assert torch.equal(net.prop_layer(None, None, None, conf), torch.sigmoid(conf))  # SPN not gated here


def test_untrained_gate_starts_open():
    torch.manual_seed(0)
    logits = Gate()(torch.rand(1, 1, 32, 48) + 0.5, torch.rand(1, 1, 32, 48) + 0.5, torch.rand(1, 3, 32, 48))
    assert torch.softmax(logits, 1)[:, 0].mean() > 0.9
