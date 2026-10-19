import pytest

torch = pytest.importorskip("torch")


from tp.omnidc.gates import cell_keep, gated  # noqa: E402


def test_gates_patch_one_instance_and_restore_it():
    class Backbone(torch.nn.Module):
        def forward(self, x):
            return (None, None, None, None, torch.ones(1, 1, 2, 2), None)

    class Spn(torch.nn.Module):
        def forward(self, feat, guide, sp_dep, confidence):
            return sp_dep

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone, self.prop_layer = Backbone(), Spn()

    net, other = Net(), Net()
    flag = torch.zeros(1, 1, 8, 8, dtype=torch.bool); flag[0, 0, 0, 0] = True
    keep = cell_keep(flag, 4)
    assert keep.tolist() == [[[[0.0, 1.0], [1.0, 1.0]]]]
    dep = torch.ones(1, 1, 8, 8)
    with gated(net, ddi_keep=keep, spn_keep=1.0 - flag.float()):
        assert net.backbone(None)[4].tolist() == keep.tolist()
        assert net.prop_layer(None, None, dep, None)[0, 0, 0, 0] == 0 and net.prop_layer(None, None, dep, None).sum() == 63
        assert other.backbone(None)[4].sum() == 4                       # other instances untouched
    assert net.backbone(None)[4].sum() == 4 and net.prop_layer(None, None, dep, None).sum() == 64
    assert "forward" not in vars(net.backbone) and "forward" not in vars(net.prop_layer)
