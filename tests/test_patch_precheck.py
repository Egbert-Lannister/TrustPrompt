import numpy as np
import pytest

from scripts.patch_precheck import SITES, key, necessity, restoration, run_sets, verdict


def test_run_sets():
    sets = run_sets()
    assert len(sets) == 22 and len(set(sets)) == 22
    assert key(frozenset()) == "none" and key(frozenset(SITES)) == "N+F4+F3+F2+F1"


def _x(scene, vals):
    return dict(scene=scene, runs={k: dict(C_sum=v, C_n=1) for k, v in vals.items()})


def test_restoration_and_necessity():
    vals = {"none": 1.0, key(SITES): 0.2, "F4": 0.4, key(frozenset(SITES) - {"F4"}): 0.7}
    sc = [[_x("a", vals)]]
    assert abs(restoration(sc, "F4") - 0.75) < 1e-9
    assert abs(necessity(sc, "F4") - 0.625) < 1e-9


def _res(te_lo, hits):
    r = {key(S): (0.1, 0.0, 0.2) for S in run_sets() if S}
    r.update(hits)
    return dict(TE=(0.1, te_lo, 0.2), restoration=r)


def test_verdict():
    good = {"F4": (0.8, 0.6, 0.9)}
    assert verdict(dict(nyu=_res(0.01, good), hammer=_res(0.01, good)))["outcome"] == "localised"
    assert verdict(dict(nyu=_res(0.01, good), hammer=_res(0.01, {"F3": (0.8, 0.6, 0.9)})))["outcome"] == "diffuse"
    assert verdict(dict(nyu=_res(0.01, good), hammer=_res(0.01, {"F4": (0.8, 0.4, 0.9)})))["outcome"] == "diffuse"
    assert verdict(dict(nyu=_res(-0.01, good), hammer=_res(0.01, good)))["outcome"].startswith("void")


def test_patched_forward_on_a_promptda_shaped_toy():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F
    from scripts.patch_precheck import patched_forward

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = nn.Conv2d(c, c, 3, padding=1)
            self.resConfUnit_depth = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

        def forward(self, *xs, prompt_depth=None, size=None):
            out = self.conv(xs[0]) + (xs[1] if len(xs) == 2 else 0)
            p = F.interpolate(prompt_depth, out.shape[2:], mode="bilinear", align_corners=False)
            out = out + self.resConfUnit_depth(p)
            return F.interpolate(out, size=size, mode="bilinear", align_corners=True) if size else out

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.enc = nn.Conv2d(3, 4, 3, padding=1)
            self.depth_head = nn.Module()
            self.depth_head.scratch = nn.Module()
            for k in (1, 2, 3, 4):
                setattr(self.depth_head.scratch, f"refinenet{k}", Block(4))
            self.out = nn.Conv2d(4, 1, 1)

        def normalize(self, pd):
            mn = torch.quantile(pd.reshape(1, -1), 0., dim=1, keepdim=True)[:, :, None, None]
            mx = torch.quantile(pd.reshape(1, -1), 1., dim=1, keepdim=True)[:, :, None, None]
            return (pd - mn) / (mx - mn), mn, mx

        def forward(self, x, prompt_depth=None):
            pd, mn, mx = self.normalize(prompt_depth)
            f = self.enc(x)
            s = self.depth_head.scratch
            p4 = s.refinenet4(F.avg_pool2d(f, 8), size=(4, 4), prompt_depth=pd)
            p3 = s.refinenet3(p4, F.avg_pool2d(f, 4), size=(8, 8), prompt_depth=pd)
            p2 = s.refinenet2(p3, F.avg_pool2d(f, 2), size=(16, 16), prompt_depth=pd)
            p1 = s.refinenet1(p2, f, prompt_depth=pd)
            return torch.sigmoid(self.out(p1)) * (mx - mn) + mn

        def predict(self, x, pd):
            with torch.no_grad():
                return self.forward(x, pd)

    m = Toy().eval()
    x = torch.rand(1, 3, 16, 16)
    raw = torch.rand(1, 1, 12, 16) * 3 + 1
    ora = raw.clone()
    ora[..., :4, :4] = 1.5
    ora[0, 0, 0, 0] = 0.5                                # changes the min too
    assert torch.equal(patched_forward(m, x, raw, ora, frozenset()), m.predict(x, raw))
    assert torch.allclose(patched_forward(m, x, raw, ora, frozenset(SITES)), m.predict(x, ora), atol=1e-6)
    one = patched_forward(m, x, raw, ora, frozenset({"F2"}))
    assert not torch.allclose(one, m.predict(x, raw)) and not torch.allclose(one, m.predict(x, ora))
    assert "normalize" not in m.__dict__               # instance override removed
    for k in (1, 2, 3, 4):
        assert not getattr(m.depth_head.scratch, f"refinenet{k}").resConfUnit_depth._forward_pre_hooks
