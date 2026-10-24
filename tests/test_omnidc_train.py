import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_hypersim_condition import room  # noqa: E402
from tp.omnidc.data import augment, make_sample, resize_frame, sparsify  # noqa: E402
from tp.omnidc.losses import confinput_original, confinput_two_sided  # noqa: E402


def test_resize_keeps_rays():
    f = room(96, 128)
    f.K[0, 1] = 1.5                                         # some skew
    r = resize_frame(f, 48, 64)
    for (u, v) in ((0, 0), (10, 7), (63, 47)):
        uo, vo = (u + 0.5) * 2 - 0.5, (v + 0.5) * 2 - 0.5     # same point on the original grid
        ray = lambda K, x, y: np.linalg.solve(K, [x, y, 1.0])
        assert np.allclose(ray(r.K, u, v), ray(f.K, uo, vo))
    assert r.depth.shape == (48, 64) and r.semantic.shape == (48, 64) and r.rgb.shape == (48, 64, 3)


def test_samples_are_deterministic_and_shaped():
    f = resize_frame(room(480, 640))
    a, b = make_sample(f, "f1", np.random.default_rng(3)), make_sample(f, "f1", np.random.default_rng(3))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["rgb"].shape == (3, 480, 640) and a["dep"].shape == (1, 480, 640) and a["labels"].shape == (480, 640)
    v = a["dep"][0] > 0
    assert np.isclose(np.median(a["dep"][0][v]), 1.0, atol=1e-6)              # median-normalised
    f0 = make_sample(f, "f0", np.random.default_rng(1))
    assert ((f0["labels"] == 1).mean() < (make_sample(f, "f1", np.random.default_rng(1))["labels"] == 1).mean())


def test_augment_and_sparsify():
    rng = np.random.default_rng(0)
    x = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
    a, b = augment([x, x.copy()], rng)
    assert a.shape == (480, 640) and np.array_equal(a, b)
    P = np.ones((480, 640), np.float32); V = np.ones((480, 640), bool)
    P2, V2 = sparsify(P, V, rng)
    assert 100 <= V2.sum() <= 2000 and np.all(P2[~V2] == 0)


def test_confinput_original_equals_omnidc_formula(monkeypatch):
    from baselines.completion import source_dir
    import sys
    src = source_dir("omnidc") / "src"
    if not (src / "loss" / "submodule" / "confinputloss.py").is_file():
        pytest.skip("OMNI-DC checkout missing")
    sys.path.insert(0, str(src))
    try:
        from loss.submodule import confinputloss as cl
    finally:
        sys.path.remove(str(src))
    shim = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    shim.abs = lambda x: torch.abs(x.float()) if x.dtype == torch.bool else torch.abs(x)
    monkeypatch.setattr(cl, "torch", shim)
    g = torch.Generator().manual_seed(0)
    gt = torch.rand(2, 1, 32, 48, generator=g) + 0.5
    dep = gt + (torch.rand(2, 1, 32, 48, generator=g) - 0.5) * 0.6; dep[:, :, :4] = 0
    conf = torch.rand(2, 1, 8, 12, generator=g) * 0.98 + 0.01
    ref = cl.ConfInputLoss(None)(conf, dep, gt)
    assert torch.allclose(confinput_original(conf, dep, gt), ref)
    lab = torch.zeros(2, 32, 48, dtype=torch.long)
    confident = torch.full_like(conf, 0.9)                 # a confident head is right on clean, wrong on far readings
    assert confinput_two_sided(confident, lab, dep) < confinput_two_sided(confident, torch.ones_like(lab), dep)
    assert confinput_two_sided(confident, 2 * torch.ones_like(lab), dep) == confinput_two_sided(confident, torch.ones_like(lab), dep)


def test_lr_schedule():
    from scripts.train_omnidc import lr_at
    cfg = dict(lr=1e-4, warmup_steps=10, final_lr_ratio=0.1)
    lrs = [lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-5) and max(lrs) == pytest.approx(1e-4) and lrs[-1] == pytest.approx(1e-5, rel=0.05)
    assert all(b <= a + 1e-12 for a, b in zip(lrs[10:], lrs[11:]))
