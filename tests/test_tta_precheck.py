import sys

import numpy as np
import pytest

from scripts.tta_precheck import auroc_hist, hist, image_stats, scene_bootstrap, verdict


def test_auroc_hist_matches_pairwise_count():
    rng = np.random.default_rng(0)
    pos, neg = 10 ** rng.uniform(-3, 0, 300), 10 ** rng.uniform(-4, -1, 400)
    brute = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
    assert abs(auroc_hist(hist(pos), hist(neg)) - brute) < 0.01
    assert auroc_hist(hist(pos), hist([])) is None


def test_image_stats_counts_inheritance():
    G = np.full((6, 6), 2.0, np.float32)
    P = G.copy()
    P[:3] = 3.0                                     # wrong readings (rows 0-2)
    pv = np.ones_like(G, bool)
    gv = pv.copy()
    scope = np.zeros_like(pv)
    scope[:3] = True
    scope[3] = True                                 # 18 wrong + 6 correct readings in scope
    D_copy = P.copy()                               # copies the readings -> all C inherited
    D_fix = G.copy()                                # follows GT -> none inherited
    st = image_stats(P, pv, G, gv, scope, {0: D_fix, 100: D_copy})
    assert st["C"] == 18 and st["correct"] == 6
    assert st["steps"]["100"]["inherited"] == 18 and st["steps"]["0"]["inherited"] == 0
    assert sum(st["steps"]["0"]["pos_main"]) == 18 and sum(st["steps"]["0"]["neg_main"]) == 6
    assert "mean" in st["steps"]


def test_scene_bootstrap_constant():
    p, lo, hi = scene_bootstrap([1.0, 1.0, 1.0], lambda s: float(np.mean(s)), np.random.default_rng(1), draws=50)
    assert p == lo == hi == 1.0


def _res(dir_lo, ir100, dauc, dauc_lo):
    return dict(delta_IR=(0.1, dir_lo, 0.2), IR={"100": ir100}, delta_AUROC_10_0=(dauc, dauc_lo, 0.1))


def test_verdict_rule():
    v = verdict(dict(nyu=_res(0.01, 0.4, 0.05, 0.01), hammer=_res(-0.01, 0.1, 0.04, 0.005)))
    assert v["outcome"] == "go"
    v = verdict(dict(nyu=_res(0.01, 0.4, 0.05, 0.01), hammer=_res(-0.01, 0.1, 0.02, 0.005)))
    assert v["outcome"] == "a_without_b"
    v = verdict(dict(nyu=_res(0.01, 0.2, 0.05, 0.01), hammer=_res(-0.01, 0.4, 0.05, 0.01)))
    assert v["outcome"] == "premise_fails"


def test_recording_copy_matches_capa_loop():
    """run_recorded must give CAPA's _run_optimization result exactly (stub model, CPU)."""
    torch = pytest.importorskip("torch")
    from baselines.completion import source_dir
    src = source_dir("capa")
    if not (src / "capa" / "protocol.py").is_file():
        pytest.skip("CAPA source not fetched (external/baselines/capa)")
    sys.path.insert(0, str(src))
    try:
        from capa.protocol import CAPAProtocol
    except ImportError as e:
        pytest.skip(f"CAPA dependencies missing: {e}")
    from scripts.tta_precheck import run_recorded

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(3)
            self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)
            with torch.no_grad():
                self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * 0.1)
                self.conv.bias.fill_(1.0)
            self.delta = None

    class Stub:
        device = torch.device("cpu")

        def reset(self):
            self.model = Net()

        def inject_lora(self, rank, alpha, target_modules):
            self.model.delta = torch.nn.Parameter(torch.randn(1, 1, 8, 10) * 0.01)   # global RNG, like LoRA init

        def predict_depth(self, rgb):
            x = self.model.conv(rgb)
            return torch.nn.functional.softplus(x + self.model.delta)[:, 0]

        def get_trainable_params(self, lr):
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.delta.requires_grad = True
            return [{"params": [self.model.delta], "lr": lr}]

    proto = object.__new__(CAPAProtocol)
    proto.config = dict(lora_rank=4, lora_alpha=8, lora_target_modules=["qkv"])
    proto.device, proto.model, proto.tuning_mode = torch.device("cpu"), Stub(), "lora"
    proto.n_steps, proto.lr, proto.gradient_clip_norm = 12, 1e-2, 1.0
    proto.max_bs_optimize, proto.max_bs_infer, proto.pct_random_key_frame, proto.max_nan_retries = 8, 16, 10, 0
    proto.lr_scheduler_cfg, proto.align_n_points, proto.align_seed = None, 50, 47
    g = torch.Generator().manual_seed(5)
    rgb = torch.rand(1, 3, 8, 10, generator=g)
    cond = torch.rand(1, 8, 10, generator=g) * 3 + 1
    mask = torch.rand(1, 8, 10, generator=g) > 0.3
    torch.manual_seed(11)
    ref = proto._run_optimization(rgb, cond, mask)
    torch.manual_seed(11)
    mine, rec, losses = run_recorded(proto, rgb, cond, mask, record=(0, 5, 12))
    assert torch.equal(ref, mine)
    assert np.array_equal(rec[12], mine[0].numpy()) and set(rec) == {0, 5, 12} and len(losses) == 12
    assert not np.array_equal(rec[0], rec[12])
