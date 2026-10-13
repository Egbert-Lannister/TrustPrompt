"""T v0 network and the train -> infer -> cache loop on synthetic shards (needs torch)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tp.provenance import sha256, write_json
from tp.trust.model import IN_CH, TrustNetV0, auroc, make_inputs


def test_inputs_and_forward_shapes():
    rng = np.random.default_rng(0)
    p = rng.uniform(1, 3, (192, 256)).astype(np.float16); v = rng.random((192, 256)) > 0.2
    x = make_inputs(p, v, p, rng.integers(0, 255, (192, 256, 3), dtype=np.uint8))
    assert x.shape == (IN_CH, 192, 256) and np.isfinite(x).all()
    for use in (True, False):
        m = TrustNetV0(use_features=use)
        out = m(torch.from_numpy(x)[None], torch.zeros(1, 384, 24, 32) if use else None)
        assert out.shape == (1, 1, 192, 256)


def test_auroc_matches_definition():
    s = np.array([0.1, 0.4, 0.35, 0.8]); pos = np.array([False, True, False, True])
    assert auroc(s, pos) == 1.0 and auroc(s, ~pos) == 0.0 and auroc(s, np.ones(4, bool)) is None


def _fake_shard(root, dataset, scenes, split, rng):
    d = root / "shard_0of1"; (d / "items").mkdir(parents=True)
    index = []
    for sc in scenes:
        for variant in ("real", "sim0"):
            if split != "train" and variant != "real":
                continue
            gt = rng.uniform(1, 3, (192, 256)).astype(np.float32)
            p = gt.copy(); lab = np.zeros((192, 256), np.uint8)
            p[60:120, 80:160] *= 1.6; lab[60:120, 80:160] = 1                 # a far "mirror" block
            f = d / "items" / f"d435__{sc}__f0__{variant}.npz"
            focus = np.zeros_like(lab, bool); focus[50:130, 70:170] = True
            np.savez_compressed(f, prompt=p.astype(np.float16), valid=np.ones_like(lab, bool), dg=(gt * 1.05).astype(np.float16),
                                rgb=np.zeros((192, 256, 3), np.uint8), label=lab, gt=gt.astype(np.float16),
                                gt_valid=np.ones_like(lab, bool), focus=focus)
            index.append(dict(dataset=dataset, split=split, scene=sc, frame="f0", sensor="d435", variant=variant,
                              rgb_sha256=f"rgb_{sc}", sample_sha256=f"s_{sc}", path=f"items/{f.name}", sha256=sha256(f),
                              clean=int((lab == 0).sum()), contaminated=int((lab == 1).sum())))
    write_json(d / "index.json", index)
    write_json(d / "receipt.json", dict(stage="trust_data_verified"))
    return index


def test_train_infer_cache_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data_hammer"; feats = tmp_path / "feat"; (feats / "features").mkdir(parents=True)
    idx = _fake_shard(data, "hammer", ["scene2", "scene10"], "train", rng)
    feat_index = {}
    for e in idx:
        f = feats / "features" / f"{e['rgb_sha256']}.npy"
        if not f.exists():
            np.save(f, np.zeros((384, 24, 32), np.float16))
        feat_index[e["rgb_sha256"]] = dict(path=f"features/{f.name}")
    write_json(feats / "index.json", feat_index); write_json(feats / "receipt.json", dict(stage="features_verified"))
    model = tmp_path / "model"
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "scripts.train_trust", "--data", str(data), "--features", str(feats),
                        "--regime", "real_sim", "--output", str(model), "--steps", "60", "--batch", "2",
                        "--eval-every", "30", "--device", "cpu"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads((model / "receipt.json").read_text())
    assert rec["stage"] == "trust_model_trained" and rec["counts"]["selection_scenes"] == [["hammer", "scene10"]]
    assert rec["best"]["selection_auroc"] > 0.9                 # the block is trivially separable via log P/D_g
    cache = tmp_path / "cache"
    r = subprocess.run([sys.executable, "-m", "scripts.infer_trust", "--model", str(model), "--data", str(data),
                        "--features", str(feats), "--output", str(cache), "--device", "cpu"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    from tp.trust.cache import TrustCache
    c = TrustCache(cache)

    class S:
        sensor, scene, frame = "d435", "scene10", "f0"
        gt = np.zeros((456, 608), np.float32)
    t = c.trust(S(), "s_scene10")
    assert t.shape == (456, 608)
    block = np.zeros((192, 256), bool); block[60:120, 80:160] = True
    from tp.geom import resize_nearest
    b = resize_nearest(block, (456, 608))
    assert t[b].mean() < c.threshold < t[~b].mean()


def test_encoder_hook_takes_deepest_patch_tokens_from_a_dinov2_like_model():
    import torch.nn as nn
    from baselines.promptda import encoder_tokens, find_encoder_norm

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity() for _ in range(3)]); self.norm = nn.LayerNorm(8)

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained = ViT(); self.head_norm = nn.LayerNorm(8)   # a decoder LayerNorm must not be chosen

        def predict(self, x, prompt):
            ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
            for layer in (1, 2, 3):                                     # norm called once per intermediate layer
                tok = torch.full((1, 1 + ph * pw, 8), float(layer))
                self.pretrained.norm(tok)
            return torch.zeros(1, 1, *x.shape[-2:])

    class Pred:
        pass
    pred = Pred(); pred.torch = torch; pred.model = Fake(); pred.device = "cpu"
    with torch.no_grad():
        pred.model.pretrained.norm.weight.fill_(1); pred.model.pretrained.norm.bias.fill_(0)
    assert find_encoder_norm(pred.model) == "pretrained.norm"
    feat, info = encoder_tokens(pred, np.zeros((456, 608, 3), np.uint8), np.ones((192, 256), np.float32))
    assert feat.shape == (8, 24, 32) and info["hook_calls"] == 3 and info["token_grid"] == [33, 43]


def test_v1_error_selection_post_processing_and_mask_diagnostics(tmp_path):
    rng = np.random.default_rng(1)
    data = tmp_path / "data_hammer"; feats = tmp_path / "feat"; (feats / "features").mkdir(parents=True)
    idx = _fake_shard(data, "hammer", ["scene2", "scene10"], "train", rng)
    feat_index = {}
    for e in idx:
        f = feats / "features" / f"{e['rgb_sha256']}.npy"
        if not f.exists():
            np.save(f, np.zeros((384, 24, 32), np.float16))
        feat_index[e["rgb_sha256"]] = dict(path=f"features/{f.name}")
    write_json(feats / "index.json", feat_index); write_json(feats / "receipt.json", dict(stage="features_verified"))
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "model"
    r = subprocess.run([sys.executable, "-m", "scripts.train_trust", "--data", str(data), "--features", str(feats),
                        "--regime", "real_sim", "--output", str(model), "--steps", "60", "--batch", "2", "--eval-every", "30",
                        "--device", "cpu", "--selection", "error", "--feat-dropout", "0.2", "--feat-drop-sample", "0.2"],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    b = json.loads((model / "receipt.json").read_text())["best"]
    assert b["selection"] == "error" and isinstance(b["post"], bool) and b["objective"] < 0.05     # D_g error is 0.05
    cache = tmp_path / "cache_hammer"
    r = subprocess.run([sys.executable, "-m", "scripts.infer_trust", "--model", str(model), "--data", str(data),
                        "--features", str(feats), "--output", str(cache), "--device", "cpu"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((cache / "receipt.json").read_text())["post"] == b["post"]
    out = tmp_path / "diag"
    r = subprocess.run([sys.executable, "-m", "scripts.trust_mask_diagnostics", "--data", str(data),
                        "--caches", f"v1={tmp_path / 'cache_{tag}'}", "--output", str(out)], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(Path(f"{out}.json").read_text())["results"]
    assert res["hammer/train holdout/v1"]["recall_scope"] > 0.9 and res["hammer/train holdout/Gate"]["images"] == 1
