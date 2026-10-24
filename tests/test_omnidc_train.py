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
    g = a["gt"][0] > 0
    assert np.isclose(np.median(a["gt"][0][g]), 1.0, atol=1e-6)               # GT-median-normalised (OMNI-DC)
    assert np.allclose(a["dep"][0] * a["median_m"], make_sample(f, "f1", np.random.default_rng(3))["dep"][0] * a["median_m"])
    small = make_sample(f, "f1", np.random.default_rng(4), crop=(256, 320))
    assert small["rgb"].shape == (3, 256, 320) and small["labels"].shape == (256, 320)
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


def test_color_jitter_matches_torchvision():
    tvf = pytest.importorskip("torchvision.transforms.functional")
    from tp.omnidc.data import color_jitter
    rgb = (np.random.default_rng(0).random((16, 20, 3)) * 255).astype(np.uint8)
    b, c, s = np.random.default_rng(5).uniform(0.6, 1.4, 3)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
    ref = tvf.adjust_saturation(tvf.adjust_contrast(tvf.adjust_brightness(x, b), c), s)
    out = torch.from_numpy(color_jitter(rgb, np.random.default_rng(5)).transpose(2, 0, 1).copy()).float() / 255
    assert (out - ref).abs().max() < 2 / 255


def test_lr_schedule():
    from scripts.train_omnidc import lr_at
    cfg = dict(lr=1e-4, warmup_steps=10, final_lr_ratio=0.1)
    lrs = [lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-5) and max(lrs) == pytest.approx(1e-4) and lrs[-1] == pytest.approx(1e-5, rel=0.05)
    assert all(b <= a + 1e-12 for a, b in zip(lrs[10:], lrs[11:]))


@pytest.mark.parametrize("minutes", [0, -1, float("nan"), float("inf")])
def test_pilot_rejects_unbounded_or_nonpositive_budget(minutes):
    from scripts.train_omnidc import pilot_clock
    with pytest.raises(ValueError, match="positive and finite"):
        pilot_clock(minutes, 100)


def test_pilot_deadline_includes_startup_and_reserves_checkpoint_time():
    from scripts.train_omnidc import pilot_clock
    clock = pilot_clock(60, 100)
    assert clock["deadline"] == 3700
    assert clock["train_deadline"] == 3640
    assert clock["cleanup_reserve_seconds"] == 60
    # Expensive startup consumes the same wall budget, not an extra hour.
    assert clock["train_deadline"] - 700 == 2940
    assert pilot_clock(None, 100) is None
    assert 0 < pilot_clock(0.5, 100)["cleanup_reserve_seconds"] < 30


def _resume_fixture(tmp_path):
    from scripts.train_omnidc import training_contract
    from tp.provenance import write_json
    cfg = {"data": {"root": "data/hypersim", "selection": "configs/selection.json",
                    "training": "configs/training.yaml"}, "train": {"batch": 6}}
    for name in ("scripts/code.py", "configs/selection.json", "configs/training.yaml",
                 "data/hypersim/frames_v1.json", "configs/baselines/registry.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    contract = training_contract(cfg, "f1", tmp_path)
    out = tmp_path / "results/run"
    (out / "checkpoints").mkdir(parents=True)
    (out / "checkpoints/last.pt").write_bytes(b"fixture checkpoint")
    config = dict(variant="f1", config=cfg)
    write_json(out / "config.json", config)
    write_json(out / "receipt.json", dict(config=config, resume_contract=contract, stage="pilot_complete"))
    return out, cfg, contract


def test_resume_requires_checkpoint_and_matching_variant_configuration(tmp_path):
    from scripts.train_omnidc import validate_resume
    out, cfg, contract = _resume_fixture(tmp_path)
    assert validate_resume(out, "f1", cfg, contract)["stage"] == "pilot_complete"
    with pytest.raises(ValueError, match="variant"):
        validate_resume(out, "f0", cfg, contract)
    with pytest.raises(ValueError, match="configuration"):
        validate_resume(out, "f1", dict(cfg, train={"batch": 1}), contract)
    (out / "checkpoints/last.pt").unlink()
    with pytest.raises(ValueError, match="refusing to restart"):
        validate_resume(out, "f1", cfg, contract)


@pytest.mark.parametrize("changed", ["scripts/code.py", "configs/selection.json", "configs/training.yaml",
                                      "data/hypersim/frames_v1.json", "configs/baselines/registry.json"])
def test_resume_rejects_changed_code_selection_split_order_and_baseline(tmp_path, changed):
    from scripts.train_omnidc import training_contract, validate_resume
    out, cfg, _ = _resume_fixture(tmp_path)
    (tmp_path / changed).write_text("changed\n", encoding="utf-8")
    current = training_contract(cfg, "f1", tmp_path)
    with pytest.raises(ValueError, match="hashes differ"):
        validate_resume(out, "f1", cfg, current)


def test_checkpoint_cannot_be_swapped_between_variants_or_invalid_steps():
    from scripts.train_omnidc import validate_checkpoint
    contract = {"variant": "f1"}
    checkpoint = dict(variant="f1", resume_contract=contract, step=17)
    assert validate_checkpoint(checkpoint, "f1", contract, 100) == 17
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(dict(checkpoint, variant="f0"), "f1", contract, 100)
    for value in (-1, 101, True, 1.5):
        with pytest.raises(ValueError, match="step"):
            validate_checkpoint(dict(checkpoint, step=value), "f1", contract, 100)


def test_forced_checkpoint_roundtrip_is_atomic_and_retains_rng_cpu_only(tmp_path, monkeypatch):
    from scripts.train_omnidc import save_checkpoint
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward(); opt.step()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, opt, 7, "f1", {"test": True})
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["step"] == 7  # Deliberately not a periodic checkpoint step.
    assert set(saved["rng"]) == {"torch", "cuda", "numpy", "python"}
    assert not path.with_suffix(".tmp").exists()
    original = path.read_bytes()
    def fail_save(payload, temp):
        temp.write_bytes(b"partial")
        raise OSError("simulated disk error")
    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(path, model, opt, 8, "f1", {"test": True})
    assert path.read_bytes() == original


def test_nonfinite_gradient_guard_stops_before_optimizer_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
    assert parameter.item() == 1.0


def test_pilot_loop_counts_startup_and_saves_nonperiodic_step_on_cpu(tmp_path, monkeypatch):
    """Run the actual control loop with a tiny CPU model and a deterministic clock."""
    import json
    import sys
    import yaml
    import scripts.train_omnidc as trainer
    import tp.omnidc.data as training_data
    from tp.provenance import write_json
    clock = [0.0]
    monkeypatch.setattr(trainer, "time", types.SimpleNamespace(perf_counter=lambda: clock[0]))
    cfg = dict(seed=7, data=dict(root="data", selection="selection.json", training="training.yaml",
                               splits=["train"], p_sparse=.3),
               train=dict(samples=100, batch=1, workers=0, lr=1e-4, weight_decay=.01, warmup_steps=10,
                          final_lr_ratio=.1, grad_clip=1., log_every=20, checkpoint_every=2000))
    config_path = tmp_path / "p1.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training.yaml").write_text("min_mirror_share: 0.05", encoding="utf-8")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "CONFIG", config_path)
    monkeypatch.setattr(trainer, "training_contract", lambda *args: {"fixture": True})
    def start(out, config, command, seed):
        out.mkdir(parents=True)
        return dict(config=config, stage="started")
    monkeypatch.setattr(trainer, "start_run", start)
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, batch):
            clock[0] += 11.0
            return {"value": self.weight.square().sum()}
    def build(*args):
        clock[0] += 10.0  # Simulated model/GPU startup counts against the pilot.
        return ToyModel(), lambda batch, out: (out["value"], {"toy": float(out["value"].detach())}), types.SimpleNamespace(deviations=[])
    monkeypatch.setattr(trainer, "build", build)
    monkeypatch.setattr(trainer, "freeze_batchnorm", lambda model: None)
    monkeypatch.setattr(training_data, "training_frames", lambda *args: [{"role": "clean"}])
    monkeypatch.setattr(training_data, "HypersimTraining", lambda *args: None)
    class Loader:
        def __init__(self, *args, **kwargs):
            self.sampler = kwargs["sampler"]
        def __iter__(self):
            return iter({"index": torch.tensor([n])} for n in self.sampler)
    monkeypatch.setattr(torch.utils.data, "DataLoader", Loader)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture only")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(sys, "argv", ["train_omnidc", "--variant", "f1", "--pilot-minutes", "1"])
    trainer.main()
    run = next((tmp_path / "results").iterdir())
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["stage"] == "pilot_complete"
    assert receipt["pilot"]["steps"] == 4  # 10 startup + 4*11 training > 48-second train deadline.
    assert receipt["pilot"]["startup_wall_seconds"] == 10
    assert receipt["pilot"]["process_wall_seconds"] == 54
    checkpoint = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 4
    assert not (run / "checkpoints/final.pt").exists()
