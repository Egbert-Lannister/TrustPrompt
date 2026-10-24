"""CPU-only checkpoint/caching regressions; no real weights or dataset reads."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from baselines import omnidc_finetuned as ft
from baselines.cached_prior import CachedPrior
from scripts import cache_omnidc_finetuned as cache
from scripts.make_synthetic import make_sample
from tp.data.schema import save_sample
from tp.eval.metrics import evaluate_depth
from tp.eval.regions import regions
from tp.provenance import ROOT, sha256, write_json


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_module = torch.nn.Linear(1, 1, bias=False)
        self.head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.depth_module.weight.fill_(7)
            self.head.weight.fill_(1)


def training_run(tmp_path, checkpoint="last.pt", variant="f0", stage=None):
    run = tmp_path / "training"
    (run / "checkpoints").mkdir(parents=True)
    spec = ft.registry()["models"]["omnidc"]
    code = {"baselines/completion.py": sha256(ROOT / "baselines/completion.py")}
    contract = dict(variant=variant, code_sha256=code, input_sha256={"selection.json": "c" * 64})
    cfg = dict(variant=variant, config=dict(data=dict(splits=["train"]), variants={variant: dict(gate=False)}),
               smoke=False, pilot_minutes=60, config_sha256="a" * 64)
    frames = dict(requested_splits=["train"], frame_count=1, scene_count=1,
                  scenes={"room": dict(split="train", frame_count=1)},
                  frames=[dict(scene="room", frame="001", split="train", role="clean")])
    write_json(run / "training_frames.json", frames)
    # Use the actual training writer's helper, including frozen-parameter omission.
    from scripts.train_omnidc import trainable_state
    wrapped = torch.nn.Module()
    wrapped.add_module("net", TinyNet())
    with torch.no_grad():
        wrapped.net.head.weight.fill_(3)
    state = trainable_state(wrapped)
    ck = dict(model=state, variant=variant, step=4)
    if checkpoint == "last.pt":
        ck["resume_contract"] = contract
        ck["optimizer"] = {"unused": torch.ones(3)}
    torch.save(ck, run / "checkpoints" / checkpoint)
    digest = sha256(run / "checkpoints" / checkpoint)
    weights = {"mock/frozen.pth": dict(path="/training/frozen.pth", size=32, sha256="b" * 64, revision="v1")}
    receipt = dict(config=cfg, stage=stage or ("training_complete" if checkpoint == "final.pt" else "pilot_complete"),
                   resume_contract=contract, source_sha256=code, final_step=4, final_sha256=digest,
                   pilot=dict(final_step=4, last_checkpoint_step=4),
                   training_data=dict(manifest="training_frames.json", sha256=sha256(run / "training_frames.json"),
                                      frame_count=1, scene_count=1, scene_splits={"room": "train"}),
                   model_version=dict(name=spec["label"], key="omnidc", variant=variant, registry_entry=spec,
                                      source_commit=spec["commit"], source_status="", weights=weights, load_info={}))
    write_json(run / "receipt.json", receipt); write_json(run / "config.json", cfg)
    return run, digest, receipt, ck


def mock_base(monkeypatch, receipt):
    spec = ft.registry()["models"]["omnidc"]
    monkeypatch.setattr(ft, "source_git", lambda src, *args: spec["commit"] if args[0] == "rev-parse" else "")
    actual = deepcopy(receipt["model_version"]["weights"])
    actual["mock/frozen.pth"]["path"] = "D:/different/host/frozen.pth"
    monkeypatch.setattr(ft, "verify_weights", lambda key: actual)
    base = SimpleNamespace(net=TinyNet(), min_points=1, load_info={"mock": True}, deviations=["CPU fixture"])
    base.calls = 0
    def predict(rgb, depth, K):
        assert not base.net.training and not base.net.depth_module.training
        base.calls += 1
        return depth.copy(), {"mock": True}
    base.predict = predict
    monkeypatch.setattr(ft, "build_baseline", lambda key, device: base)
    return base


@pytest.mark.parametrize("checkpoint", ["last.pt", "final.pt"])
@pytest.mark.parametrize("variant", ["f0", "f1"])
def test_loads_both_checkpoint_schemas_without_overwriting_frozen_or_training_forward(tmp_path, monkeypatch, checkpoint, variant):
    run, digest, receipt, ck = training_run(tmp_path, checkpoint, variant)
    if checkpoint == "final.pt":
        assert set(ck) == {"model", "step", "variant"}  # exact final writer schema; no embedded contract
    base = mock_base(monkeypatch, receipt)
    model = ft.FineTunedOmniDC(run, checkpoint, digest, variant, device="cpu")
    assert base.net.head.weight.item() == 3
    assert base.net.depth_module.weight.item() == 7
    rgb = np.zeros((17, 19, 3), np.uint8)
    prompt = np.full((17, 19), 20, np.float32); prompt[0, 0] = 0
    got, _ = model.predict(rgb, prompt, np.eye(3))
    np.testing.assert_array_equal(got, prompt)
    assert base.calls == 1
    version = model.model_version
    assert version["checkpoint_sha256"] == digest and version["step"] == 4
    assert version["training_artifact_sha256"]["config.json"] == sha256(run / "config.json")
    assert version["formal_experiment_complete"] is False
    assert version["experiment_role"] == ft.ROLE


@pytest.mark.parametrize("kind", ["sha", "config", "variant", "stage", "source_code", "frames_sha", "frame_split", "final_sha", "source_commit"])
def test_rejects_run_identity_failures_before_weight_verification_or_model_build(tmp_path, monkeypatch, kind):
    checkpoint = "final.pt" if kind == "final_sha" else "last.pt"
    run, digest, receipt, _ = training_run(tmp_path, checkpoint)
    if kind == "sha":
        digest = "0" * 64
    elif kind == "config":
        write_json(run / "config.json", {})
    elif kind == "variant":
        receipt["config"]["variant"] = "f1"
        write_json(run / "config.json", receipt["config"])
    elif kind == "stage":
        receipt["stage"] = "training"
    elif kind == "source_code":
        receipt["source_sha256"] = {}
    elif kind == "frames_sha":
        (run / "training_frames.json").write_text("{}", encoding="utf-8")
    elif kind == "frame_split":
        frames = json.loads((run / "training_frames.json").read_text())
        frames["frames"][0]["split"] = "test"
        write_json(run / "training_frames.json", frames)
        receipt["training_data"]["sha256"] = sha256(run / "training_frames.json")
    elif kind == "final_sha":
        receipt["final_sha256"] = "0" * 64
    elif kind == "source_commit":
        receipt["model_version"]["source_commit"] = "wrong"
    write_json(run / "receipt.json", receipt)
    monkeypatch.setattr(ft, "verify_base_version", lambda _: pytest.fail("must reject before weight/model reads"))
    with pytest.raises(ValueError):
        ft.FineTunedOmniDC(run, checkpoint, digest, "f0", device="cpu")


@pytest.mark.parametrize("kind", ["variant", "step", "contract"])
@pytest.mark.parametrize("checkpoint", ["last.pt", "final.pt"])
def test_checkpoint_payload_is_bound_to_receipt(tmp_path, monkeypatch, kind, checkpoint):
    run, _, receipt, ck = training_run(tmp_path, checkpoint)
    mock_base(monkeypatch, receipt)
    if kind == "variant": ck["variant"] = "f1"
    if kind == "step": ck["step"] = 5
    if kind == "contract": ck["resume_contract"] = {}
    path = run / "checkpoints" / checkpoint
    torch.save(ck, path)
    if checkpoint == "final.pt":
        receipt["final_sha256"] = sha256(path)
        write_json(run / "receipt.json", receipt)
    monkeypatch.setattr(ft, "build_baseline", lambda *a, **k: pytest.fail("must reject before model build"))
    with pytest.raises(ValueError):
        ft.FineTunedOmniDC(run, checkpoint, sha256(path), "f0", device="cpu")


@pytest.mark.parametrize("bad_state", [{"net.depth_module.weight": torch.ones(1, 1)},
                                      {"gate.weight": torch.ones(1, 1)},
                                      {"net.other.weight": torch.ones(1, 1)}, {}])
def test_state_rejects_frozen_gate_unexpected_missing_keys(bad_state):
    with pytest.raises(ValueError):
        ft.restore_state(TinyNet(), bad_state)


@pytest.mark.parametrize("kind", ["weights", "dirty", "head"])
def test_actual_source_and_frozen_weights_must_match_training(tmp_path, monkeypatch, kind):
    run, digest, receipt, _ = training_run(tmp_path)
    mock_base(monkeypatch, receipt)
    if kind == "weights":
        changed = deepcopy(receipt["model_version"]["weights"])
        changed["mock/frozen.pth"]["sha256"] = "d" * 64
        monkeypatch.setattr(ft, "verify_weights", lambda _: changed)
    else:
        commit = receipt["model_version"]["source_commit"]
        monkeypatch.setattr(ft, "source_git", lambda src, *args: ("other" if kind == "head" else commit)
                            if args[0] == "rev-parse" else " M file.py")
    monkeypatch.setattr(ft, "build_baseline", lambda *a, **k: pytest.fail("must reject before model build"))
    with pytest.raises(ValueError):
        ft.FineTunedOmniDC(run, "last.pt", digest, "f0", device="cpu")


def test_f2_rejected_without_reading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="only f0/f1"):
        ft.inspect_run(tmp_path / "absent", "last.pt", "a" * 64, "f2")


def test_test_split_rejected_before_any_config_data_checkpoint_or_model_access(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "FineTunedOmniDC", lambda *a, **k: pytest.fail("model must not be read"))
    with pytest.raises(ValueError, match="test is locked"):
        cache.run(tmp_path / "absent", "f0", "last.pt", "a" * 64, "nyu", tmp_path / "out",
                  splits=["test"], config_path=tmp_path / "absent.yaml")
    assert not (tmp_path / "out").exists()


def test_formal_config_rejected_before_manifest_or_checkpoint(tmp_path):
    config = tmp_path / "formal.yaml"
    config.write_text("experiment_role: formal_main_result\n", encoding="utf-8")
    with pytest.raises(ValueError, match="development-only"):
        cache.run(tmp_path / "absent", "f0", "last.pt", "a" * 64, "nyu", tmp_path / "out", config_path=config)


def test_raw_prompt_model_input_does_not_read_gt():
    class NoGT:
        rgb = np.zeros((2, 3, 3), np.uint8)
        prompt = np.array([[20, 2, np.nan], [3, 4, 5]], np.float32)
        prompt_valid = np.array([[True, True, False], [True, True, True]])
        @property
        def gt(self): raise AssertionError("GT must not enter raw input")
    depth, guide, info = cache.model_input(NoGT(), "raw", tau=.05)
    assert depth[0, 0] == 20 and depth[0, 2] == 0 and guide is None and info["removed"] == 0


def synthetic_manifest(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    records, samples = [], []
    for i in range(3):
        s = make_sample(i)
        s.prompt = s.gt.copy(); s.prompt_valid = s.gt_valid.copy(); s.K_prompt = s.K_rgb.copy()
        s.scene = "same_scene"; s.split = "val"
        path = processed / f"{i}.npz"
        save_sample(s, path)
        records.append(dict(path=path.name, sha256=sha256(path), **{k: getattr(s, k) for k in cache.IDENTITY}))
        samples.append(s)
    path = processed / "manifest.json"
    write_json(path, dict(schema="trustprompt.manifest.v1", synthetic=True, records=records))
    config = tmp_path / "cache.yaml"
    config.write_text(yaml.safe_dump(dict(seed=1, experiment_role=ft.ROLE, storage="float32", alignment="none",
                                         datasets={"nyu": dict(manifest=str(path), train_scope="all")})), encoding="utf-8")
    return config, samples, records


@pytest.mark.parametrize("legacy_without_sensor", [False, True])
def test_cache_and_existing_evaluator_preserve_float_invalids_exact_cohort_and_provenance(tmp_path, monkeypatch, legacy_without_sensor):
    config, samples, records = synthetic_manifest(tmp_path)
    if legacy_without_sensor:
        path = tmp_path / "processed/manifest.json"
        manifest = json.loads(path.read_text())
        for record in manifest["records"]:
            record.pop("sensor")
        write_json(path, manifest)
    class MockModel:
        min_points = 1
        model_version = dict(name="OMNI-DC-v1.1-F0", training_stage="pilot_complete",
                             experiment_role=ft.ROLE, formal_experiment_complete=False)
        def __init__(self, *args, **kwargs): pass
        def predict(self, rgb, depth, K):
            pred = depth.copy(); pred[0, :5] = [np.nan, np.inf, 0, -1, 20]
            return pred, dict(mock=True)
    monkeypatch.setattr(cache, "FineTunedOmniDC", MockModel)
    output = tmp_path / "cache"
    # The existing evaluator is actually exercised on the two synthetic frames.
    receipt = cache.run(tmp_path, "f0", "last.pt", "a" * 64, "nyu", output, splits=["val"], limit=2,
                        device="cpu", config_path=config, evaluate=True)
    assert receipt["frame_count"] == 2 and receipt["synthetic"] is True
    assert receipt["validation_stage"] == "synthetic_fixture_verified"
    assert receipt["stage_semantics"].startswith("legacy_reader_compatibility_only")
    assert receipt["formal_experiment_complete"] is False
    assert "uint16" in receipt["comparison_caution"]
    index = json.loads((output / "images.json").read_text())
    assert all(r["storage"] == "float32" and r["clipped_pixels"] == 0 and r["invalid_pixels"] == 4 for r in index)
    reader = CachedPrior(output, expected_model="OMNI-DC-v1.1-F0")
    pred = reader.predict(samples[0], records[0]["sha256"])
    assert pred.dtype == np.float32 and pred[0, 4] == 20
    assert np.isnan(pred[0, 0]) and np.isposinf(pred[0, 1]) and pred[0, 2] == 0 and pred[0, 3] == -1
    s = samples[0]
    metrics = evaluate_depth(pred, s.gt, s.prompt, regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid))["regions"]["all"]
    assert metrics["count"] == pred.size and metrics["invalid_count"] == 4
    assert metrics["absrel"] is None and metrics["status"] == "invalid_predictions"
    assert metrics["delta_1_25"] == pytest.approx((pred.size - 5) / pred.size)
    selected = json.loads((output / "selected_manifest.json").read_text())
    assert len(selected["records"]) == 2
    assert all(r["sensor"] == samples[0].sensor for r in selected["records"])
    generated = yaml.safe_load((output / "eval_val.yaml").read_text())
    assert generated["alignment"] == "none" and generated["experiment_role"] == ft.ROLE
    evaluated = json.loads((output / "evaluation_val/per_image.json").read_text())
    assert len(evaluated) == 2 and all(r["metrics"]["regions"]["all"]["invalid_count"] == 4 for r in evaluated)
    assert all(r["synthetic"] is True for r in evaluated)
    summary = json.loads((output / "evaluation_val/summary.json").read_text())
    assert summary["qualification"]["synthetic"] is True
    assert (output / "evaluation_val/summary.json").is_file()
    # Integrity verification is inherited from CachedPrior.
    with (output / "predictions/000000.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        reader.predict(samples[0], records[0]["sha256"])
    # Document the deployed reader limitation rather than silently assuming a
    # new explicit synthetic enum is accepted. Do not change that old API here.
    receipt["stage"] = "synthetic_fixture_verified"
    write_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="not verified/complete"):
        CachedPrior(output, expected_model="OMNI-DC-v1.1-F0")
