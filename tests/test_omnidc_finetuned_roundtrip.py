"""Saved final.pt -> production loader -> official CPU inference input/output path.

Only the large pretrained network construction and external Git/weight access
are replaced. torch.save/load, receipt verification, trainable-state filtering,
parameter restoration, and OmniDC.predict/outputs all execute their real code.
No model download, GPU operation, or dataset access is involved.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines import omnidc_finetuned as ft
from baselines.completion import OmniDC
from scripts.train_omnidc import trainable_state
from tests.test_omnidc_finetuned import training_run
from tp.provenance import sha256, write_json


class CpuInferenceNet(torch.nn.Module):
    """Small arithmetic net with trainable state AND a frozen subtree/buffers."""
    def __init__(self, trained=False):
        super().__init__()
        self.depth_module = torch.nn.Linear(1, 1, bias=False)
        self.depth_module.register_buffer("frozen_buffer", torch.tensor([29.0 if trained else 11.0]))
        self.head = torch.nn.Conv2d(1, 1, 1, bias=True)
        self.register_buffer("calibration_offset", torch.tensor([0.5 if trained else -5.0]))
        self.register_buffer("updates", torch.tensor(16666 if trained else 0, dtype=torch.int64))
        with torch.no_grad():
            # Deliberately different: a saved final must not overwrite DAv2.
            self.depth_module.weight.fill_(99.0 if trained else 7.0)
            self.head.weight.fill_(2.0 if trained else -1.0)
            self.head.bias.fill_(0.25 if trained else -3.0)
        self.observed = None

    def forward(self, sample):
        assert not self.training and not torch.is_grad_enabled()
        assert sample["rgb"].device.type == sample["dep"].device.type == "cpu"
        self.observed = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                         for key, value in sample.items()}
        pred = self.head(sample["dep"]) + self.calibration_offset + self.depth_module.weight.reshape(1, 1, 1, 1) / 100
        return {"pred": pred}


def final_artifact(tmp_path, variant):
    run, _, receipt, _ = training_run(tmp_path, "final.pt", variant)
    wrapped = torch.nn.Module()
    wrapped.add_module("net", CpuInferenceNet(trained=True))
    state = trainable_state(wrapped)
    # Same payload as the real full-training final writer; no optimizer or
    # embedded resume_contract exists in this format.
    payload = dict(model=state, step=16666, variant=variant)
    path = run / "checkpoints/final.pt"
    torch.save(payload, path)
    receipt.update(final_step=16666, final_sha256=sha256(path))
    write_json(run / "receipt.json", receipt)
    return run, receipt, payload


def external_dependencies(monkeypatch, receipt):
    spec = receipt["model_version"]["registry_entry"]
    monkeypatch.setattr(ft, "source_git", lambda src, *args: spec["commit"] if args[0] == "rev-parse" else "")
    weights = deepcopy(receipt["model_version"]["weights"])
    weights["mock/frozen.pth"]["path"] = "D:/local_pinned_weights/frozen.pth"
    monkeypatch.setattr(ft, "verify_weights", lambda key: deepcopy(weights))
    return weights


def small_official_baseline():
    # Use official preprocessing/padding/cropping without its heavyweight
    # constructor. The production finetuned loader still performs restoration.
    base = OmniDC.__new__(OmniDC)
    base.net = CpuInferenceNet()
    base.device, base.torch = "cpu", torch
    base.args = SimpleNamespace(num_resolution=3)
    base.load_info, base.deviations = {"fixture": "CPU arithmetic net"}, []
    return base


@pytest.mark.parametrize("variant", ["f0", "f1"])
def test_saved_final_roundtrip_restores_parameters_buffers_and_real_cpu_prediction(tmp_path, monkeypatch, variant):
    run, receipt, payload = final_artifact(tmp_path, variant)
    assert set(payload) == {"model", "step", "variant"}
    assert set(payload["model"]) == {"net.head.weight", "net.head.bias", "net.calibration_offset", "net.updates"}
    assert not any(key.startswith("net.depth_module.") for key in payload["model"])
    external_dependencies(monkeypatch, receipt)
    base = small_official_baseline()
    frozen_before = {key: value.clone() for key, value in base.net.depth_module.state_dict().items()}
    def build(key, device):
        assert key == "omnidc" and device == "cpu"
        return base
    monkeypatch.setattr(ft, "build_baseline", build)
    loaded = ft.FineTunedOmniDC(run, "final.pt", receipt["final_sha256"], variant, device="cpu")
    for key, value in payload["model"].items():
        torch.testing.assert_close(base.net.state_dict()[key.removeprefix("net.")], value, rtol=0, atol=0)
    for key, value in frozen_before.items():
        torch.testing.assert_close(base.net.depth_module.state_dict()[key], value, rtol=0, atol=0)
    # Odd dimensions exercise official bottom/right padding and output crop.
    rgb = np.zeros((17, 19, 3), np.uint8)
    prompt = np.full((17, 19), 20.0, np.float32)
    prompt[0, :3] = [0.0, 1.0, 2.0]
    prediction, report = loaded.predict(rgb, prompt, np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(prediction, 2 * prompt + np.float32(0.82), rtol=0, atol=1e-6)
    assert prediction.shape == prompt.shape and prediction.dtype == np.float32
    assert prediction[1, 1] > 40  # no training median normalization or storage clipping
    assert report == {"pad": [15, 13], "divisor": 16}
    observed = base.net.observed
    assert set(observed) == {"rgb", "dep", "K", "pattern"}  # no GT/training loss/gate input
    assert observed["rgb"].shape == (1, 3, 32, 32) and observed["dep"].shape == (1, 1, 32, 32)
    np.testing.assert_array_equal(observed["dep"][0, 0, :17, :19].numpy(), prompt)
    assert torch.count_nonzero(observed["dep"][0, 0, 17:]) == 0
    torch.testing.assert_close(observed["rgb"][0, :, 0, 0],
                               -torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225]))
    assert loaded.model_version["checkpoint"] == "final.pt" and loaded.model_version["step"] == 16666


@pytest.mark.parametrize("field", ["sha256", "revision", "size", "reference"])
def test_saved_final_rejects_wrong_pretrained_weight_contract_before_payload_load(tmp_path, monkeypatch, field):
    run, receipt, _ = final_artifact(tmp_path, "f1")
    actual = external_dependencies(monkeypatch, receipt)
    if field == "reference":
        actual["other/frozen.pth"] = actual.pop("mock/frozen.pth")
    else:
        actual["mock/frozen.pth"][field] = {"sha256": "d" * 64, "revision": "wrong-revision", "size": 33}[field]
    monkeypatch.setattr(ft, "verify_weights", lambda _: actual)
    monkeypatch.setattr(ft, "build_baseline", lambda *a, **k: pytest.fail("weight identity must fail before model build"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("weight identity must fail before checkpoint unpickle"))
    with pytest.raises(ValueError, match="pretrained/frozen weights differ"):
        ft.FineTunedOmniDC(run, "final.pt", receipt["final_sha256"], "f1", device="cpu")


@pytest.mark.parametrize("key", ["net.depth_module.weight", "net.depth_module.frozen_buffer"])
def test_saved_final_rejects_frozen_parameter_or_buffer_injection(tmp_path, monkeypatch, key):
    run, receipt, payload = final_artifact(tmp_path, "f0")
    payload["model"][key] = torch.tensor([[123.0]]) if key.endswith("weight") else torch.tensor([123.0])
    path = run / "checkpoints/final.pt"
    torch.save(payload, path)
    receipt["final_sha256"] = sha256(path)  # valid file binding cannot legitimize a frozen override
    write_json(run / "receipt.json", receipt)
    external_dependencies(monkeypatch, receipt)
    base = small_official_baseline()
    monkeypatch.setattr(ft, "build_baseline", lambda *a, **k: base)
    with pytest.raises(ValueError, match="frozen DAv2 override"):
        ft.FineTunedOmniDC(run, "final.pt", receipt["final_sha256"], "f0", device="cpu")
    assert base.net.depth_module.weight.item() == 7 and base.net.depth_module.frozen_buffer.item() == 11
