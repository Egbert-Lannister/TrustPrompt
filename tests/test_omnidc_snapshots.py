import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def test_config_variant_and_snapshots_on_cpu(tmp_path, monkeypatch):
    """--config picks the variant table; train.snapshot_every keeps hash-recorded weights (tiny CPU model)."""
    import yaml
    import scripts.train_omnidc as trainer
    import tp.omnidc.data as training_data
    from tp.provenance import sha256
    cfg = dict(seed=7, run_prefix="toy", data=dict(root="data", selection="selection.json", training="training.yaml",
                                                   splits=["train"], p_sparse=.3),
               train=dict(samples=6, batch=1, workers=0, lr=1e-4, weight_decay=.01, warmup_steps=2, final_lr_ratio=.1,
                          grad_clip=1., log_every=20, checkpoint_every=100, snapshot_every=2),
               loss=dict(gate_weight=1.0), variants=dict(f1s=dict(gate=False, inputs="f1", label="soft")))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "toy.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training.yaml").write_text("min_mirror_share: 0.05", encoding="utf-8")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "training_contract", lambda *args: {"fixture": True})
    monkeypatch.setattr(trainer, "verified_model_version", lambda variant: {"name": "CPU toy", "variant": variant})
    monkeypatch.setattr(trainer, "start_run", lambda out, config, command, seed: (out.mkdir(parents=True), dict(config=config, stage="started"))[1])
    seen = {}

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, batch):
            return {"value": self.weight.square().sum()}

    def build(variant, cfg_):
        seen["label"] = cfg_["variants"][variant]["label"]
        return ToyModel(), lambda batch, out: (out["value"], {}), types.SimpleNamespace(deviations=[], load_info={})
    monkeypatch.setattr(trainer, "build", build)
    monkeypatch.setattr(trainer, "freeze_batchnorm", lambda model: None)
    monkeypatch.setattr(trainer, "grad_norms", lambda model: {})
    monkeypatch.setattr(training_data, "training_frames", lambda *args: [dict(scene="s", camera="cam_00", frame=0, split="train", role="clean")])
    monkeypatch.setattr(training_data, "HypersimTraining", lambda *args: seen.setdefault("inputs", args[3]))

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
    monkeypatch.setattr(sys, "argv", ["train_omnidc", "--config", "configs/toy.yaml", "--variant", "f1s"])
    trainer.main()
    run = next((tmp_path / "results").iterdir())
    assert run.name.startswith("toy_f1s_")
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["stage"] == "training_complete" and seen == dict(label="soft", inputs="f1")
    assert sorted(receipt["snapshots"]) == ["step_000002.pt", "step_000004.pt"]
    for name, digest in receipt["snapshots"].items():
        assert sha256(run / "checkpoints" / name) == digest
        assert torch.load(run / "checkpoints" / name, weights_only=False)["step"] == int(name[5:11])
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["train_omnidc", "--config", "configs/toy.yaml", "--variant", "f0"])
        trainer.main()
