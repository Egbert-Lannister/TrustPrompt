"""Load a locally produced F0/F1 checkpoint for development inference only.

No download, training loss, GT normalization, gate, or scale alignment. Both the
training run directory and an independently recorded checkpoint SHA256 are
required. A pilot/last checkpoint is a diagnostic, never a finished experiment.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from baselines.completion import build as build_baseline, registry, source_dir, verify_weights
from scripts.setup_baselines import git as source_git
from tp.provenance import ROOT, sha256

FROZEN = "net.depth_module."
ROLE = "development_diagnostic_not_final_test"
STAGES = {"pilot_complete", "pilot_no_training", "smoke_complete", "training_complete"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _weights_identity(weights):
    # Machine-local paths deliberately do not participate in identity.
    return {key: {field: value[field] for field in ("size", "sha256", "revision")}
            for key, value in weights.items()}


def inspect_run(run_dir, checkpoint, checkpoint_sha256, variant):
    """Verify metadata and hashes before unpickling a trusted training artifact.

    last.pt has no embedded config hash: its resume contract and reported step
    are checked, and the caller-supplied SHA binds the copied file. Preserve the
    original run receipt/config/training_frames together when copying a run.
    """
    if variant not in {"f0", "f1"}:
        raise ValueError("only f0/f1 development checkpoints are supported")
    if checkpoint not in {"last.pt", "final.pt"}:
        raise ValueError("checkpoint must be last.pt or final.pt inside the run")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(checkpoint_sha256)):
        raise ValueError("an independently recorded checkpoint SHA256 is required")
    run = Path(run_dir).resolve()
    path = run / "checkpoints" / checkpoint
    digest = sha256(path)
    if digest != checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA256 mismatch")
    receipt, saved = _json(run / "receipt.json"), _json(run / "config.json")
    if receipt.get("config") != saved:
        raise ValueError("training receipt/config mismatch")
    stage = receipt.get("stage")
    if stage not in STAGES:
        raise ValueError("training run is incomplete/failed or still writing")
    if saved.get("variant") != variant:
        raise ValueError("training configuration variant mismatch")
    cfg = saved["config"]
    if cfg["data"]["splits"] != ["train"] or cfg["variants"][variant].get("gate") is not False:
        raise ValueError("expected train-only, ungated F0/F1 configuration")
    contract = receipt.get("resume_contract", {})
    code, inputs = contract.get("code_sha256", {}), contract.get("input_sha256", {})
    if contract.get("variant") != variant or not code or not inputs:
        raise ValueError("missing training code/data identity contract")
    if any(receipt.get("source_sha256", {}).get(k) != v for k, v in code.items()):
        raise ValueError("training contract code differs from receipt source hashes")
    # We reuse this inference wrapper, so a different implementation needs an
    # explicit compatibility review rather than silently loading the same keys.
    wrapper = "baselines/completion.py"
    if code.get(wrapper) != sha256(ROOT / wrapper):
        raise ValueError("baseline inference wrapper differs from training source")
    td = receipt.get("training_data", {})
    if td.get("manifest") != "training_frames.json" or sha256(run / "training_frames.json") != td.get("sha256"):
        raise ValueError("training frame manifest SHA256 mismatch/missing")
    frames = _json(run / "training_frames.json")
    rows = frames.get("frames", [])
    counts = Counter(r["scene"] for r in rows)
    scenes = {s: dict(split="train", frame_count=n) for s, n in counts.items()}
    if (not rows or any(r["split"] != "train" for r in rows)
            or frames.get("requested_splits") != ["train"] or frames.get("scenes") != scenes
            or frames.get("frame_count") != len(rows) or frames.get("scene_count") != len(counts)
            or td.get("frame_count") != len(rows) or td.get("scene_count") != len(counts)
            or td.get("scene_splits") != {s: "train" for s in counts}):
        raise ValueError("training frame/split identities inconsistent or not train-only")
    if checkpoint == "final.pt":
        if stage != "training_complete" or receipt.get("final_sha256") != digest:
            raise ValueError("final checkpoint is not bound to a completed training receipt")
        step = receipt.get("final_step")
    elif stage.startswith("pilot_"):
        step = receipt.get("pilot", {}).get("last_checkpoint_step")
        if step != receipt.get("pilot", {}).get("final_step"):
            raise ValueError("pilot checkpoint step mismatch")
    else:
        step = receipt.get("final_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("missing/invalid reported checkpoint step")
    trained = receipt.get("model_version", {})
    spec = registry()["models"]["omnidc"]
    if (trained.get("key") != "omnidc" or trained.get("variant") != variant
            or trained.get("registry_entry") != spec or trained.get("source_commit") != spec["commit"]
            or trained.get("source_status") != "" or "load_info" not in trained):
        raise ValueError("training model/source/frozen-weight version is incomplete or differs")
    return dict(run=run, path=path, receipt=receipt, config=saved, step=step, variant=variant,
                checkpoint_sha256=digest, checkpoint=checkpoint,
                hashes={name: sha256(run / name) for name in ("receipt.json", "config.json", "training_frames.json")})


def verify_base_version(trained):
    """Read-only check of the actual pinned code and all pretrained/frozen bytes."""
    spec = registry()["models"]["omnidc"]
    src = source_dir("omnidc")
    head = source_git(src, "rev-parse", "HEAD")
    status = source_git(src, "status", "--porcelain=v1", "--untracked-files=no")
    if head != spec["commit"] or status:
        raise ValueError("actual OMNI-DC source is not the clean pinned commit")
    weights = verify_weights("omnidc")
    if not weights or _weights_identity(weights) != _weights_identity(trained["weights"]):
        raise ValueError("actual pretrained/frozen weights differ from training receipt")
    return dict(source=str(src), source_commit=head, source_status=status, weights=weights)


def restore_state(net, state):
    """Only the non-frozen net.* state is legal; DAv2 remains verified pretrained."""
    if not isinstance(state, dict) or not state:
        raise ValueError("empty/malformed trainable checkpoint state")
    if any(not k.startswith("net.") or k.startswith(FROZEN) for k in state):
        raise ValueError("checkpoint includes a gate, foreign key, or frozen DAv2 override")
    converted = {k[4:]: v for k, v in state.items()}
    expected = {k for k in net.state_dict() if not k.startswith("depth_module.")}
    if set(converted) != expected:
        raise ValueError("checkpoint trainable keys do not exactly match the model")
    missing, unexpected = net.load_state_dict(converted, strict=False)
    if unexpected or any(not k.startswith("depth_module.") for k in missing):
        raise ValueError("checkpoint load mismatch")
    net.eval()


class FineTunedOmniDC:
    """F0/F1 metric raw-prompt inference; deliberately no training wrapper forward."""

    def __init__(self, run_dir, checkpoint, checkpoint_sha256, variant, device="cuda"):
        import torch
        meta = inspect_run(run_dir, checkpoint, checkpoint_sha256, variant)
        trained = meta["receipt"]["model_version"]
        actual = verify_base_version(trained)
        # Only trusted own-training files with a caller-provided digest reach
        # pickle. mmap avoids materialising the unused optimizer tensors.
        ck = torch.load(meta["path"], map_location="cpu", weights_only=False, mmap=True)
        if (ck.get("variant") != variant or isinstance(ck.get("step"), bool)
                or not isinstance(ck.get("step"), int) or ck.get("step") != meta["step"]):
            raise ValueError("checkpoint variant/step differs from run receipt")
        # The actual final writer saves only model/step/variant. Its digest is
        # bound by the completed receipt; do not require a nonexistent field.
        # If a future final format includes a contract, it must still agree.
        if (checkpoint == "last.pt" or "resume_contract" in ck) and ck.get("resume_contract") != meta["receipt"]["resume_contract"]:
            raise ValueError("checkpoint resume contract differs from run receipt")
        state = ck["model"]
        self.base = build_baseline("omnidc", device=device)
        restore_state(self.base.net, state)
        self.min_points = self.base.min_points
        self.model_version = dict(
            name=f"OMNI-DC-v1.1-{variant.upper()}", key="omnidc_finetuned", variant=variant,
            checkpoint=checkpoint, checkpoint_sha256=meta["checkpoint_sha256"], step=meta["step"],
            training_stage=meta["receipt"]["stage"], training_run=str(meta["run"]),
            training_artifact_sha256=meta["hashes"], training_model_version=trained,
            training_config=meta["config"], training_git_commit=meta["receipt"].get("git_commit"),
            training_source_sha256=meta["receipt"]["source_sha256"],
            training_resume_contract=meta["receipt"]["resume_contract"],
            training_data=meta["receipt"]["training_data"], actual_base_version=actual,
            load_info=self.base.load_info, deviations=self.base.deviations,
            checkpoint_binding="caller SHA + receipt/config + resume contract (last) or receipt final SHA (final)",
            experiment_role=ROLE, dev_only=True, formal_experiment_complete=False,
            inference="official eval forward, raw metric prompt; no GT normalization, clipping, or alignment")

    def predict(self, rgb, depth, K):
        return self.base.predict(rgb, depth, K)
