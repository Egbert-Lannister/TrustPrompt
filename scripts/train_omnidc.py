"""Fine-tune OMNI-DC v1.1 for the mainline (P1: f0, f1; P2: f2 and ablations). GPU; meant for a rented 48 GB card.

  python -m scripts.train_omnidc --variant f1 [--pilot-minutes 60] [--resume results/p1_f1_<time>] [--smoke STEPS]
      [--config configs/omnidc/p1.yaml]

--config: another fixed configuration (e.g. configs/omnidc/p1v2_precheck.yaml). A variant entry may name its input
recipe (`inputs`: f0 / f1, tp.omnidc.data) and its ConfInput target (`label`: original / two_sided / soft,
tp.omnidc.losses); without them f0 = f0 inputs + original label, others = f1 inputs + two-sided label (P1 as run).
Run folders are named <run_prefix>_<variant>_<time> (run_prefix default p1). train.snapshot_every N (optional) also
keeps the weights every N steps as checkpoints/step_<N>.pt (SHA-256 in the receipt), for trend checks.

Config: configs/omnidc/p1.yaml (fixed before training). Data: tp/omnidc/data.py on the downloaded Hypersim frames
(only Hypersim's train scenes; no NYU / HAMMER image is used, so both stay zero-shot evaluations). Model:
tp/omnidc/model.GatedOMNIDC (gate only in f2). Loss: tp/omnidc/losses.FineTuneLoss. Depth Anything V2 stays frozen
as in OMNI-DC; batch-norm statistics are frozen (fine-tuning with small batches).
--pilot-minutes N  process-wall budget including startup, reserving up to 60 seconds to save. Report training
                   throughput separately from startup/save time; scripts/linux/p1_run.sh adds a hard watchdog.
Writes results/p1_<variant>_<time>/: receipt, config, log.jsonl (every log_every steps), checkpoints/last.pt (every
checkpoint_every steps and at pilot exit, atomic, with RNG states) and final.pt, both without the frozen Depth Anything V2 weights (load_trainable puts
them back from OMNI-DC's pinned weights). --resume continues from last.pt with the same config and code.
--smoke STEPS      laptop check of this exact script (scripts/omnidc_p1_smoke.py): the same data, model and loss with
                   256 x 320 crops, batch 1, 2 workers, a log line every step and a checkpoint every 10 steps; stops
                   after STEPS steps in total (with --resume: continues to STEPS). Writes results/p1smoke_<variant>_<time>/
                   and no final.pt; nothing from it is used for any result.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
import json
import math
import random
import sys
import time
from pathlib import Path

import yaml

from tp.provenance import ROOT, sha256, start_run, write_json

CONFIG = ROOT / "configs" / "omnidc" / "p1.yaml"
SMOKE_TRAIN = dict(batch=1, workers=2, log_every=1, checkpoint_every=10)
FROZEN = "net.depth_module."                              # Depth Anything V2: frozen, pinned weights, not saved
SMOKE_CROP = [256, 320]


def pilot_clock(minutes, started):
    """Process-wall budget; leave a bounded save window, not a GPU-kernel timer."""
    if minutes is None:
        return None
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("pilot-minutes must be positive and finite")
    seconds = 60 * minutes
    reserve = min(60.0, seconds * 0.2)
    return dict(deadline=started + seconds, train_deadline=started + seconds - reserve,
                budget_seconds=seconds, cleanup_reserve_seconds=reserve)


def training_contract(cfg, variant, root=ROOT):
    """Bind resume to executable code, data selection/order and pinned baseline."""
    root = Path(root)
    code = {}
    for folder in ("tp", "baselines", "scripts"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts:
                code[path.relative_to(root).as_posix()] = sha256(path)
    data = cfg["data"]
    paths = [root / data["selection"], root / data["training"],
             root / data["root"] / "frames_v1.json", root / "configs/baselines/registry.json"]
    return dict(variant=variant, code_sha256=code,
                input_sha256={path.relative_to(root).as_posix(): sha256(path) for path in paths})


def validate_resume(out, variant, cfg, contract):
    out = Path(out)
    last = out / "checkpoints/last.pt"
    if not last.is_file():
        raise ValueError("resume requires checkpoints/last.pt; refusing to restart in an old run")
    saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    if saved.get("variant") != variant or receipt.get("config", {}).get("variant") != variant:
        raise ValueError("resume variant mismatch")
    if saved.get("config") != cfg:
        raise ValueError("resume needs the identical configuration")
    if receipt.get("resume_contract") != contract:
        raise ValueError("resume code/data/baseline hashes differ or are missing")
    return receipt


def validate_checkpoint(checkpoint, variant, contract, stop):
    if checkpoint.get("variant") != variant or checkpoint.get("resume_contract") != contract:
        raise ValueError("checkpoint variant/code/data identity mismatch")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= stop:
        raise ValueError("checkpoint step is outside the requested run")
    return step


def verified_model_version(variant):
    """Verify the actual upstream checkout and every weight before model load."""
    from baselines.completion import registry, source_dir, verify_weights
    from scripts.setup_baselines import git as source_git
    spec = registry()["models"]["omnidc"]
    source = source_dir("omnidc")
    head = source_git(source, "rev-parse", "HEAD")
    status = source_git(source, "status", "--porcelain=v1", "--untracked-files=no")
    if head != spec["commit"]:
        raise ValueError(f"OMNI-DC source commit {head} differs from pinned {spec['commit']}")
    if status:
        raise ValueError("OMNI-DC tracked source has local modifications")
    return dict(name=spec["label"], key="omnidc", variant=variant, registry_entry=spec,
                source=str(source), source_commit=head, source_status=status,
                weights=verify_weights("omnidc"))


def record_training_frames(out, frames, splits, min_mirror, seed):
    """Persist the actual filtered ordered frame pool, rather than just counts."""
    from collections import Counter
    if not frames:
        raise ValueError("training frame selection is empty")
    scenes = {}
    for frame in frames:
        scene, split = frame["scene"], frame["split"]
        if split not in splits:
            raise ValueError("training frame split differs from requested splits")
        if scene in scenes and scenes[scene]["split"] != split:
            raise ValueError("training scene occurs in multiple splits")
        entry = scenes.setdefault(scene, dict(split=split, frame_count=0))
        entry["frame_count"] += 1
    counts = dict(frame_count=len(frames), scene_count=len(scenes),
                  split_counts=dict(Counter(f["split"] for f in frames)),
                  role_counts=dict(Counter(f["role"] for f in frames)))
    manifest = dict(schema="trustprompt.omnidc.training_frames.v1", requested_splits=list(splits),
                    min_mirror_share=float(min_mirror), seed=int(seed), **counts,
                    scenes={key: scenes[key] for key in sorted(scenes)}, frames=frames,
                    frame_order="filtered source order; sampler permutes with default_rng(seed).permutation(N)")
    path = Path(out) / "training_frames.json"
    write_json(path, manifest)
    return dict(manifest=path.name, sha256=sha256(path), **counts,
                scene_splits={key: scenes[key]["split"] for key in sorted(scenes)})


def save_checkpoint(last, model, opt, step, variant, contract):
    """Atomic replacement retains the previous valid checkpoint on save failure."""
    import numpy as np
    import torch
    import shutil
    payload = dict(model=trainable_state(model), optimizer=opt.state_dict(), step=step, variant=variant,
                   resume_contract=contract,
                   rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                            numpy=np.random.get_state(), python=random.getstate()))
    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(v) for v in value)
        return 0
    required = int(tensor_bytes(payload) * 1.1) + 64 * 2 ** 20
    if shutil.disk_usage(last.parent).free < required:
        raise OSError(f"insufficient checkpoint disk space: need {required} free bytes for atomic temp file")
    tmp = last.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, last)


def lr_at(step, total, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    t = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    lo = cfg["final_lr_ratio"]
    return cfg["lr"] * (lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def build(variant, cfg):
    import torch
    from baselines.completion import build as build_baseline
    from tp.omnidc.gate import Gate
    from tp.omnidc.losses import FineTuneLoss
    from tp.omnidc.model import GatedOMNIDC
    base = build_baseline("omnidc", device="cuda")
    for p in base.net.depth_module.parameters():         # Depth Anything V2 stays frozen, as in OMNI-DC
        p.requires_grad_(False)
    v = cfg["variants"][variant]
    model = GatedOMNIDC(base.net, Gate() if v["gate"] else None, v.get("paths", ())).cuda()
    base.net.args.max_depth = cfg["train"]["max_depth"]
    loss = FineTuneLoss(base.net.args, variant, cfg["loss"]["gate_weight"], v.get("label")).cuda()
    return model, loss, base


def freeze_batchnorm(model):
    import torch
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    model.net.depth_module.eval()


def trainable_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith(FROZEN)}


def load_trainable(model, state):
    """Load a checkpoint written with trainable_state into a model built by build()."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(FROZEN) for k in missing):
        raise KeyError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")


def grad_norms(model):
    """Gradient norms of the parts the fine-tuning must reach (logged with every log line)."""
    norm = lambda ps: float(sum(float(p.grad.norm()) ** 2 for p in ps if p.grad is not None) ** 0.5)
    bb = model.net.backbone
    out = dict(grad_confidence_head=norm(bb.cfi_dec0.parameters()))
    if model.gate is not None:
        out["grad_gate"] = norm(model.gate.parameters())
    if "input" in model.paths:
        w = bb.conv1_dep[0].weight
        out["grad_gate_channel"] = float(w.grad[:, -1].norm()) if w.grad is not None else 0.0
    return out


def main():
    invocation_t0 = time.perf_counter()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tp.omnidc.data import HypersimTraining, training_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot-minutes", type=float, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", type=int, default=None, metavar="STEPS")
    a = ap.parse_args()
    clock = pilot_clock(a.pilot_minutes, invocation_t0)
    if a.smoke is not None and a.smoke <= 0:
        ap.error("smoke must be a positive step count")
    config_path = (ROOT / a.config) if a.config else CONFIG
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "variants" in cfg and a.variant not in cfg["variants"]:
        ap.error(f"variant must be one of {sorted(cfg['variants'])} in {config_path.name}")
    if a.smoke:
        cfg["train"].update(SMOKE_TRAIN); cfg["data"]["crop"] = SMOKE_CROP
    t = cfg["train"]
    steps_total = t["samples"] // t["batch"]
    stop = min(a.smoke, steps_total) if a.smoke else steps_total
    contract = training_contract(cfg, a.variant)
    if a.resume:
        out = Path(a.resume) if os.path.isabs(a.resume) else ROOT / a.resume
        receipt = validate_resume(out, a.variant, cfg, contract)
        receipt.setdefault("resume_history", []).append(dict(command=[sys.executable, *sys.argv],
                                                             prior_stage=receipt.get("stage")))
    else:
        prefix = cfg.get("run_prefix", "p1")
        out = ROOT / "results" / f"{prefix}{'smoke' if a.smoke else ''}_{a.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, dict(variant=a.variant, config=cfg, config_sha256=sha256(config_path),
                                      config_path=config_path.relative_to(ROOT).as_posix(), pilot_minutes=a.pilot_minutes,
                                      smoke=bool(a.smoke)),
                            [sys.executable, "-m", "scripts.train_omnidc", *sys.argv[1:]], int(cfg["seed"]))
        receipt["resume_contract"] = contract
    (out / "checkpoints").mkdir(exist_ok=True)
    log = None
    try:
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        receipt["model_version"] = verified_model_version(a.variant)
        write_json(out / "receipt.json", receipt)
        model, loss_fn, base = build(a.variant, cfg)
        receipt["model_version"].update(load_info=base.load_info, deviations=base.deviations)
        receipt.update(model_deviations=base.deviations, gpu=torch.cuda.get_device_name(),
                       torch=torch.__version__, steps_total=steps_total)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
        start = 0
        last = out / "checkpoints" / "last.pt"
        if a.resume:
            ck = torch.load(last, map_location="cpu", weights_only=False)
            start = validate_checkpoint(ck, a.variant, contract, stop)
            load_trainable(model, ck["model"]); opt.load_state_dict(ck["optimizer"])
            torch.set_rng_state(ck["rng"]["torch"].cpu())   # OMNI-DC's PVT trains with stochastic depth (drop_path
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["rng"]["cuda"]])   # 0.1), drawn from the CUDA RNG
            np.random.set_state(ck["rng"]["numpy"]); random.setstate(ck["rng"]["python"])
            del ck
        d = cfg["data"]
        selection = json.loads((ROOT / d["selection"]).read_text(encoding="utf-8"))
        min_mirror = float(yaml.safe_load((ROOT / d["training"]).read_text(encoding="utf-8"))["min_mirror_share"])
        frames = training_frames(ROOT / d["root"], min_mirror, tuple(d["splits"]))
        receipt["training_data"] = record_training_frames(out, frames, d["splits"], min_mirror, cfg["seed"])
        inputs = cfg.get("variants", {}).get(a.variant, {}).get("inputs", "f0" if a.variant == "f0" else "f1")
        data = HypersimTraining(ROOT / d["root"], selection, frames, inputs, cfg["seed"], d["p_sparse"],
                                d.get("crop", (480, 640)))
        sampler = range(start * t["batch"], stop * t["batch"])
        loader = DataLoader(data, batch_size=t["batch"], sampler=sampler, num_workers=t["workers"],
                            persistent_workers=t["workers"] > 0, pin_memory=True, drop_last=True,
                            # Worker startup must not advance the restored model CPU RNG.
                            generator=torch.Generator().manual_seed(cfg["seed"]))
        receipt.update(frames=len(frames), frame_roles={r: sum(1 for f in frames if f["role"] == r) for r in ("mirror", "window", "clean")},
                       stage="training", started_step=start)
        write_json(out / "receipt.json", receipt)
        log = (out / "log.jsonl").open("a", encoding="utf-8")
        model.train(); freeze_batchnorm(model)
        t0 = time.perf_counter(); step = start; n_since = 0; saved_step = start if a.resume else None
        iterator = iter(loader) if not clock or time.perf_counter() < clock["train_deadline"] else None
        while step < stop:
            if clock and time.perf_counter() >= clock["train_deadline"]:
                break
            batch = next(iterator)
            if clock and time.perf_counter() >= clock["train_deadline"]:
                break
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps_total, t)
            opt.zero_grad(set_to_none=True)
            out_ = model(batch)
            total, terms = loss_fn(batch, out_)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: {terms}")
            total.backward()
            logging = step + 1 == stop or (step + 1) % t["log_every"] == 0
            groups = grad_norms(model) if logging else {}
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, t["grad_clip"], error_if_nonfinite=True))
            opt.step()
            step += 1; n_since += t["batch"]
            if logging:
                el = time.perf_counter() - t0
                rec = dict(step=step, index=int(batch["index"][0]), lr=opt.param_groups[0]["lr"], loss=float(total),
                           grad_norm=gnorm, samples_per_s=n_since / el,
                           peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **terms, **groups)
                if "gate" in out_:
                    rec["gate_mean"] = float(out_["gate"].mean())
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if step % t["checkpoint_every"] == 0 or step == stop:
                save_checkpoint(last, model, opt, step, a.variant, contract); saved_step = step
            if t.get("snapshot_every") and step % t["snapshot_every"] == 0 and step < stop:
                snap = out / "checkpoints" / f"step_{step:06d}.pt"        # weights only, kept for trend checks
                torch.save(dict(model=trainable_state(model), step=step, variant=a.variant), snap.with_suffix(".tmp"))
                os.replace(snap.with_suffix(".tmp"), snap)
                receipt.setdefault("snapshots", {})[snap.name] = sha256(snap)
        if clock:
            train_seconds = time.perf_counter() - t0
            save_t0 = time.perf_counter()
            if saved_step != step:
                save_checkpoint(last, model, opt, step, a.variant, contract)
            save_seconds = time.perf_counter() - save_t0
            rate = n_since / train_seconds if n_since and train_seconds > 0 else None
            report = dict(steps=step - start, final_step=step, samples=n_since, samples_per_s=rate,
                          full_run_gpu_hours=t["samples"] / rate / 3600 if rate else None,
                          peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                          training_wall_seconds=train_seconds, startup_wall_seconds=t0 - invocation_t0,
                          checkpoint_save_seconds=save_seconds, process_wall_seconds=time.perf_counter() - invocation_t0,
                          budget_seconds=clock["budget_seconds"], cleanup_reserve_seconds=clock["cleanup_reserve_seconds"],
                          timing_definition="process wall budget; training throughput excludes startup and final save; not GPU kernel time",
                          last_checkpoint_step=step, last_checkpoint_bytes=last.stat().st_size)
            receipt.update(stage="pilot_complete" if rate else "pilot_no_training", pilot=report)
            print(json.dumps(report), flush=True)
            return
        if a.smoke:
            receipt.update(stage="smoke_complete", final_step=step, total_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30)
            return
        torch.save(dict(model=trainable_state(model), step=step, variant=a.variant), out / "checkpoints" / "final.pt")
        receipt.update(stage="training_complete", final_step=step, final_sha256=sha256(out / "checkpoints" / "final.pt"))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        if log is not None:
            log.close()
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
