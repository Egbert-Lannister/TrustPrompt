"""Fine-tune OMNI-DC v1.1 for the mainline (P1: f0, f1; P2: f2 and ablations). GPU; meant for a rented 48 GB card.

  python -m scripts.train_omnidc --variant f1 [--pilot-minutes 60] [--resume results/p1_f1_<time>] [--smoke STEPS]

Config: configs/omnidc/p1.yaml (fixed before training). Data: tp/omnidc/data.py on the downloaded Hypersim frames
(only Hypersim's train scenes; no NYU / HAMMER image is used, so both stay zero-shot evaluations). Model:
tp/omnidc/model.GatedOMNIDC (gate only in f2). Loss: tp/omnidc/losses.FineTuneLoss. Depth Anything V2 stays frozen
as in OMNI-DC; batch-norm statistics are frozen (fine-tuning with small batches).
--pilot-minutes N  train for N minutes, then write the measured throughput and the GPU-hour estimate for the full
                   run (plan sec. 7: measure before committing the budget) and stop.
Writes results/p1_<variant>_<time>/: receipt, config, log.jsonl (every log_every steps), checkpoints/last.pt (every
checkpoint_every steps, atomic) and final.pt, both without the frozen Depth Anything V2 weights (load_trainable puts
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
import sys
import time
from pathlib import Path

import yaml

from tp.provenance import ROOT, sha256, start_run, write_json

CONFIG = ROOT / "configs" / "omnidc" / "p1.yaml"
SMOKE_TRAIN = dict(batch=1, workers=2, log_every=1, checkpoint_every=10)
FROZEN = "net.depth_module."                              # Depth Anything V2: frozen, pinned weights, not saved
SMOKE_CROP = [256, 320]


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
    loss = FineTuneLoss(base.net.args, variant, cfg["loss"]["gate_weight"]).cuda()
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
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tp.omnidc.data import HypersimTraining, training_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["f0", "f1", "f2"])
    ap.add_argument("--pilot-minutes", type=float, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", type=int, default=None, metavar="STEPS")
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if a.smoke:
        cfg["train"].update(SMOKE_TRAIN); cfg["data"]["crop"] = SMOKE_CROP
    t = cfg["train"]
    steps_total = t["samples"] // t["batch"]
    stop = min(a.smoke, steps_total) if a.smoke else steps_total
    if a.resume:
        out = Path(a.resume) if os.path.isabs(a.resume) else ROOT / a.resume
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        if json.loads((out / "config.json").read_text(encoding="utf-8"))["config"] != cfg:
            raise ValueError("resume needs the identical configuration")
    else:
        out = ROOT / "results" / f"p1{'smoke' if a.smoke else ''}_{a.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, dict(variant=a.variant, config=cfg, config_sha256=sha256(CONFIG), pilot_minutes=a.pilot_minutes,
                                      smoke=bool(a.smoke)),
                            [sys.executable, "-m", "scripts.train_omnidc", *sys.argv[1:]], int(cfg["seed"]))
    (out / "checkpoints").mkdir(exist_ok=True)
    try:
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        model, loss_fn, base = build(a.variant, cfg)
        receipt.update(model_deviations=base.deviations, gpu=torch.cuda.get_device_name(),
                       torch=torch.__version__, steps_total=steps_total)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
        start = 0
        last = out / "checkpoints" / "last.pt"
        if a.resume and last.is_file():
            ck = torch.load(last, map_location="cuda", weights_only=False)
            load_trainable(model, ck["model"]); opt.load_state_dict(ck["optimizer"]); start = ck["step"]
        d = cfg["data"]
        selection = json.loads((ROOT / d["selection"]).read_text(encoding="utf-8"))
        min_mirror = float(yaml.safe_load((ROOT / d["training"]).read_text(encoding="utf-8"))["min_mirror_share"])
        frames = training_frames(ROOT / d["root"], min_mirror, tuple(d["splits"]))
        data = HypersimTraining(ROOT / d["root"], selection, frames, a.variant, cfg["seed"], d["p_sparse"],
                                d.get("crop", (480, 640)))
        sampler = range(start * t["batch"], stop * t["batch"])
        loader = DataLoader(data, batch_size=t["batch"], sampler=sampler, num_workers=t["workers"],
                            persistent_workers=t["workers"] > 0, pin_memory=True, drop_last=True)
        receipt.update(frames=len(frames), frame_roles={r: sum(1 for f in frames if f["role"] == r) for r in ("mirror", "window", "clean")},
                       stage="training", started_step=start)
        write_json(out / "receipt.json", receipt)
        log = (out / "log.jsonl").open("a", encoding="utf-8")
        model.train(); freeze_batchnorm(model)
        t0 = time.perf_counter(); step = start; n_since = 0
        for batch in loader:
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
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, t["grad_clip"]))
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
                tmp = last.with_suffix(".tmp")
                torch.save(dict(model=trainable_state(model), optimizer=opt.state_dict(), step=step, variant=a.variant), tmp)
                os.replace(tmp, last)
            if a.pilot_minutes and time.perf_counter() - t0 > 60 * a.pilot_minutes:
                rate = n_since / (time.perf_counter() - t0)
                receipt.update(stage="pilot_complete", pilot=dict(steps=step - start, samples_per_s=rate,
                               full_run_gpu_hours=t["samples"] / rate / 3600, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
                print(json.dumps(receipt["pilot"]), flush=True)
                return
            if step >= stop:
                break
        if a.smoke:
            receipt.update(stage="smoke_complete", final_step=step, total_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30)
            return
        torch.save(dict(model=trainable_state(model), step=step, variant=a.variant), out / "checkpoints" / "final.pt")
        receipt.update(stage="training_complete", final_step=step, final_sha256=sha256(out / "checkpoints" / "final.pt"))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
