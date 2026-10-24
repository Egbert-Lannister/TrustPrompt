"""Fine-tune OMNI-DC v1.1 for the mainline (P1: f0, f1; P2: f2 and ablations). GPU; meant for a rented 48 GB card.

  python -m scripts.train_omnidc --variant f1 [--pilot-minutes 60] [--resume results/p1_f1_<time>]

Config: configs/omnidc/p1.yaml (fixed before training). Data: tp/omnidc/data.py on the downloaded Hypersim frames
(only Hypersim's train scenes; no NYU / HAMMER image is used, so both stay zero-shot evaluations). Model:
tp/omnidc/model.GatedOMNIDC (gate only in f2). Loss: tp/omnidc/losses.FineTuneLoss. Depth Anything V2 stays frozen
as in OMNI-DC; batch-norm statistics are frozen (fine-tuning with small batches).
--pilot-minutes N  train for N minutes, then write the measured throughput and the GPU-hour estimate for the full
                   run (plan sec. 7: measure before committing the budget) and stop.
Writes results/p1_<variant>_<time>/: receipt, config, log.jsonl (every log_every steps), checkpoints/last.pt (every
checkpoint_every steps, atomic) and final.pt. --resume continues from last.pt with the same config and code.
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


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tp.omnidc.data import HypersimTraining, training_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["f0", "f1", "f2"])
    ap.add_argument("--pilot-minutes", type=float, default=None)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    t = cfg["train"]
    steps_total = t["samples"] // t["batch"]
    if a.resume:
        out = Path(a.resume) if os.path.isabs(a.resume) else ROOT / a.resume
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        if json.loads((out / "config.json").read_text(encoding="utf-8"))["config"] != cfg:
            raise ValueError("resume needs the identical configuration")
    else:
        out = ROOT / "results" / f"p1_{a.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, dict(variant=a.variant, config=cfg, config_sha256=sha256(CONFIG), pilot_minutes=a.pilot_minutes),
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
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = ck["step"]
        d = cfg["data"]
        selection = json.loads((ROOT / d["selection"]).read_text(encoding="utf-8"))
        min_mirror = float(yaml.safe_load((ROOT / d["training"]).read_text(encoding="utf-8"))["min_mirror_share"])
        frames = training_frames(ROOT / d["root"], min_mirror, tuple(d["splits"]))
        data = HypersimTraining(ROOT / d["root"], selection, frames, a.variant, cfg["seed"], d["p_sparse"])
        sampler = range(start * t["batch"], steps_total * t["batch"])
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
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, t["grad_clip"]))
            opt.step()
            step += 1; n_since += t["batch"]
            if step % t["log_every"] == 0 or step == steps_total:
                el = time.perf_counter() - t0
                rec = dict(step=step, lr=opt.param_groups[0]["lr"], loss=float(total), grad_norm=gnorm, samples_per_s=n_since / el,
                           peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **terms)
                if "gate" in out_:
                    rec["gate_mean"] = float(out_["gate"].mean())
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if step % t["checkpoint_every"] == 0 or step == steps_total:
                tmp = last.with_suffix(".tmp")
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step, variant=a.variant), tmp)
                os.replace(tmp, last)
            if a.pilot_minutes and time.perf_counter() - t0 > 60 * a.pilot_minutes:
                rate = n_since / (time.perf_counter() - t0)
                receipt.update(stage="pilot_complete", pilot=dict(steps=step - start, samples_per_s=rate,
                               full_run_gpu_hours=t["samples"] / rate / 3600, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
                print(json.dumps(receipt["pilot"]), flush=True)
                return
            if step >= steps_total:
                break
        torch.save(dict(model=model.state_dict(), step=step, variant=a.variant), out / "checkpoints" / "final.pt")
        receipt.update(stage="training_complete", final_step=step, final_sha256=sha256(out / "checkpoints" / "final.pt"))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
