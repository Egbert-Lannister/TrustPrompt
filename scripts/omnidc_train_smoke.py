"""Mainline P0-D2 (docs/plan_omnidc.md sec. 7, 14): does OMNI-DC + gate train at all on this machine?
(GPU, Windows, .venvs/baselines; nothing is kept but the report)

  python -m scripts.omnidc_train_smoke [--steps 3] [--crop 256 320]

1. Open gate: OMNI-DC v1.1 with the gate wrapper and g = 1 everywhere must give OMNI-DC's own prediction
   (zero-initialised gate channel; the DySPN logit transform is exact up to float error). Pass: max |difference| of
   the prediction < 1e-3 m on two Hypersim frames.
2. Training steps: two selected Hypersim mirror frames (data/source_cache/hypersim_v1) with the condition-C synthesis
   (tp.sim.hypersim_condition), random crops, batch 1, AdamW lr 1e-4 on the gate and OMNI-DC (Depth Anything V2
   frozen as in OMNI-DC, batch-norm statistics frozen). Loss = OMNI-DC's training loss (src/training_scripts/
   train_full.sh) without its SeqLaplace term, whose output-confidence head is not in the released v1.1 weights:
   1 SeqGradL1 + 1 ConfInput + 1 SeqL1 + 2 GradMatchingScale; + cross-entropy of the gate against the synthesis
   labels. Pass: every loss finite, non-zero gradients reach the gate, the new gate input
   channel and OMNI-DC's input-confidence head, peak memory below 95 % of the card.
Writes results/omnidc_train_smoke_<time>/ and docs/omnidc_train_smoke.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
import json
import sys
import time

import numpy as np

from baselines.completion import _sys_path, build, source_dir
from tp.data.hypersim import load_frame
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.sim.hypersim_condition import condition

DATA = ROOT / "data" / "source_cache" / "hypersim_v1"
SELECTION = ROOT / "configs" / "hypersim" / "selection_v1.json"
TRAIN_LOSS = dict(loss="1.0*SeqGradL1+1.0*ConfInput+1.0*SeqL1+2.0*GradMatchingScale",   # train_full.sh minus
                  intermediate_loss_weight=1.0)                                        # SeqLaplace (no head in v1.1)
SEED = 20261020


def frames(n=2):
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    rows = [f for f in json.loads((DATA / "frames_v1.json").read_text(encoding="utf-8"))["frames"] if f["role"] == "mirror"]
    picked = []
    for f in rows:
        d = DATA / f["scene"] / "images" / f"scene_{f['camera']}_geometry_hdf5" / f"frame.{f['frame']:04d}.depth_meters.hdf5"
        if d.is_file():
            picked.append(load_frame(DATA, f["scene"], f["camera"], f["frame"], sel["scenes"][f["scene"]]["camera"]["M_cam_from_uv"]))
        if len(picked) == n:
            return picked
    raise FileNotFoundError("fewer than 2 downloaded mirror frames: run scripts/windows/step13_hypersim.ps1 first")


def to_sample(torch, frame, synth, crop, rng, device):
    import torchvision.transforms.functional as TF
    H, W = frame.depth.shape
    ch, cw = crop
    y0, x0 = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
    sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    rgb = TF.normalize(torch.from_numpy(frame.rgb[sl].transpose(2, 0, 1).copy()).float() / 255,
                       (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dep = np.where(synth["V_corrupt"][sl], synth["P_corrupt"][sl], 0).astype(np.float32)
    gt = np.where(frame.valid[sl], frame.depth[sl], 0).astype(np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None].to(device)
    return dict(rgb=rgb[None].to(device), dep=t(dep), gt=t(gt), K=torch.eye(3, device=device)[None], pattern=0), \
        torch.from_numpy(synth["labels"][sl].astype(np.int64))[None].to(device), torch.from_numpy(dep > 0)[None].to(device)


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--crop", type=int, nargs=2, default=[256, 320])
    a = ap.parse_args()
    out_dir = ROOT / "results" / f"omnidc_train_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out_dir, dict(steps=a.steps, crop=a.crop, train_loss=TRAIN_LOSS, seed=SEED),
                        [sys.executable, "-m", "scripts.omnidc_train_smoke", *sys.argv[1:]], SEED)
    try:
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
        base = build("omnidc", device="cuda")
        net = base.net
        from tp.omnidc.gate import Gate, gate_targets
        from tp.omnidc.gates import soft_gated
        from tp.omnidc.model import GatedOMNIDC
        fr = frames()
        synth = [condition(f, np.random.default_rng([SEED, i])) for i, f in enumerate(fr)]

        # 1. open gate == OMNI-DC (reference computed before the backbone gets its gate channel)
        crops = [to_sample(torch, f, s, a.crop, np.random.default_rng([SEED, 100 + i]), "cuda")[0]
                 for i, (f, s) in enumerate(zip(fr, synth))]
        net.eval()
        with torch.no_grad():
            refs = [net(c)["pred"] for c in crops]
        model = GatedOMNIDC(net, Gate()).cuda()
        open_diff = []
        with torch.no_grad():
            for c, ref in zip(crops, refs):
                with soft_gated(net, torch.ones_like(c["dep"]), model.rate, model.paths):
                    open_diff.append(float((net(c)["pred"] - ref).abs().max()))
        open_ok = max(open_diff) < 1e-3

        # 2. training steps
        omni_args = net.args
        for k, v in TRAIN_LOSS.items():
            setattr(omni_args, k, v)
        with _sys_path(source_dir("omnidc") / "src"):
            from loss.sequentialloss import SequentialLoss       # OMNI-DC's own training loss
        omni_loss = SequentialLoss(omni_args); omni_loss.cuda(0)
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        steps = []
        for step in range(a.steps):
            i = step % len(fr)
            sample, labels, valid = to_sample(torch, fr[i], synth[i], a.crop, rng, "cuda")
            torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
            opt.zero_grad()
            out = model(sample)
            l_omni, parts = omni_loss(sample, out)
            l_gate = torch.nn.functional.cross_entropy(out["gate_logits"], gate_targets(labels, valid), ignore_index=-100)
            (l_omni.sum() + l_gate).backward()
            norm = lambda ps: float(sum(p.grad.norm() ** 2 for p in ps if p.grad is not None) ** 0.5)
            grads = dict(gate=norm(model.gate.parameters()),
                         gate_input_channel=float(net.backbone.conv1_dep[0].weight.grad[:, -1].norm()),
                         confidence_head=norm(net.backbone.cfi_dec0.parameters()),
                         all=norm(params))
            opt.step(); torch.cuda.synchronize()
            steps.append(dict(step=step, frame=f"{fr[i].scene}/{fr[i].camera}/{fr[i].frame}", loss_omni=float(l_omni.sum()),
                              loss_parts=[float(x) for x in parts.flatten()], loss_gate=float(l_gate), grads=grads,
                              seconds=time.perf_counter() - t, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                              gate_mean=float(out["gate"].mean())))
            print(json.dumps(steps[-1]), flush=True)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        finite = all(np.isfinite([s["loss_omni"], s["loss_gate"], *s["grads"].values()]).all() for s in steps)
        reach = all(s["grads"][k] > 0 for s in steps for k in ("gate", "gate_input_channel", "confidence_head"))
        fits = max(s["peak_gb"] for s in steps) < 0.95 * total_gb
        summary = dict(open_gate_max_abs_diff=open_diff, open_gate_ok=open_ok, finite=finite, gradients_reach=reach,
                       fits_memory=fits, total_gb=total_gb, passed=bool(open_ok and finite and reach and fits))
        write_json(out_dir / "smoke.json", dict(summary=summary, steps=steps))
        lines = ["# OMNI-DC + gate: open-gate check and training steps (mainline P0-D2)", "",
                 f"Generated by `python -m scripts.omnidc_train_smoke` ({out_dir.name}); crop {a.crop}, batch 1.", "",
                 f"- open gate vs OMNI-DC, max |prediction difference| (m): {', '.join(f'{d:.2e}' for d in open_diff)} "
                 f"-> {'ok' if open_ok else 'NOT ok'}",
                 "", "| step | frame | OMNI-DC loss | gate CE | grad gate | grad gate channel | grad conf. head | s | peak GB |",
                 "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {s['step']} | {s['frame']} | {s['loss_omni']:.4f} | {s['loss_gate']:.4f} | {s['grads']['gate']:.2e} | "
                  f"{s['grads']['gate_input_channel']:.2e} | {s['grads']['confidence_head']:.2e} | {s['seconds']:.1f} | {s['peak_gb']:.2f} |"
                  for s in steps]
        lines += ["", f"- finite: {finite}; gradients reach gate / gate channel / confidence head: {reach}; "
                      f"fits {total_gb:.1f} GB: {fits}", f"- **{'passed' if summary['passed'] else 'NOT passed'}**", ""]
        (ROOT / "docs" / "omnidc_train_smoke.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="smoke_complete", summary=summary, smoke_sha256=sha256(out_dir / "smoke.json"))
        print("\n".join(lines))
        if not summary["passed"]:
            raise SystemExit(2)
    except BaseException as e:
        if not isinstance(e, SystemExit):
            receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out_dir / "receipt.json", receipt)


if __name__ == "__main__":
    main()
