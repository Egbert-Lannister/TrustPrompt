"""Mainline P1, laptop check before renting a GPU (docs/plan_omnidc.md sec. 17): does scripts/train_omnidc.py run
end to end? (GPU, Windows, .venvs/baselines; about 10 minutes; nothing from it is used for any result)

  python -m scripts.omnidc_p1_smoke

Runs the training script itself, as separate processes, in its --smoke mode (same data pipeline, model, loss and
optimiser as P1; 256 x 320 crops, batch 1, 2 data workers, Hypersim train frames on this machine):
  f0, f2          10 steps each
  f1              10 steps; resumed from its step-10 checkpoint to step 15; and once without a break to step 15
  f1 pilot        --pilot-minutes 0.5 (the throughput report a rented card would give; its numbers mean nothing here)
Checks, fixed before the first run:
  S1  every run exits 0 with the expected stage (smoke_complete / pilot_complete)
  S2  every logged loss term and gradient norm is finite
  S3  gradients reach OMNI-DC's input-confidence head in every run, and the gate and its input channel in f2
  S4  peak GPU memory below 95 % of the card
  S5  resume is exact up to GPU nondeterminism: steps 11-15 of the resumed f1 run use the same samples as the
      unbroken run, and their losses differ by < 1 % (relative)
  S6  the pilot writes a positive throughput and a GPU-hour estimate
Writes results/omnidc_p1_smoke_<time>/ (receipt, smoke.json, one log per run) and docs/omnidc_p1_smoke.md. The runs'
own folders (results/p1smoke_*) are kept without their checkpoints, which are removed at the end (about 1 GB each).
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from datetime import datetime
import json
import math
import subprocess
import sys
import time

from tp.provenance import ROOT, sha256, start_run, write_json

RUNS = [  # name, variant, extra arguments, expected stage
    ("f0", "f0", ["--smoke", "10"], "smoke_complete"),
    ("f2", "f2", ["--smoke", "10"], "smoke_complete"),
    ("f1", "f1", ["--smoke", "10"], "smoke_complete"),
    ("f1_resumed", "f1", ["--smoke", "15", "--resume", "{f1}"], "smoke_complete"),
    ("f1_unbroken", "f1", ["--smoke", "15"], "smoke_complete"),
    ("f1_pilot", "f1", ["--smoke", "100000", "--pilot-minutes", "0.5"], "pilot_complete"),
]
RESUME_TOL = 0.01


def newest(variant, before):
    dirs = sorted(p for p in (ROOT / "results").glob(f"p1smoke_{variant}_*") if p.name not in before)
    return dirs[-1] if dirs else None


def run(variant, args, log_path):
    before = {p.name for p in (ROOT / "results").glob(f"p1smoke_{variant}_*")}
    t = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "-m", "scripts.train_omnidc", "--variant", variant, *args],
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    resume = "--resume" in args
    out = ROOT / args[args.index("--resume") + 1] if resume else newest(variant, before)
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8")) if out and (out / "receipt.json").is_file() else {}
    log = [json.loads(l) for l in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()] if out and (out / "log.jsonl").is_file() else []
    return dict(returncode=rc, dir=out.relative_to(ROOT).as_posix() if out else None, stage=receipt.get("stage"),
                seconds=time.perf_counter() - t, receipt=receipt, log=log)


def main():
    out_dir = ROOT / "results" / f"omnidc_p1_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out_dir, dict(runs=RUNS, resume_tolerance=RESUME_TOL),
                        [sys.executable, "-m", "scripts.omnidc_p1_smoke", *sys.argv[1:]], 0)
    try:
        res = {}
        for name, variant, args, expect in RUNS:
            args = [a.format(f1=res["f1"]["dir"]) if "{" in a else a for a in args]
            print(f"=== {name}: train_omnidc --variant {variant} {' '.join(args)}", flush=True)
            res[name] = r = run(variant, args, out_dir / f"{name}.log")
            r["expected_stage"] = expect
            print(f"    exit {r['returncode']}, stage {r['stage']}, {r['seconds']:.0f} s, {r['dir']}", flush=True)
            if r["returncode"] != 0 and name == "f1":
                break                                       # the resume runs need f1's checkpoint
        total_gb = next((r["receipt"].get("total_gb") for r in res.values() if r["receipt"].get("total_gb")), None)
        num = lambda rec: [v for k, v in rec.items() if isinstance(v, float)]
        checks = {}
        checks["S1"] = all(r["returncode"] == 0 and r["stage"] == r["expected_stage"] for r in res.values()) and len(res) == len(RUNS)
        checks["S2"] = all(r["log"] and all(math.isfinite(v) for rec in r["log"] for v in num(rec)) for r in res.values())
        def reach(r, keys):
            return bool(r["log"]) and all(rec.get(k, 0) > 0 for rec in r["log"] for k in keys)
        checks["S3"] = (all(reach(r, ["grad_confidence_head"]) for r in res.values())
                        and reach(res.get("f2", {"log": []}), ["grad_gate", "grad_gate_channel"]))
        peak = max((rec["peak_gb"] for r in res.values() for rec in r["log"]), default=None)
        checks["S4"] = peak is not None and total_gb is not None and peak < 0.95 * total_gb
        resumed = [rec for rec in res.get("f1_resumed", {"log": []})["log"] if rec["step"] > 10]
        unbroken = {rec["step"]: rec for rec in res.get("f1_unbroken", {"log": []})["log"]}
        pairs = [(rec, unbroken.get(rec["step"])) for rec in resumed]
        resume_diff = max((abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-12) for a, b in pairs if b), default=None)
        checks["S5"] = (len(pairs) == 5 and all(b and a["index"] == b["index"] for a, b in pairs)
                        and resume_diff is not None and resume_diff < RESUME_TOL)
        pilot = res.get("f1_pilot", {"receipt": {}})["receipt"].get("pilot") or {}
        checks["S6"] = pilot.get("samples_per_s", 0) > 0 and pilot.get("full_run_gpu_hours", 0) > 0
        summary = dict(checks=checks, passed=all(checks.values()), peak_gb=peak, total_gb=total_gb,
                       resume_max_rel_loss_diff=resume_diff, pilot=pilot,
                       frames=res.get("f1", {"receipt": {}})["receipt"].get("frames"))
        for r in res.values():
            r.pop("receipt", None)
        freed = 0                                            # the smoke checkpoints are not results: free the disk
        for r in res.values():
            ck = ROOT / r["dir"] / "checkpoints" / "last.pt" if r["dir"] else None
            if ck and ck.is_file():
                freed += ck.stat().st_size; ck.unlink()
        summary["checkpoint_gb_removed"] = freed / 2 ** 30
        write_json(out_dir / "smoke.json", dict(summary=summary, runs=res))
        fmt = lambda v, p=3: "NA" if v is None else f"{v:.{p}g}"
        lines = ["# P1 training script: laptop smoke test", "",
                 f"Generated by `python -m scripts.omnidc_p1_smoke` ({out_dir.name}). `scripts/train_omnidc.py --smoke`: "
                 "the P1 data pipeline, model, loss and optimiser with 256 x 320 crops, batch 1. Checks only; no result "
                 "of any run is used.", "",
                 "| run | exit | stage | steps logged | last loss | grad conf. head | grad gate | grad gate channel | peak GB | s |",
                 "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in res.items():
            last = r["log"][-1] if r["log"] else {}
            lines.append(f"| {name} | {r['returncode']} | {r['stage']} | {len(r['log'])} | {fmt(last.get('loss'))} | "
                         f"{fmt(last.get('grad_confidence_head'))} | {fmt(last.get('grad_gate'))} | "
                         f"{fmt(last.get('grad_gate_channel'))} | {fmt(last.get('peak_gb'))} | {r['seconds']:.0f} |")
        lines += ["", f"- training frames found: {summary['frames']}; peak memory {fmt(peak)} of {fmt(total_gb)} GB",
                  f"- resume: steps 11-15 resumed vs unbroken, max relative loss difference {fmt(resume_diff)} "
                  f"(must be < {RESUME_TOL})",
                  f"- pilot report (laptop, small crops; not an estimate for the rented card): {json.dumps(pilot)}",
                  "- checks fixed before running: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in checks.items()),
                  f"- **{'passed: the training script runs end to end' if summary['passed'] else 'NOT passed: fix before renting'}**",
                  f"- logs of each run: results/{out_dir.name}/<run>.log", ""]
        (ROOT / "docs" / "omnidc_p1_smoke.md").write_text("\n".join(lines), encoding="utf-8")
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
