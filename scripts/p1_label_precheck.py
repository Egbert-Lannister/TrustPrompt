"""P1 v2 laptop pre-check, evaluation (docs/plan_omnidc.md sec. 21). GPU, Windows, .venvs/baselines; development
data only, test never read; a mechanism check, not a depth result.

  python -m scripts.p1_label_precheck [--runs results/p1v2pre_f1_<time> results/p1v2pre_f1s_<time>]

Models: OMNI-DC v1.1 (R0) and the two pre-check runs (configs/omnidc/p1v2_precheck.yaml: P1's F1 recipe with the
two-sided cell label = control, and with the soft cell label = candidate; final.pt hash-checked against each run's
receipt). Default runs: the newest finished p1v2pre_f1_* and p1v2pre_f1s_*.
Populations as D1 / the P1 gate: NYU val + train-holdout (30 images); HAMMER dev, 3 frames per scene x sensor (90).
Per model: AUROC of 1 - input confidence for wrong vs correct readings, mean confidence on correct and on wrong
readings (main scope and full image; scripts/omnidc_confidence), and the pixel-pooled AbsRel of the prediction on
clean pixels (tp.eval.regions) inside / outside the main scope and on C.
Reading, fixed before the first run (NYU, full image unless noted):
  K1  the control reproduces the problem: mean confidence on correct readings <= 0.85 (R0 0.89, F0 0.99, P1's F1 0.62)
  K2  the soft label avoids it: mean confidence on correct readings >= 0.92
  K3  the soft label keeps the detection: main-scope AUROC >= 0.80 (P1's F1 0.91, R0 0.52)
  K1 and K2 and K3 -> the label is the mechanism; write P1 v2 with the soft label (training is the user's decision).
  not K1 -> inconclusive at this length (the control has not drifted yet); K1 but not K2 or K3 -> the soft label is
  not enough; rethink before paying.
Writes results/p1v2_precheck_<time>/ and docs/omnidc_p1v2_precheck.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from scripts.cache_completion import model_input, select_records
from scripts.omnidc_confidence import Tally, full_res
from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json

TAU = 0.05
SETS = {"nyu": dict(processed="data/processed/mirror3d_dev_v1", scope="holdout", per_group=None),
        "hammer": dict(processed="data/processed/hammer_dev_v1", scope="all", per_group=3)}
K = dict(K1_max_control_conf_correct=0.85, K2_min_soft_conf_correct=0.92, K3_min_soft_auroc_main=0.80)


def newest(variant):
    done = []
    for d in sorted((ROOT / "results").glob(f"p1v2pre_{variant}_*")):
        r = d / "receipt.json"
        if r.is_file() and json.loads(r.read_text(encoding="utf-8")).get("stage") == "training_complete":
            done.append(d)
    if not done:
        raise FileNotFoundError(f"no finished results/p1v2pre_{variant}_* run")
    return done[-1]


def load_run(model, run):
    import torch
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    final = run / "checkpoints" / "final.pt"
    if receipt.get("stage") != "training_complete" or sha256(final) != receipt.get("final_sha256"):
        raise ValueError(f"{run}: not finished or final.pt differs from its receipt")
    state = torch.load(final, map_location=model.device, weights_only=False)["model"]
    net_state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    if len(net_state) != len(state):
        raise KeyError(f"{run}: unexpected non-network keys (a gated run?)")
    missing, unexpected = model.net.load_state_dict(net_state, strict=False)
    if unexpected or any(not k.startswith("depth_module.") for k in missing):
        raise KeyError(f"{run}: checkpoint does not fit OMNI-DC: {unexpected[:3]} {missing[:3]}")
    return dict(run=run.relative_to(ROOT).as_posix(), variant=receipt["config"]["variant"],
                label=receipt["config"]["config"]["variants"][receipt["config"]["variant"]].get("label"),
                final_sha256=receipt["final_sha256"], step=receipt.get("final_step"))


class Err:
    def __init__(self):
        self.s = {}; self.n = {}

    def add(self, key, pred, gt, where):
        if where.any():
            e = np.abs(pred[where] - gt[where]) / gt[where]
            self.s[key] = self.s.get(key, 0.0) + float(e.sum()); self.n[key] = self.n.get(key, 0) + int(e.size)

    def summary(self):
        return {k: self.s[k] / self.n[k] for k in self.s}


def evaluate(model):
    out = {}
    for ds, d in SETS.items():
        processed = ROOT / d["processed"]
        recs = per_group_subsample(select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"],
                                                  {"train", "val"}, d["scope"]), d["per_group"])
        tallies, err = {"main": Tally(), "full": Tally()}, Err()
        for r in recs:
            path = processed / r["path"]
            if sha256(path) != r["sha256"]:
                raise ValueError(f"checksum mismatch {path}")
            s = load_sample(path)
            depth, _, _ = model_input(s, "raw", tau=TAU)
            o, _ = model.outputs(s.rgb, depth)
            H, W = depth.shape
            conf = full_res(o["confidence_input"], depth.shape, model.args.backbone_output_downsample_rate)
            pred = o["pred"][0, 0, :H, :W].float().cpu().numpy().astype(np.float64)
            reg = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=TAU, band=5)
            scope = focus_scope(s)[1]
            both = s.prompt_valid & s.gt_valid
            tallies["main"].add(conf, conf, reg["C"], both & scope)        # T column unused here (score = conf)
            tallies["full"].add(conf, conf, reg["C"], both)
            ok = np.isfinite(pred) & (pred > 0)
            gt = s.gt.astype(np.float64)
            err.add("clean_in_scope", pred, gt, reg["clean"] & scope & ok)
            err.add("clean_outside_scope", pred, gt, reg["clean"] & ~scope & ok)
            err.add("C_in_scope", pred, gt, reg["C"] & scope & ok)
        out[ds] = dict(images=len(recs), absrel=err.summary(),
                       **{k: {m: v for m, v in t.summary().items() if m != "auroc_T"} for k, t in tallies.items()})
    return out


def main():
    import torch
    from baselines.completion import build
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs=2, default=None, metavar=("CONTROL", "SOFT"))
    a = ap.parse_args()
    runs = [Path(x) if os.path.isabs(x) else ROOT / x for x in a.runs] if a.runs else [newest("f1"), newest("f1s")]
    out = ROOT / "results" / f"p1v2_precheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, sets=SETS, reading=K, runs=[r.relative_to(ROOT).as_posix() for r in runs]),
                        [sys.executable, "-m", "scripts.p1_label_precheck", *sys.argv[1:]], 20261025)
    try:
        res, models = {}, {}
        for name, run in (("R0", None), ("control (two-sided)", runs[0]), ("soft", runs[1])):
            model = build("omnidc", device="cuda")
            models[name] = dict(run=None, label="OMNI-DC v1.1") if run is None else load_run(model, run)
            with torch.no_grad():
                res[name] = evaluate(model)
            print(name, json.dumps(res[name]["nyu"]["full"]), flush=True)
            del model; torch.cuda.empty_cache()
        if models["control (two-sided)"]["label"] != "two_sided" or models["soft"]["label"] != "soft":
            raise ValueError(f"runs have labels {models}; expected two_sided then soft")
        c, s_ = res["control (two-sided)"]["nyu"], res["soft"]["nyu"]
        checks = dict(K1=c["full"]["mean_conf_correct"] <= K["K1_max_control_conf_correct"],
                      K2=s_["full"]["mean_conf_correct"] >= K["K2_min_soft_conf_correct"],
                      K3=s_["main"]["auroc_omnidc"] >= K["K3_min_soft_auroc_main"])
        verdict = ("label is the mechanism: write P1 v2 with the soft label" if all(checks.values()) else
                   "inconclusive: the control has not drifted at this length" if not checks["K1"] else
                   "the soft label is not enough: rethink before paying")
        write_json(out / "precheck.json", dict(models=models, results=res, checks=checks, verdict=verdict))
        fmt = lambda v, p=3: "NA" if v is None else f"{v:.{p}f}"
        lines = ["# P1 v2 laptop pre-check: soft vs two-sided ConfInput cell label", "",
                 f"Generated by `python -m scripts.p1_label_precheck` ({out.name}). Runs: "
                 + ", ".join(f"{k} = {v['run'] or v['label']}" for k, v in models.items())
                 + ". 256 x 320 crops, batch 1, 1500 steps (configs/omnidc/p1v2_precheck.yaml); development data only, "
                 "test never read; a mechanism check, not a depth result.", "",
                 "| model | dataset | images | AUROC main | AUROC full | conf correct main | conf correct full | conf C main | "
                 "AbsRel clean in scope | AbsRel clean outside | AbsRel C |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in res.items():
            for ds, x in r.items():
                ab = x["absrel"]
                lines.append(f"| {name} | {ds} | {x['images']} | {fmt(x['main']['auroc_omnidc'])} | {fmt(x['full']['auroc_omnidc'])} | "
                             f"{fmt(x['main']['mean_conf_correct'])} | {fmt(x['full']['mean_conf_correct'])} | "
                             f"{fmt(x['main']['mean_conf_C'])} | {fmt(ab.get('clean_in_scope'), 4)} | "
                             f"{fmt(ab.get('clean_outside_scope'), 4)} | {fmt(ab.get('C_in_scope'), 4)} |")
        lines += ["", "Reading fixed before the run (NYU): "
                  f"K1 control conf on correct (full) <= {K['K1_max_control_conf_correct']}: {checks['K1']}; "
                  f"K2 soft conf on correct (full) >= {K['K2_min_soft_conf_correct']}: {checks['K2']}; "
                  f"K3 soft AUROC (main) >= {K['K3_min_soft_auroc_main']}: {checks['K3']}", "", f"**{verdict}**", ""]
        (ROOT / "docs" / "omnidc_p1v2_precheck.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", checks=checks, verdict=verdict,
                       precheck_sha256=sha256(out / "precheck.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
