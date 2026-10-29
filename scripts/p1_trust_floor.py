"""F1+ : P1's F1 with a conformally calibrated trust floor at inference (docs/plan_omnidc.md sec. 24).
(GPU, Windows, .venvs/baselines; no training; development data only, test never read)

  python -m scripts.p1_trust_floor            # calibrate, cache F1+ on the gate's populations, evaluate
  python -m scripts.p1_gate --spec configs/omnidc/p1plus_gate_v1.yaml        # then the unchanged G1-G3

Rule. F1's input confidence c (OMNI-DC's 1/4-resolution cells, the weight of a reading in the depth integration) is
kept only where it is low: c' = c if c < lambda, else 1. Everything else in F1 is unchanged.
Risk controlled. Per image, the false-distrust rate FD(lambda) = share of its correct readings (|P - GT| / GT <= tau,
whole image) whose cell has c < lambda; it lies in [0, 1] and does not decrease with lambda. Conformal risk control
(Angelopoulos et al. 2022, images as exchangeable units): lambda_hat = the largest grid value with
n/(n+1) * mean FD(lambda) + 1/(n+1) <= alpha, alpha = 0.1 (method v1's level), so E[FD] <= 0.1 on a new image;
no admissible value -> lambda = 0 (full trust everywhere).
Calibration data, never the evaluated images:
  NYU     the train-split scenes outside the inner holdout (the gate evaluates val + holdout scenes)
  HAMMER  leave-one-scene-out: each of the 10 dev scenes gets the lambda calibrated on the other 9 (all frames)
Outputs, in the format of scripts/cache_omnidc_finetuned (float32, invalid values kept), so tp.eval.run_eval and
scripts/p1_gate read them as they read F0 / F1: cache/p1_f1plus_{nyu,hammer}_dev/ with evaluation_{val,train}/.
Also (descriptive, not part of any decision): the lambdas, the false-distrust rate reached on the evaluated images, and
on NYU a sweep alpha in {0.05, 0.2} (pixel-pooled AbsRel). Writes results/p1_trust_floor_<time>/ and
docs/omnidc_p1_trust_floor.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from scripts.cache_completion import model_input, select_records, store
from scripts.cache_omnidc_finetuned import IDENTITY, ROLE, evaluation_configs
from scripts.omnidc_confidence import full_res
from scripts.p1_coupling import override_confidence
from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.trust.data import scene_inner_holdout

TAU, ALPHA = 0.05, 0.1
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
SWEEP = (0.05, 0.2)
SETS = {"nyu": dict(manifest="data/processed/mirror3d_dev_v1/manifest.json", eval_scope="holdout"),
        "hammer": dict(manifest="data/processed/hammer_dev_v1/manifest.json", eval_scope="all")}
JOB = ROOT / "results" / "autodl_full_20261024" / "mirror" / "job" / "operator.json"
VARIANT = "f1plus"                       # cache model name OMNI-DC-v1.1-F1PLUS (scripts.cache_omnidc_finetuned)


def crc_max_lambda(losses, alpha, grid=GRID):
    """losses (n, len(grid)), non-decreasing in lambda. Largest lambda meeting the CRC bound (0.0 if none)."""
    n = losses.shape[0]
    bound = n / (n + 1) * losses.mean(0) + 1.0 / (n + 1)
    ok = np.flatnonzero(bound <= alpha)
    return (float(grid[ok[-1]]), float(bound[ok[-1]])) if ok.size else (0.0, float(bound[0]))


def false_distrust_curve(conf_full, s, grid=GRID):
    """Per-lambda share of the image's correct readings whose cell has c < lambda (0 if it has none)."""
    both = s.prompt_valid & s.gt_valid
    correct = both & (np.abs(s.prompt - s.gt) <= TAU * s.gt)
    c = conf_full[correct]
    return np.array([(c < lam).mean() if c.size else 0.0 for lam in grid]), int(correct.sum())


def floor_fn(lam):
    import torch
    return lambda c: torch.where(c < lam, c, torch.ones_like(c))


def manifest_records(ds):
    path = ROOT / SETS[ds]["manifest"]
    return path, json.loads(path.read_text(encoding="utf-8"))


def load(path, r):
    if sha256(path) != r["sha256"]:
        raise ValueError(f"checksum mismatch {path}")
    return load_sample(path)


def conf_and_pred(model, s, lam=None):
    import torch
    depth, _, pinfo = model_input(s, "raw", tau=TAU)
    with torch.no_grad(), (override_confidence(model.net, floor_fn(lam)) if lam is not None else _null()):
        o, _ = model.outputs(s.rgb, depth)
    H, W = depth.shape
    conf = full_res(o["confidence_input"], depth.shape, model.args.backbone_output_downsample_rate)
    return conf, o["pred"][0, 0, :H, :W].float().cpu().numpy().astype(np.float32), depth, pinfo


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cache_f1plus(model, ds, records, lam_of, out_root, base_receipt):
    """F1+ predictions for `records` (lam_of(record) gives its lambda), cached and evaluated like F0 / F1."""
    from tp.eval.run_eval import evaluate_manifest
    path, manifest = manifest_records(ds)
    output = out_root
    receipt = start_run(output, dict(base_receipt, dataset=ds, variant=VARIANT),
                        [sys.executable, "-m", "scripts.p1_trust_floor", *sys.argv[1:]], 20261029)
    try:
        (output / "predictions").mkdir()
        rows, selected, fd = [], [], []
        for i, r in enumerate(records):
            spath = (path.parent / r["path"]).resolve()
            s = load(spath, r)
            lam = lam_of(r)
            conf, pred, depth, pinfo = conf_and_pred(model, s, lam)
            curve, n_correct = false_distrust_curve(conf, s)
            fd.append(dict(scene=s.scene, lam=lam, false_distrust=float(np.interp(lam, GRID, curve)), correct_px=n_correct))
            name = output / "predictions" / f"{i:06d}.npz"
            store(name, pred, "float32")
            rows.append(dict(**{k: getattr(s, k) for k in IDENTITY}, sample_npz_sha256=r["sha256"],
                             rgb_sha256=hashlib.sha256(s.rgb.tobytes()).hexdigest(),
                             prompt_in_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                             prediction_path=str(name), prediction_sha256=sha256(name), trust_floor_lambda=lam,
                             prompt=pinfo, storage="float32", clipped_pixels=0,
                             invalid_pixels=int((~(np.isfinite(pred) & (pred > 0))).sum())))
            selected.append(dict(r, path=str(spath), sensor=s.sensor))
            if i % 100 == 0:
                print(f"F1+ {ds} {i + 1}/{len(records)} lambda={lam}", flush=True)
        write_json(output / "images.json", rows)
        sel = output / "selected_manifest.json"
        write_json(sel, dict(schema=manifest["schema"], synthetic=manifest["synthetic"], records=selected,
                             source_manifest_sha256=sha256(path)))
        configs = evaluation_configs(output, sel, rows, VARIANT, 20261029)
        receipt.update(stage="real_dev_verified", validation_stage="development_cache_verified",
                       model_version=dict(name=f"OMNI-DC-v1.1-{VARIANT.upper()}", base=base_receipt["f1"],
                                          rule="c' = c if c < lambda else 1", alpha=ALPHA),
                       completed_at_utc=datetime.now(timezone.utc).isoformat(), images_sha256=sha256(output / "images.json"),
                       selected_manifest_sha256=sha256(sel), frame_count=len(rows), experiment_role=ROLE,
                       inputs=[{k: x[k] for k in (*IDENTITY, "rgb_sha256", "sample_npz_sha256")} for x in rows])
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(output / "receipt.json", receipt)
    for split, cfg in configs.items():
        evaluate_manifest(cfg, output / f"evaluation_{split}",
                          [sys.executable, "-m", "tp.eval.run_eval", "--config", str(output / f"eval_{split}.yaml"),
                           "--output", str(output / f"evaluation_{split}")])
    return fd


def main():
    import torch
    from baselines.omnidc_finetuned import FineTunedOmniDC
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None, help="engineering check only")
    a = ap.parse_args()
    job = json.loads(JOB.read_text(encoding="utf-8"))
    f1w = dict(run=job["run_dirs"]["f1"], sha256=job["checkpoints"]["f1"]["final"]["sha256"])
    out = ROOT / "results" / f"p1_trust_floor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, alpha=ALPHA, grid=[float(GRID[0]), float(GRID[-1]), 0.01], sweep=SWEEP, sets=SETS,
                                  f1=f1w, limit=a.limit), [sys.executable, "-m", "scripts.p1_trust_floor", *sys.argv[1:]], 20261029)
    try:
        f1 = FineTunedOmniDC(f1w["run"], "final.pt", f1w["sha256"], "f1", device="cuda").base
        lim = (lambda rs: rs[:a.limit]) if a.limit else (lambda rs: rs)
        # ---- calibration curves (no override)
        nyu_path, nyu_m = manifest_records("nyu")
        nyu_cal = lim([r for r in select_records(nyu_m["records"], {"train"}, "all") if not scene_inner_holdout("mirror3d_nyu", r["scene"])])
        nyu_eval = lim(select_records(nyu_m["records"], {"train", "val"}, SETS["nyu"]["eval_scope"]))
        if {r["scene"] for r in nyu_cal} & {r["scene"] for r in nyu_eval}:
            raise ValueError("NYU calibration and evaluation scenes overlap")
        ham_path, ham_m = manifest_records("hammer")
        ham_all = lim(select_records(ham_m["records"], {"train", "val"}, "all"))
        curves = {"nyu": [], "hammer": {}}
        for r in nyu_cal:
            s = load((nyu_path.parent / r["path"]).resolve(), r)
            conf, _, _, _ = conf_and_pred(f1, s)
            curves["nyu"].append(false_distrust_curve(conf, s)[0])
        print(f"NYU calibration curves: {len(curves['nyu'])}", flush=True)
        for i, r in enumerate(ham_all):
            s = load((ham_path.parent / r["path"]).resolve(), r)
            conf, _, _, _ = conf_and_pred(f1, s)
            curves["hammer"].setdefault(s.scene, []).append(false_distrust_curve(conf, s)[0])
            if i % 100 == 0:
                print(f"HAMMER calibration curves {i + 1}/{len(ham_all)}", flush=True)
        lam_nyu, bound_nyu = crc_max_lambda(np.array(curves["nyu"]), ALPHA)
        lam_ham = {}
        for scene in curves["hammer"]:
            others = np.array([c for sc, cs in curves["hammer"].items() if sc != scene for c in cs])
            lam_ham[scene] = crc_max_lambda(others, ALPHA)
        sweep = {al: crc_max_lambda(np.array(curves["nyu"]), al)[0] for al in SWEEP}
        receipt.update(lambda_nyu=lam_nyu, bound_nyu=bound_nyu, lambda_hammer={k: v[0] for k, v in lam_ham.items()},
                       sweep_lambda_nyu=sweep, calibration_images=dict(nyu=len(nyu_cal), hammer=len(ham_all)))
        write_json(out / "receipt.json", receipt)
        print(f"lambda NYU {lam_nyu} (bound {bound_nyu:.4f}); HAMMER {receipt['lambda_hammer']}", flush=True)
        # ---- F1+ caches on the gate populations, evaluated like F0 / F1
        base = dict(tau=TAU, alpha=ALPHA, f1=f1w, calibration_run=out.relative_to(ROOT).as_posix())
        suffix = "_limit" if a.limit else ""
        fd_nyu = cache_f1plus(f1, "nyu", nyu_eval, lambda r: lam_nyu, ROOT / "cache" / f"p1_f1plus_nyu_dev{suffix}", base)
        fd_ham = cache_f1plus(f1, "hammer", ham_all, lambda r: lam_ham[r["scene"]][0], ROOT / "cache" / f"p1_f1plus_hammer_dev{suffix}", base)
        # ---- descriptive: alpha sweep on NYU (pixel-pooled AbsRel)
        sweep_rows = {}
        for al, lam in [(ALPHA, lam_nyu), *sweep.items()]:
            tot = dict(clean_in=[0.0, 0], clean_out=[0.0, 0], C=[0.0, 0])
            for r in nyu_eval:
                s = load((nyu_path.parent / r["path"]).resolve(), r)
                _, pred, _, _ = conf_and_pred(f1, s, lam)
                reg = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=TAU, band=5)
                scope = focus_scope(s)[1]; ok = np.isfinite(pred) & (pred > 0); gt = s.gt.astype(np.float64)
                for k, m in (("clean_in", reg["clean"] & scope), ("clean_out", reg["clean"] & ~scope), ("C", reg["C"] & scope)):
                    m = m & ok; e = np.abs(pred[m] - gt[m]) / gt[m]; tot[k][0] += float(e.sum()); tot[k][1] += int(e.size)
            sweep_rows[al] = dict(lam=lam, **{k: (v[0] / v[1] if v[1] else None) for k, v in tot.items()})
        fdm = lambda rows: float(np.mean([x["false_distrust"] for x in rows])) if rows else None
        summary = dict(lambda_nyu=lam_nyu, lambda_hammer=receipt["lambda_hammer"], false_distrust_nyu_eval=fdm(fd_nyu),
                       false_distrust_hammer_eval=fdm(fd_ham), sweep_nyu=sweep_rows)
        write_json(out / "trust_floor.json", dict(summary=summary, false_distrust=dict(nyu=fd_nyu, hammer=fd_ham)))
        fmt = lambda v, p=4: "NA" if v is None else f"{v:.{p}f}"
        lines = ["# F1+ : F1 with a conformally calibrated trust floor (inference only)", "",
                 f"Generated by `python -m scripts.p1_trust_floor` ({out.name}). c' = c if c < lambda else 1 on F1's input-confidence "
                 f"cells; lambda by conformal risk control of the false-distrust rate at alpha = {ALPHA} (NYU: train scenes outside "
                 "the holdout; HAMMER: leave one scene out). Development data, test never read." + (" ENGINEERING CHECK (--limit)." if a.limit else ""), "",
                 f"- lambda NYU: {lam_nyu} (CRC bound {bound_nyu:.4f}); lambda HAMMER per held-out scene: {json.dumps(receipt['lambda_hammer'])}",
                 f"- false-distrust rate reached on the evaluated images (target E <= {ALPHA}): NYU {fmt(fdm(fd_nyu), 3)}, HAMMER {fmt(fdm(fd_ham), 3)}",
                 "- the decision is the unchanged P1 gate with F1+ in place of F1: `python -m scripts.p1_gate --spec configs/omnidc/p1plus_gate_v1.yaml`",
                 "", "Descriptive alpha sweep on NYU (pixel-pooled AbsRel; F1 0.0281 / 0.0376 / 0.1336, F0 0.0141 / 0.0025 / 1.039):", "",
                 "| alpha | lambda | clean in scope | clean outside scope | C |", "|---:|---:|---:|---:|---:|"]
        lines += [f"| {al} | {x['lam']} | {fmt(x['clean_in'])} | {fmt(x['clean_out'])} | {fmt(x['C'])} |" for al, x in sweep_rows.items()]
        (ROOT / "docs" / ("omnidc_p1_trust_floor_limit.md" if a.limit else "omnidc_p1_trust_floor.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", summary=summary, trust_floor_sha256=sha256(out / "trust_floor.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
