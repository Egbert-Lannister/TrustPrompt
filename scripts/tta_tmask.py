"""Supplementary experiment S2 (docs/plan_supplement.md): does method v1's trust mask stop CAPA from copying wrong
readings? CAPA (same code, weights, config and per-image seed as the CAPA pre-check) adapts only to the readings that
T v1 trusts. GPU; development data only, test never read.

  python -m scripts.tta_tmask --limit 1                       # engineering check
  python -m scripts.tta_tmask --datasets nyu                  # NYU first (about 1.5 h)
  python -m scripts.tta_tmask --resume latest [--max-minutes 120]   # continue the newest run (all datasets)

Images (T must be out of fit):
  NYU     val + train-holdout scenes (scripts.cache_completion.select_records, scope "holdout"): 30 images.
  HAMMER  the 30 images of the CAPA pre-check; T maps and thresholds from the leave-scene-out cross-fitted cache
          (configs/trust/hammer_crossfit_v1.yaml), so every scene is out of fold.
Variants:
  tmask        condition = readings with T v1 trust >= the conformal threshold (alpha 0.1, method v1); deployable.
  oracle_mask  condition = readings outside C [oracle, reads GT]; NYU only (HAMMER takes ~10 min per image per variant).
Reference: CAPA on all readings (step 100) and the aligned frozen prior (step 0), both read from
results/tta_precheck_20261101_012932 (same images, settings and seed).

PRE-REGISTERED RULE (fixed before any run): reduction rho = 1 - AbsRel_C(tmask) / AbsRel_C(CAPA raw), main scope,
pixel-pooled per dataset, paired scene bootstrap (2000 draws, seed 20261104).
  NYU: rho >= 0.50 with CI low > 0;  HAMMER: CI low of rho > 0.
  outcome "rescues" if both hold, "nyu_only" if only NYU holds, otherwise "does_not_rescue".
Descriptive: oracle-mask rho on NYU; clean-pixel AbsRel of tmask vs CAPA raw (the masking cost); C AbsRel vs the
prior; IR; invalid pixels.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from scripts import tta_precheck as T
from scripts.patch_precheck import image_metrics

SEED_BOOT, DRAWS = 20261104, 2000
CAPA_RUN = "results/tta_precheck_20261101_012932"
VARIANTS = {"nyu": ("tmask", "oracle_mask"), "hammer": ("tmask",)}
TRUST = {"nyu": "cache/trust_t_v1_real_sim_nofeat_nyu_dev_v1", "hammer": "cache/trust_t_v1cf_real_sim_nofeat_hammer_dev_v1"}


def records(ds, limit):
    rows = json.loads((ROOT / T.DATASETS[ds]["processed"] / "manifest.json").read_text(encoding="utf-8"))["records"]
    if ds == "nyu":
        from scripts.cache_completion import select_records
        rows = select_records(rows, {"train", "val"}, "holdout")
    else:
        rows = per_group_subsample([r for r in rows if r["split"] in {"train", "val"}], T.DATASETS[ds]["per_group"])
    return rows[:limit] if limit else rows


def flag(s, variant, trust, thr):
    if variant == "oracle_mask":
        return regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=T.TAU, band=5)["C"] & s.prompt_valid
    if variant == "tmask":
        return s.prompt_valid & (trust < thr)
    raise ValueError(variant)


def pooled(xs, k, what="C"):
    s = sum(x["metrics"][k][f"{what}_sum"] for x in xs)
    n = sum(x["metrics"][k][f"{what}_n"] for x in xs)
    return None if n == 0 else s / n


def reduction(scenes, k, ref="capa_raw"):
    xs = [x for s in scenes for x in s]
    a, b = pooled(xs, k), pooled(xs, ref)
    return None if a is None or not b else 1 - a / b


def summarise(rows, rng):
    res = {}
    for ds in VARIANTS:
        xs = [r for r in rows if r["dataset"] == ds]
        if not xs:
            continue
        sc = T._by_scene(xs)
        keys = ["prior", "capa_raw", *[v for v in VARIANTS[ds] if all(v in x["metrics"] for x in xs)]]
        res[ds] = dict(images=len(xs), scenes=len(sc),
                       absrel_C={k: pooled(xs, k) for k in keys}, absrel_clean={k: pooled(xs, k, "clean") for k in keys},
                       IR={k: (lambda c: None if c == 0 else sum(x["metrics"][k]["inherited"] for x in xs) / c)(
                           sum(x["metrics"][k]["C"] for x in xs)) for k in keys},
                       invalid={k: sum(x["metrics"][k]["invalid"] for x in xs) for k in keys},
                       rho={k: T.scene_bootstrap(sc, lambda s, k=k: reduction(s, k), rng) for k in keys if k not in ("prior", "capa_raw")},
                       flagged_share=float(np.mean([x["flagged"]["tmask"] / max(x["valid"], 1) for x in xs])),
                       seconds_median=float(np.median([x["seconds"]["tmask"] for x in xs])))
    return res


def verdict(res):
    def ok(ds, need):
        if ds not in res or "tmask" not in res[ds]["rho"]:
            return False
        p, lo, _ = res[ds]["rho"]["tmask"]
        return p is not None and lo > 0 and (p >= 0.50 if need else True)
    n, h = ok("nyu", True), ok("hammer", False)
    return dict(nyu=n, hammer=h, outcome="rescues" if n and h else ("nyu_only" if n else "does_not_rescue"))


def report(out, res, ver):
    f = lambda x, d=4: "NA" if x is None else f"{x:.{d}f}"
    ci = lambda t: f"{f(t[0], 3)} ({f(t[1], 3)} to {f(t[2], 3)})"
    L = ["# S2: method v1's trust mask in front of CAPA (pre-registered, development data)", "",
         f"Generated by `python -m scripts.tta_tmask` ({out.name}); rule fixed in docs/plan_supplement.md and the script "
         f"docstring before the run. CAPA raw and the prior from {CAPA_RUN}. Test never read.", "",
         f"**Outcome: {ver['outcome']}** (NYU {ver.get('nyu')}, HAMMER {ver.get('hammer')}).", "",
         "| dataset | images / scenes | variant | AbsRel C | AbsRel clean | IR | reduction vs CAPA raw (95 % CI) | invalid px |",
         "|---|---:|---|---:|---:|---:|---:|---:|"]
    for ds, x in res.items():
        for k in x["absrel_C"]:
            r = ci(x["rho"][k]) if k in x["rho"] else "-"
            L.append(f"| {ds} | {x['images']} / {x['scenes']} | {k} | {f(x['absrel_C'][k])} | {f(x['absrel_clean'][k])} | "
                     f"{f(x['IR'][k], 3)} | {r} | {x['invalid'][k]} |")
    L += ["", "Share of readings the T mask removes (mean per image): " +
          ", ".join(f"{ds} {x['flagged_share']:.3f}" for ds, x in res.items()) + ". Median seconds per image (tmask): " +
          ", ".join(f"{ds} {x['seconds_median']:.0f}" for ds, x in res.items()) + ".", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-endpoint", default=None)
    a = ap.parse_args()
    import importlib.util
    if importlib.util.find_spec("peft") is None:
        raise SystemExit("missing package peft; install with: python -m pip install peft")
    import torch
    from tp.trust.cache import TrustCache
    from scripts.cache_completion import crc_threshold
    run = ROOT / CAPA_RUN
    rec = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    if rec.get("stage") != "complete" or sha256(run / "tta_precheck.json") != rec["result_sha256"]:
        raise SystemExit(f"{CAPA_RUN} is not a complete, untouched CAPA run")
    cache_cfg = yaml.safe_load((ROOT / "configs/baselines/cache.yaml").read_text(encoding="utf-8"))
    config = dict(capa_run=CAPA_RUN, capa_result_sha256=rec["result_sha256"], variants=VARIANTS, trust=TRUST,
                  crc=cache_cfg["crc"], seed=T.SEED, boot_seed=SEED_BOOT, draws=DRAWS, limit=a.limit,
                  align_points=T.ALIGN_POINTS, capa=T.CAPA)
    command = [sys.executable, "-m", "scripts.tta_tmask", *sys.argv[1:]]
    if a.resume is None:
        out = ROOT / "results" / f"tta_tmask_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, config, command, SEED_BOOT)
    else:
        if str(a.resume) == "latest":
            runs = sorted((ROOT / "results").glob("tta_tmask_*"))
            if not runs:
                raise SystemExit("no results/tta_tmask_* run to resume")
            out = runs[-1]
            print(f"resuming {out.relative_to(ROOT).as_posix()}", flush=True)
        else:
            out = a.resume if a.resume.is_absolute() else ROOT / a.resume
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        if receipt["config"] != json.loads(json.dumps(config)):
            raise ValueError("resume requires the identical configuration")
        if sha256(ROOT / "scripts" / "tta_tmask.py") != receipt["source_sha256"]["scripts/tta_tmask.py"]:
            raise ValueError("scripts/tta_tmask.py changed since the run started; start a new run")
        receipt.setdefault("resumed", []).append(command)
    t0 = time.time()
    try:
        info = T.fetch_capa(a.hf_endpoint)
        if info["weights_revision"] != rec["capa_setup"]["capa"]["weights_revision"]:
            raise RuntimeError("UniDepthV2-L weights differ from the CAPA pre-check")
        device = torch.device(a.device)
        proto, cfg = T.load_protocol(device, info["weights_dir"])
        tcache = {ds: TrustCache(ROOT / TRUST[ds]) for ds in a.datasets}
        thr = {ds: (None if tcache[ds].per_item else crc_threshold(cache_cfg, {"nyu": "mirror3d_nyu", "hammer": "hammer"}[ds])[0])
               for ds in a.datasets}
        receipt.update(trust_cache_receipts={ds: sha256(tcache[ds].root / "receipt.json") for ds in a.datasets},
                       thresholds={ds: thr[ds] if thr[ds] is not None else "per_item" for ds in a.datasets},
                       model_version=f"CAPA {T.CAPA['commit']} + UniDepthV2-L {info['weights_revision']}")
        per = out / "per_image"
        per.mkdir(exist_ok=True)
        (out / "predictions").mkdir(exist_ok=True)
        todo = [(ds, r) for ds in a.datasets for r in records(ds, a.limit)]
        for i, (ds, r) in enumerate(todo):
            name = f"{ds}__{Path(r['path']).stem}"
            if (per / f"{name}.json").is_file():
                continue
            if a.max_minutes is not None and time.time() - t0 > 60 * a.max_minutes:
                receipt.update(stage="partial_time_budget")
                print(f"time budget reached; resume with: python -m scripts.tta_tmask --resume "
                      f"{out.relative_to(ROOT).as_posix()} --datasets {' '.join(a.datasets)}"
                      + ("" if a.limit is None else f" --limit {a.limit}"), flush=True)
                return
            path = ROOT / T.DATASETS[ds]["processed"] / r["path"]
            h = sha256(path)
            if h != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            s = load_sample(path)
            if s.split == "test":
                raise ValueError("test sample reached the experiment")
            trust = tcache[ds].trust(s, h, s.prompt.shape)
            th = tcache[ds].threshold_for(s) if thr[ds] is None else thr[ds]
            rgb, cond, mask = T.tensors(s)
            z = np.load(run / "predictions" / f"{name}.npz")
            preds = {"prior": z["step_0"].astype(np.float32), "capa_raw": z["step_100"].astype(np.float32)}
            flagged, seconds = {}, {}
            for v in VARIANTS[ds]:
                fl = flag(s, v, trust, th)
                keep = mask & torch.from_numpy(~fl)[None]
                flagged[v] = int(fl.sum())
                torch.manual_seed(T.SEED)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(T.SEED)
                t1 = time.time()
                pred, _, _ = T.run_with_retries(proto, rgb, cond, keep)
                seconds[v] = time.time() - t1
                preds[v] = pred[0].float().cpu().numpy()
            _, scope = focus_scope(s)
            metrics = {k: image_metrics(D, s.prompt, s.prompt_valid, s.gt, s.gt_valid, scope) for k, D in preds.items()}
            np.savez_compressed(out / "predictions" / f"{name}.npz", **{v: preds[v].astype(np.float16) for v in VARIANTS[ds]})
            write_json(per / f"{name}.json", dict(dataset=ds, scene=s.scene, sensor=s.sensor, threshold=float(th),
                                                 valid=int(s.prompt_valid.sum()), flagged=flagged, seconds=seconds, metrics=metrics))
            print(f"[{i + 1}/{len(todo)}] {name}: " + ", ".join(f"{v} {seconds[v]:.0f} s" for v in seconds), flush=True)
        rows = [json.loads((per / f"{ds}__{Path(r['path']).stem}.json").read_text(encoding="utf-8")) for ds, r in todo]
        res = summarise(rows, np.random.default_rng(SEED_BOOT))
        full = a.limit is None and set(a.datasets) == set(VARIANTS)
        ver = verdict(res) if full else dict(outcome="partial or engineering check (no verdict)")
        write_json(out / "tta_tmask.json", dict(results=res, verdict=ver))
        lines = report(out, res, ver)
        doc = ROOT / "docs" / ("tta_tmask.md" if full else "tta_tmask_partial.md")
        doc.write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="complete" if full else "partial_complete", verdict=ver,
                       result_sha256=sha256(out / "tta_tmask.json"), doc=doc.relative_to(ROOT).as_posix())
        print("\n".join(lines))
        print("TTA TMASK DONE" if full else "TTA TMASK PARTIAL DONE; continue with: python -m scripts.tta_tmask --resume latest",
              flush=True)
    except BaseException as e:
        if receipt.get("stage") in {"started", None}:
            receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
