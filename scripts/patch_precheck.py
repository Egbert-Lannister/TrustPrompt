"""Direction 2 pre-check (docs/plan_patching.md): where inside PromptDA-L does a wrong reading get copied into the
output? Activation patching on the prompt path; GPU, inference only; development data only, test never read.

  python -m scripts.patch_precheck                 # NYU 95 + HAMMER 30 (the images of the CAPA pre-check)
  python -m scripts.patch_precheck --limit 2       # engineering check

PromptDA-L (DepthAnything/PromptDA @ 9161547) uses the prompt at five places ("sites"):
  N        min / max of the 192x256 prompt: it normalises the prompt and de-normalises the output (global scale);
  F4..F1   the zero-initialised prompt branch `resConfUnit_depth` of the DPT fusion blocks refinenet4 (coarsest) to
           refinenet1 (finest), whose output is added to the decoder features.
Two prompts per image: raw (the sensor prompt as PromptDA receives it) and oracle (wrong readings C replaced by the
globally aligned UniDepth prior D_g, the project's oracle_prior variant). A run takes the oracle version at the sites
in a set S and the raw version elsewhere; S = {} is PromptDA on the raw prompt, S = all sites is PromptDA on the
oracle prompt (both checked against model.predict). Runs: {}, all, the 5 single sites, the 10 pairs, the 5
all-but-one sets (22 forwards per image).

Metric: AbsRel on main-scope wrong-reading pixels C (NYU mirrors, HAMMER objects; tau 0.05), pixel-pooled per
dataset. Total effect TE = A(raw) - A(oracle). Restoration of S: r(S) = (A(raw) - A(S)) / TE.

PRE-REGISTERED RULE (fixed before any run): inheritance is **localised** if one set S of at most two sites has
r(S) >= 0.70 with scene-bootstrap CI low >= 0.50 (2000 draws, seed 20261102) on both datasets (the same S on both);
otherwise **diffuse**. A localised result is a go for designing an intervention at those sites (its own rule, written
before it is run); a diffuse result ends direction 2. Requires TE > 0 with CI low > 0 on both datasets, otherwise the
pre-check is void (nothing to localise). Everything else is descriptive (necessity of each site from the all-but-one
runs, IR, clean-pixel AbsRel).
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from scripts import tta_precheck as T

TAU, BAND, SEED, DRAWS = 0.05, 5, 20261102, 2000
SITES = ("N", "F4", "F3", "F2", "F1")
CONFIGS = {"nyu": "configs/promptda/nyu_vitl_oracle_prior.yaml", "hammer": "configs/promptda/hammer_vitl_oracle_prior.yaml"}


def run_sets():
    """All sets evaluated: {}, all, singles, pairs, all-but-one (as frozensets, in a fixed order)."""
    sets = [frozenset(), frozenset(SITES)]
    sets += [frozenset([s]) for s in SITES]
    sets += [frozenset(p) for p in combinations(SITES, 2)]
    sets += [frozenset(SITES) - {s} for s in SITES]
    return sets


def key(S):
    return "+".join(s for s in SITES if s in S) or "none"


# ---------------------------------------------------------------- patched forward (torch)

def patched_forward(model, x, p_raw, p_oracle, S):
    """PromptDA.forward(x, prompt) with the oracle prompt at the sites in S and the raw prompt elsewhere.
    N in S: min / max taken from the oracle prompt, else from the raw prompt. Fk in S: the prompt branch of
    refinenet k receives the oracle prompt (normalised with the run's min / max, resized exactly as the block does)."""
    import torch
    import torch.nn.functional as F
    stats_src = p_oracle if "N" in S else p_raw
    _, mn, mx = model.normalize(stats_src)
    raw_n = (p_raw - mn) / (mx - mn)
    ora_n = (p_oracle - mn) / (mx - mn)
    handles = []
    had_own = "normalize" in model.__dict__
    original = model.__dict__.get("normalize")
    try:
        model.normalize = lambda pd: (raw_n, mn, mx)
        for k in (1, 2, 3, 4):
            if f"F{k}" in S:
                branch = getattr(model.depth_head.scratch, f"refinenet{k}").resConfUnit_depth

                def pre(module, args, ora_n=ora_n):
                    return (F.interpolate(ora_n, args[0].shape[2:], mode="bilinear", align_corners=False),)
                handles.append(branch.register_forward_pre_hook(pre))
        with torch.no_grad():
            return model.forward(x, p_raw)
    finally:
        if had_own:
            model.normalize = original
        else:
            del model.normalize
        for h in handles:
            h.remove()


# ---------------------------------------------------------------- statistics (numpy, unit-tested)

def image_metrics(D, P, pv, G, gv, scope):
    R = regions(P, pv, G, gv, tau=TAU, band=BAND)
    C, clean = R["C"] & scope, R["clean"] & scope
    valid = np.isfinite(D) & (D > 0)
    Gs = np.where(G > 0, G, 1.0)
    Ps = np.where(P > 0, P, 1.0)
    rel = np.abs(D - G) / Gs
    inh = C & valid & (np.abs(D - P) / Ps < 0.05) & (rel > 0.05)
    return dict(C_sum=float(rel[C & valid].sum()), C_n=int((C & valid).sum()), clean_sum=float(rel[clean & valid].sum()),
                clean_n=int((clean & valid).sum()), inherited=int(inh.sum()), C=int(C.sum()), invalid=int((gv & ~valid).sum()))


def pooled(xs, k, what="C"):
    s = sum(x["runs"][k][f"{what}_sum"] for x in xs)
    n = sum(x["runs"][k][f"{what}_n"] for x in xs)
    return None if n == 0 else s / n


def total_effect(scenes):
    xs = [x for s in scenes for x in s]
    a, b = pooled(xs, "none"), pooled(xs, key(SITES))
    return None if a is None or b is None else a - b


def restoration(scenes, k):
    xs = [x for s in scenes for x in s]
    a, o, r = pooled(xs, "none"), pooled(xs, key(SITES)), pooled(xs, k)
    if None in (a, o, r) or a - o <= 0:
        return None
    return (a - r) / (a - o)


def necessity(scenes, site):
    xs = [x for s in scenes for x in s]
    a, o = pooled(xs, "none"), pooled(xs, key(SITES))
    r = pooled(xs, key(frozenset(SITES) - {site}))
    if None in (a, o, r) or a - o <= 0:
        return None
    return (r - o) / (a - o)


def summarise(rows, rng):
    res = {}
    for ds in T.DATASETS:
        xs = [r for r in rows if r["dataset"] == ds]
        if not xs:
            continue
        sc = T._by_scene(xs)
        keys = [key(S) for S in run_sets()]
        res[ds] = dict(images=len(xs), scenes=len(sc),
                       TE=T.scene_bootstrap(sc, total_effect, rng),
                       absrel_C={k: pooled(xs, k) for k in keys}, absrel_clean={k: pooled(xs, k, "clean") for k in keys},
                       IR={k: (lambda c: None if c == 0 else sum(x["runs"][k]["inherited"] for x in xs) / c)(
                           sum(x["runs"][k]["C"] for x in xs)) for k in keys},
                       restoration={k: T.scene_bootstrap(sc, lambda s, k=k: restoration(s, k), rng)
                                    for k in keys if k not in ("none",)},
                       necessity={s: necessity(sc, s) for s in SITES},
                       check_max_abs_diff=max(x["check_max_abs_diff"] for x in xs),
                       seconds_median=float(np.median([x["seconds"] for x in xs])))
    return res


def verdict(res):
    te_ok = {ds: x["TE"][1] is not None and x["TE"][1] > 0 for ds, x in res.items()}
    small = [key(S) for S in run_sets() if 1 <= len(S) <= 2]
    hits = {ds: [k for k in small if x["restoration"][k][0] is not None and x["restoration"][k][0] >= 0.70
                 and x["restoration"][k][1] >= 0.50] for ds, x in res.items()}
    full = len(res) == len(T.DATASETS)
    if not (full and all(te_ok.values())):
        return dict(outcome="void (no total effect to localise)", te_ok=te_ok, hits=hits, common=[])
    common = sorted(set.intersection(*[set(v) for v in hits.values()]), key=lambda k: (k.count("+"), k))
    return dict(outcome="localised" if common else "diffuse", te_ok=te_ok, hits=hits, common=common)


# ---------------------------------------------------------------- driver

def report(out, res, ver):
    f = lambda x, d=3: "NA" if x is None else f"{x:.{d}f}"
    ci = lambda t: f"{f(t[0])} ({f(t[1])} to {f(t[2])})"
    L = ["# Direction 2 pre-check: where PromptDA-L copies wrong readings (pre-registered, development data)", "",
         f"Generated by `python -m scripts.patch_precheck` ({out.name}); rule fixed in docs/plan_patching.md and the "
         "script docstring before the run. Test never read.", "",
         f"**Outcome: {ver['outcome']}**" + (f" (sets: {', '.join(ver['common'])})" if ver.get("common") else "") + ".", "",
         "| dataset | images / scenes | AbsRel C raw | AbsRel C oracle | total effect (95 % CI) |", "|---|---:|---:|---:|---:|"]
    for ds, x in res.items():
        L.append(f"| {ds} | {x['images']} / {x['scenes']} | {f(x['absrel_C']['none'], 4)} | {f(x['absrel_C'][key(SITES)], 4)} | "
                 f"{ci(x['TE'])} |")
    L += ["", "Restoration r(S) = share of the total effect removed by giving the oracle prompt only to the sites in S "
          "(95 % CI); necessity = share that returns when only that site keeps the raw prompt.", "",
          "| sites | " + " | ".join(f"{ds} r(S)" for ds in res) + " |", "|---|" + "---:|" * len(res)]
    for S in run_sets():
        if 1 <= len(S) <= 2 or len(S) == len(SITES) - 1:
            k = key(S)
            L.append(f"| {k} | " + " | ".join(ci(x["restoration"][k]) for x in res.values()) + " |")
    L += ["", "| site | " + " | ".join(f"{ds} necessity" for ds in res) + " |", "|---|" + "---:|" * len(res)]
    for s in SITES:
        L.append(f"| {s} | " + " | ".join(f(x["necessity"][s]) for x in res.values()) + " |")
    L += ["", "| dataset | IR raw | IR oracle | AbsRel clean raw | AbsRel clean oracle | s / image | check |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, x in res.items():
        L.append(f"| {ds} | {f(x['IR']['none'])} | {f(x['IR'][key(SITES)])} | {f(x['absrel_clean']['none'], 4)} | "
                 f"{f(x['absrel_clean'][key(SITES)], 4)} | {x['seconds_median']:.1f} | {x['check_max_abs_diff']:.1e} |")
    L += ["", "check = largest |patched run - model.predict| (metres, model resolution) for S = {} and S = all sites.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import cv2
    import torch
    from baselines.promptda import PromptDAPredictor, model_input_size, prepare_prompt
    from baselines.cached_prior import CachedPrior
    from tp.align.robust import align_depth
    cfgs = {ds: yaml.safe_load((ROOT / p).read_text(encoding="utf-8")) for ds, p in CONFIGS.items()}
    if len({(c["source"], c["ckpt"]) for c in cfgs.values()}) != 1:
        raise SystemExit("NYU and HAMMER configs name different PromptDA checkpoints")
    c0 = cfgs["nyu"]
    config = dict(tau=TAU, band=BAND, seed=SEED, draws=DRAWS, sites=list(SITES), runs=[key(S) for S in run_sets()],
                  promptda_configs=CONFIGS, limit=a.limit, images="CAPA pre-check selection (scripts.tta_precheck.select)")
    out = ROOT / "results" / f"patch_precheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, config, [sys.executable, "-m", "scripts.patch_precheck", *sys.argv[1:]], SEED)
    try:
        receipt["model_version"] = dict(name="PromptDA-L", checkpoint=c0["ckpt"], checkpoint_sha256=sha256(c0["ckpt"]),
                                        source=c0["source"])
        torch.manual_seed(int(c0["seed"]))
        pred = PromptDAPredictor(Path(c0["source"]), Path(c0["ckpt"]), "vitl", device=a.device)
        model = pred.model
        priors = {ds: CachedPrior(ROOT / c["prior_cache"]) for ds, c in cfgs.items()}
        rows = []
        todo = [(ds, r) for ds in T.DATASETS for r in T.select(ds, T.DATASETS[ds]["per_group"], a.limit)]
        for i, (ds, r) in enumerate(todo):
            path = ROOT / T.DATASETS[ds]["processed"] / r["path"]
            h = sha256(path)
            if h != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            s = load_sample(path)
            if s.split == "test":
                raise ValueError("test sample reached the pre-check")
            cfg = cfgs[ds]
            fill = align_depth(priors[ds].predict(s, h), s.prompt, s.prompt_valid, seed=int(cfg["seed"]),
                               compute_local=False, **cfg.get("robust", {}))["global_depth"]
            remove = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=float(cfg["tau"]), band=5)["C"]
            p_raw, _ = prepare_prompt(s.prompt, s.prompt_valid, variant="raw")
            p_ora, _ = prepare_prompt(s.prompt, s.prompt_valid, variant="oracle_prior", remove=remove, fill_depth=fill)
            hh, ww = s.rgb.shape[:2]
            th, tw, mode = model_input_size(hh, ww)
            image = s.rgb.astype(np.float32) / 255.0
            if (th, tw) != (hh, ww):
                image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA if mode == "official_area_downscale" else cv2.INTER_LINEAR)
            x = torch.from_numpy(image).permute(2, 0, 1)[None].to(a.device)
            pr = torch.from_numpy(np.ascontiguousarray(p_raw))[None, None].to(a.device)
            po = torch.from_numpy(np.ascontiguousarray(p_ora))[None, None].to(a.device)
            _, scope = focus_scope(s)
            t1 = time.time()
            runs, check = {}, 0.0
            for S in run_sets():
                d = patched_forward(model, x, pr, po, S)[0, 0]
                if len(S) in (0, len(SITES)):
                    ref = model.predict(x, po if S else pr)[0, 0]
                    check = max(check, float((d - ref).abs().max()))
                d = d.float().cpu().numpy()
                D = cv2.resize(d, (ww, hh), interpolation=cv2.INTER_LINEAR) if (th, tw) != (hh, ww) else d
                runs[key(S)] = image_metrics(D, s.prompt, s.prompt_valid, s.gt, s.gt_valid, scope)
            seconds = time.time() - t1
            if check > 1e-4:
                raise RuntimeError(f"patched forward differs from model.predict by {check:.2e} m on {r['path']}")
            rows.append(dict(dataset=ds, scene=s.scene, path=r["path"], seconds=seconds, check_max_abs_diff=check, runs=runs))
            print(f"[{i + 1}/{len(todo)}] {ds} {r['path']}: {seconds:.1f} s, check {check:.1e}", flush=True)
        write_json(out / "per_image.json", rows)
        res = summarise(rows, np.random.default_rng(SEED))
        ver = verdict(res) if a.limit is None else dict(outcome="engineering check (no verdict)", common=[])
        write_json(out / "patch_precheck.json", dict(results=res, verdict=ver))
        lines = report(out, res, ver)
        doc = ROOT / "docs" / ("patch_precheck.md" if a.limit is None else "patch_precheck_engineering.md")
        doc.write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="complete" if a.limit is None else "engineering_check_complete", verdict=ver,
                       result_sha256=sha256(out / "patch_precheck.json"), doc=doc.relative_to(ROOT).as_posix())
        print("\n".join(lines))
        print("PATCH PRECHECK DONE" if a.limit is None else "PATCH ENGINEERING CHECK DONE", flush=True)
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
