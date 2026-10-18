"""Ceiling of output-level selection between two runs of the same model (diagnostic, CPU).

  python -m scripts.selection_ceiling --set nyu [--workers 6]
  python -m scripts.selection_ceiling --set hammer        (val + train-holdout scenes; T v1 mask)
  python -m scripts.selection_ceiling --set hammer_cf     (all 10 HAMMER dev scenes; cross-fitted T mask)

For each model, A = prediction from the raw prompt, B = prediction from the T-masked prompt (both cached).
A selection gate picks, per region, A or B. The ceilings below let the ground truth pick (oracle) at several
granularities, so they bound what any gate can reach; they are not a method:
  pixel        per pixel, the closer of A and B (absolute ceiling; includes picking noise)
  block<k>     per k x k block, the one with the lower AbsRel over valid GT pixels in the block
  image        per image, the one with the lower AbsRel over the main scope
Reported as in every table: main-scope C AbsRel, main-scope clean AbsRel, main-scope all-pixels AbsRel;
image -> scene -> dataset means; images whose region is empty are skipped for that region.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
import time

import numpy as np

from baselines.cached_prior import CachedPrior
from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, write_json
from tp.trust.data import scene_inner_holdout

TAU = 0.05
BLOCKS = (8, 32, 128)
METHODS = ("A", "B", *(f"block{k}" for k in reversed(BLOCKS)), "image", "pixel")
LABELS = {"promptda": "PromptDA-L", "priorda": "PriorDA-B-v1.1", "omnidc": "OMNI-DC-v1.1",
          "cdm": "CDM-ViT-L-per-camera", "lingbot": "LingBot-Depth-v0.5", "any2full": "Any2Full-L"}
BL = ("priorda", "omnidc", "cdm", "lingbot", "any2full")
SETS = {
    "nyu": dict(processed="data/processed/mirror3d_dev_v1", scope="holdout",
                pairs={"promptda": ("cache/promptda_vitl_raw_nyu_dev_v1",
                                    "cache/promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_nyu_dev_v1"),
                       **{m: (f"cache/bl_{m}_raw_nyu_dev_v1", f"cache/bl_{m}_tmask_nyu_dev_v1") for m in BL}}),
    "hammer": dict(processed="data/processed/hammer_dev_v1", scope="holdout",
                   pairs={"promptda": ("cache/promptda_vitl_raw_hammer_dev_v1",
                                       "cache/promptda_vitl_trust_v1_crc10_prior_real_sim_nofeat_hammer_dev_v1"),
                          **{m: (f"cache/bl_{m}_raw_hammer_dev_v1", f"cache/bl_{m}_tmask_hammer_dev_v1") for m in BL}}),
    "hammer_cf": dict(processed="data/processed/hammer_dev_v1", scope="all",
                      pairs={"promptda": ("cache/promptda_vitl_raw_hammer_dev_v1", "cache/promptda_vitl_trust_v1cf_prior_hammer_dev_v1"),
                             **{m: (f"cache/bl_{m}_raw_all_hammer_dev_v1", f"cache/bl_{m}_tmask_cfall_hammer_dev_v1") for m in BL}}),
}


def absrel(pred, gt, m):
    if not m.any():
        return None
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


def block_select(ea, eb, valid, k):
    """Per k x k block: True where B has the lower summed relative error over valid GT pixels."""
    H, W = ea.shape
    hb, wb = -(-H // k), -(-W // k)
    pad = lambda x: np.pad(x, ((0, hb * k - H), (0, wb * k - W)))
    sa = pad(np.where(valid, ea, 0)).reshape(hb, k, wb, k).sum((1, 3))
    sb = pad(np.where(valid, eb, 0)).reshape(hb, k, wb, k).sum((1, 3))
    choose_b = sb < sa
    return np.repeat(np.repeat(choose_b, k, 0), k, 1)[:H, :W]


def one(job):
    path, sha, a_root, b_root, label = job
    s = load_sample(ROOT / path)
    A = CachedPrior(ROOT / a_root, expected_model=label).predict(s, sha).astype(np.float64)
    B = CachedPrior(ROOT / b_root, expected_model=label).predict(s, sha).astype(np.float64)
    gt = s.gt.astype(np.float64)
    valid = s.gt_valid & np.isfinite(A) & np.isfinite(B) & (A > 0) & (B > 0)
    if not (np.isfinite(A[s.gt_valid]).all() and np.isfinite(B[s.gt_valid]).all()):
        return dict(key=[s.sensor, s.scene, s.frame], split=s.split, invalid=True)
    g = np.where(valid, gt, 1.0)
    ea, eb = np.abs(A - g) / g, np.abs(B - g) / g
    reg = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=TAU, band=5)
    sc = focus_scope(s)[1] & s.gt_valid
    masks = dict(C=reg["C"] & sc, clean=reg["clean"] & sc, all=sc)
    preds = dict(A=A, B=B, pixel=np.where(eb < ea, B, A))
    for k in BLOCKS:
        preds[f"block{k}"] = np.where(block_select(ea, eb, valid, k), B, A)
    preds["image"] = B if (absrel(B, gt, sc) or 0) < (absrel(A, gt, sc) or 0) else A
    out = dict(key=[s.sensor, s.scene, s.frame], split=s.split, invalid=False,
               share_b={m: float(np.mean(preds[m][sc] == B[sc])) if sc.any() else None for m in preds})
    out["metrics"] = {m: {r: absrel(p, gt, mk) for r, mk in masks.items()} for m, p in preds.items()}
    return out


def aggregate(rows, method, region):
    by = defaultdict(list)
    for r in rows:
        v = r["metrics"][method][region]
        if v is not None:
            by[r["key"][1]].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None, len(by)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (resume by re-running)")
    a = ap.parse_args()
    spec = SETS[a.set]; partial = False
    recs = json.loads((ROOT / spec["processed"] / "manifest.json").read_text(encoding="utf-8"))["records"]
    recs = [r for r in recs if r["split"] == "val" or (r["split"] == "train" and
            (spec["scope"] == "all" or scene_inner_holdout(r["dataset"], r["scene"])))]
    results, lines = {}, [f"# Selection ceiling: `{a.set}`", "",
                          "Generated by `python -m scripts.selection_ceiling`. A = raw prompt, B = T-masked prompt (same model). "
                          "Oracle selections let the GT choose between A and B; they bound any selection gate and are not a method. "
                          "Main scope (NYU mirror / HAMMER object pixels); image -> scene -> dataset means.", ""]
    for m, (ra, rb) in spec["pairs"].items():
        if a.models and m not in a.models:
            continue
        if not (ROOT / ra / "receipt.json").is_file() or not (ROOT / rb / "receipt.json").is_file():
            lines.append(f"- {LABELS[m]}: skipped, cache missing ({ra if not (ROOT / ra).exists() else rb})"); continue
        jobs = [(str(Path_(spec["processed"], r["path"])), r["sha256"], ra, rb, LABELS[m]) for r in recs]
        part = ROOT / "results" / "selection_ceiling_parts" / f"{a.set}__{m}.jsonl"     # resumable (bridge time limits)
        part.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if part.is_file():
            for l in part.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l); done[r["job"]] = r
        todo = [j for j in jobs if j[0] not in done]
        t0 = time.time()
        with part.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(max(1, a.workers)) as ex:
            for i in range(0, len(todo), 4 * max(1, a.workers)):
                if a.budget and time.time() - t0 > a.budget:
                    break
                chunk = todo[i:i + 4 * max(1, a.workers)]
                for j, r in zip(chunk, ex.map(one, chunk)):
                    r["job"] = j[0]; done[j[0]] = r; fh.write(json.dumps(r) + "\n")
                fh.flush()
        if len(done) < len(jobs):
            print(f"PARTIAL {m}: {len(done)}/{len(jobs)} images; run again", flush=True); partial = True; continue
        rows = [done[j[0]] for j in jobs]
        bad = [r for r in rows if r["invalid"]]; rows = [r for r in rows if not r["invalid"]]
        pops = {"val": [r for r in rows if r["split"] == "val"],
                ("train holdout" if spec["scope"] == "holdout" else "train"): [r for r in rows if r["split"] == "train"]}
        if spec["scope"] == "all":
            pops = {"all dev scenes": rows}
        results[m] = {}
        f = lambda v: "NA" if v is None else f"{v:.4f}"
        for pop, pr in pops.items():
            if not pr:
                continue
            res = {meth: {reg: aggregate(pr, meth, reg)[0] for reg in ("C", "clean", "all")} for meth in METHODS}
            res["_images"] = len(pr); res["_scenes"] = aggregate(pr, "A", "all")[1]
            res["_share_b"] = {meth: float(np.mean([r["share_b"][meth] for r in pr if r["share_b"][meth] is not None])) for meth in METHODS}
            results[m][pop] = res
            lines += [f"## {LABELS[m]}, {pop} ({res['_images']} images / {res['_scenes']} scenes)", "",
                      "| selection | C AbsRel | clean AbsRel | all AbsRel | share of scope taken from B |", "|---|---:|---:|---:|---:|"]
            for meth in METHODS:
                lines.append(f"| {meth} | {f(res[meth]['C'])} | {f(res[meth]['clean'])} | {f(res[meth]['all'])} | {res['_share_b'][meth]:.2f} |")
            lines.append("")
            print(m, pop, {k: round(res[k]["all"], 4) for k in METHODS}, flush=True)
        results[m]["_invalid_images"] = len(bad)
        if bad:
            lines += [f"{len(bad)} images skipped for {LABELS[m]}: invalid prediction inside valid GT.", ""]
    if partial:
        return
    tag = a.set if not a.models else f"{a.set}__{'_'.join(a.models)}"
    write_json(ROOT / "results" / f"selection_ceiling_{tag}.json", results)
    (ROOT / "docs" / f"selection_ceiling_{a.set}.md" if not a.models else ROOT / "results" / f"selection_ceiling_{tag}.md").write_text("\n".join(lines), encoding="utf-8")


def Path_(*parts):
    from pathlib import PurePosixPath
    return PurePosixPath(*parts)


if __name__ == "__main__":
    main()
