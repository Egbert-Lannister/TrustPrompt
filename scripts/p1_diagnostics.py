"""Mainline P1 diagnostics after the gate failed on G3 (docs/plan_omnidc.md sec. 20). CPU; development data only,
test never read; descriptive (no decision rule).

  python -m scripts.p1_diagnostics [--part A B] [--hammer-per-group 3] [--hypersim-frames 60 30]

A  Readings inside the surface region, real vs synthetic. For every pixel of the region with valid GT: no reading
   (hole), wrong reading (C: |P - GT| / GT > tau) or correct reading (clean). Real: NYU train mirrors, HAMMER train
   objects (N frames per scene x sensor), the main scope of the evaluation. Synthetic: the first frames of the D3 sample
   (scripts/hypersim_synthesis_stats: same frame list and per-frame seeds), annotated mirror / window pixels after
   tp.sim.hypersim_condition. Question: does the synthesis leave correct readings on mirrors as real sensors do?
B  Where F1's clean-pixel error comes from. On the gate's populations (NYU val + train-holdout; HAMMER dev, N frames
   per scene x sensor), clean pixels of the main scope (tp.eval.regions: correct readings outside the 5-pixel band
   around C), binned by distance to the nearest C pixel of the image; per bin the pixel share and the mean AbsRel of
   R0, F0 and F1 (pixel-pooled), the share of F1's error that lies behind the GT, and the same for clean pixels
   outside the main scope. Also how concentrated the per-image increase is.
D  How many training cells the confidence labels call noisy. OMNI-DC's input confidence lives on a 1/4-resolution
   grid; a cell is labelled noisy if any reading in it is wrong: F0 rule = OMNI-DC's (P - GT > 0.1 in GT-median units),
   F1 rule = two-sided (|P - GT| / GT > tau). Measured on F0 / F1 training inputs (tp.omnidc.data.make_sample, dense,
   D3 frames, 3 draws each; also without OMNI-DC's i.i.d. outliers) and on the real dev images (outside the main
   scope too). Question: does the F1 rule teach the confidence head a base rate of distrust that real data lacks?
Predictions come from the caches the gate used (configs/omnidc/p1_gate_v1.yaml: R0 uint16, F0 / F1 float32).
Writes results/p1_diagnostics_<time>/diagnostics.json and docs/omnidc_p1_diagnostics.md.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from scripts.cache_completion import select_records
from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json

TAU, BAND = 0.05, 5
DIST_BINS = [(6, 10), (10, 20), (20, 40), (40, None)]
DATASETS = {"nyu": dict(processed="data/processed/mirror3d_dev_v1", scope="holdout"),
            "hammer": dict(processed="data/processed/hammer_dev_v1", scope="all")}
CACHES = {"nyu": dict(R0="cache/bl_omnidc_raw_nyu_dev_v1", F0="cache/p1_f0_final_nyu_dev", F1="cache/p1_f1_final_nyu_dev"),
          "hammer": dict(R0="cache/bl_omnidc_raw_all_hammer_dev_v1", F0="cache/p1_f0_final_hammer_dev",
                         F1="cache/p1_f1_final_hammer_dev")}
MODELS = ("R0", "F0", "F1")


def records(ds, splits, per_group, scope=None):
    """scope: train-split scenes admitted (the gate's: NYU holdout, HAMMER all); part A uses all train scenes, as D3."""
    d = DATASETS[ds]
    processed = ROOT / d["processed"]
    rows = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"], set(splits),
                          scope or d["scope"])
    return processed, per_group_subsample(rows, per_group)


def reading_shares(P, V, gt, gv, region):
    """Pixel counts inside `region` with valid GT: hole / wrong (C) / correct reading."""
    m = region & gv
    both = m & V
    rel = np.zeros(gt.shape); rel[both] = np.abs(P[both] - gt[both]) / gt[both]
    c = both & (rel > TAU)
    return dict(region=int(m.sum()), hole=int((m & ~V).sum()), C=int(c.sum()), clean=int((both & ~c).sum()))


def pool_shares(rows):
    tot = {k: sum(r[k] for r in rows) for k in ("region", "hole", "C", "clean")}
    n = max(1, tot["region"])
    with_clean = [r["clean"] / r["region"] for r in rows if r["region"]]
    return dict(images=len(rows), region_px=tot["region"], hole_share=tot["hole"] / n, C_share=tot["C"] / n,
                clean_share=tot["clean"] / n, clean_share_of_readings=tot["clean"] / max(1, tot["clean"] + tot["C"]),
                median_image_clean_share=float(np.median(with_clean)) if with_clean else None,
                images_clean_share_ge_10pct=sum(1 for x in with_clean if x >= 0.1))


def part_a(datasets, hammer_per_group, n_mirror, n_window):
    out = {}
    for ds, per_group in (("nyu", None), ("hammer", hammer_per_group)):
        if ds not in datasets:
            continue
        processed, recs = records(ds, ["train"], per_group, scope="all")
        rows = []
        for r in recs:
            s = load_sample(processed / r["path"])
            rows.append(reading_shares(s.prompt, s.prompt_valid, s.gt, s.gt_valid, focus_scope(s)[1]))
        out[f"real_{ds}"] = pool_shares(rows)
    if n_mirror + n_window == 0:
        return out
    from scripts.hypersim_synthesis_stats import DATA, SAMPLE, SEED, SELECTION, TRAINING
    from tp.data.hypersim import MIRROR, WINDOW, load_frame
    from tp.sim.hypersim_condition import condition
    import yaml
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    min_mirror = float(yaml.safe_load(TRAINING.read_text(encoding="utf-8"))["min_mirror_share"])
    frames = [f for f in json.loads((DATA / "frames_v1.json").read_text(encoding="utf-8"))["frames"]
              if f["role"] != "mirror" or f["mirror"] >= min_mirror]
    rng = np.random.default_rng(SEED)
    sample = []                                   # exactly D3's list and order, so frame k keeps seed [SEED, k]
    for role, n in SAMPLE.items():
        pool = [f for f in frames if f["role"] == role]
        sample += [pool[i] for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))]
    wanted = {"mirror": n_mirror, "window": n_window}
    taken = defaultdict(int)
    synth = {"mirror": [], "window": []}
    for k, f in enumerate(sample):
        if f["role"] not in wanted or taken[f["role"]] >= wanted[f["role"]]:
            continue
        taken[f["role"]] += 1
        fr = load_frame(DATA, f["scene"], f["camera"], f["frame"], selection["scenes"][f["scene"]]["camera"]["M_cam_from_uv"])
        o = condition(fr, np.random.default_rng([SEED, k]))
        for label, key in ((MIRROR, "mirror"), (WINDOW, "window")):
            region = (fr.semantic == label) & fr.valid
            if region.mean() >= 0.002:
                synth[key].append(reading_shares(o["P_corrupt"], o["V_corrupt"], fr.depth, fr.valid, region))
    out.update({f"synthetic_{k}": pool_shares(v) for k, v in synth.items()})
    out["synthetic_frames"] = dict(taken)
    return out


def load_prediction(path):
    with np.load(path) as z:
        if "depth_q" in z.files:
            q = z["depth_q"].astype(np.float64)
            return np.where(q > 0, q * float(z["scale_m"]), np.nan)
        return z["depth"].astype(np.float64)


def cache_index(cache):
    rows = json.loads((ROOT / cache / "images.json").read_text(encoding="utf-8"))
    idx = {}
    for r in rows:
        name = r["prediction_path"].replace("\\", "/").rsplit("/", 1)[-1]
        idx[(r.get("sensor"), r["scene"], r["frame"])] = (ROOT / cache / "predictions" / name, r["sample_npz_sha256"])
    return idx


class Bins:
    def __init__(self):
        self.n = defaultdict(int); self.err = defaultdict(lambda: defaultdict(float)); self.behind = defaultdict(float)

    def add(self, key, errs, f1_signed):
        self.n[key] += errs["F1"].size
        for m in MODELS:
            self.err[key][m] += float(errs[m].sum())
        self.behind[key] += float(np.abs(f1_signed[f1_signed > 0]).sum())

    def summary(self):
        total = sum(self.n.values())
        return {k: dict(pixels=self.n[k], pixel_share=self.n[k] / total if total else None,
                        **{f"absrel_{m}": self.err[k][m] / self.n[k] for m in MODELS},
                        F1_error_share_behind_gt=self.behind[k] / self.err[k]["F1"] if self.err[k]["F1"] else None)
                for k in self.n if self.n[k]}


def part_b(datasets, hammer_per_group):
    from scipy.ndimage import distance_transform_edt
    out = {}
    for ds, per_group in (("nyu", None), ("hammer", hammer_per_group)):
        if ds not in datasets:
            continue
        processed, recs = records(ds, ["train", "val"], per_group)
        idx = {m: cache_index(CACHES[ds][m]) for m in MODELS}
        recs = [r for r in recs if all((r.get("sensor"), r["scene"], r["frame"]) in idx[m] for m in MODELS)]
        inside, outside = Bins(), Bins()
        per_image = []
        for r in recs:
            key = (r.get("sensor"), r["scene"], r["frame"])
            s = load_sample(processed / r["path"])
            preds = {}
            for m in MODELS:
                path, h = idx[m][key]
                if h != r["sha256"]:
                    raise ValueError(f"{m} cache row for {key} was made from another sample")
                preds[m] = load_prediction(path)
            reg = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=TAU, band=BAND)
            scope = focus_scope(s)[1]
            ok = np.all([np.isfinite(preds[m]) & (preds[m] > 0) for m in MODELS], axis=0)
            clean = reg["clean"] & ok
            dist = distance_transform_edt(~reg["C"]) if reg["C"].any() else np.full(s.gt.shape, np.inf)
            gt = s.gt.astype(np.float64)
            for bins, where in ((inside, clean & scope), (outside, clean & ~scope)):
                for lo, hi in DIST_BINS + [("no_C", None)]:
                    if lo == "no_C":
                        sel = where & np.isinf(dist)
                        name = "no C in image"
                    else:
                        sel = where & (dist >= lo) & (dist < (hi if hi else np.inf))
                        name = f"{lo}-{hi} px" if hi else f">= {lo} px"
                    if sel.any():
                        errs = {m: np.abs(preds[m][sel] - gt[sel]) / gt[sel] for m in MODELS}
                        bins.add(name, errs, (preds["F1"][sel] - gt[sel]) / gt[sel])
            sel = clean & scope
            if sel.any():
                e = {m: float(np.mean(np.abs(preds[m][sel] - gt[sel]) / gt[sel])) for m in MODELS}
                per_image.append(dict(scene=s.scene, frame=s.frame, sensor=s.sensor, pixels=int(sel.sum()), **e))
        inc = sorted((p["F1"] - p["R0"] for p in per_image), reverse=True)
        pos = sum(x for x in inc if x > 0)
        out[ds] = dict(images=len(recs), images_with_scope_clean=len(per_image), inside_scope=inside.summary(),
                       outside_scope=outside.summary(),
                       images_F1_worse_than_R0=sum(1 for x in inc if x > 0),
                       top3_share_of_increase=sum(inc[:3]) / pos if pos else None,
                       per_image=per_image)
    return out


CELL = 4                                          # OMNI-DC backbone_output_downsample_rate (input-confidence grid)


def cells(m, r=CELL):
    H, W = m.shape
    m = m[:H // r * r, :W // r * r]
    return m.reshape(H // r, r, W // r, r).any(axis=(1, 3))


def label_rates(dep, gt, labels, region=None):
    """Share of cells with a reading that each rule labels noisy (dep, gt in GT-median units, gt = 0 if invalid)."""
    has = cells(dep > 1e-4)
    rules = dict(f0_rule=cells((dep - gt) > 0.1), f1_rule=cells((labels == 1) | (labels == 2)))
    out = {k: float((v & has).sum() / max(1, has.sum())) for k, v in rules.items()}
    if region is not None:
        keep = has & ~cells(region)
        out.update({f"{k}_outside_scope": float((v & keep).sum() / max(1, keep.sum())) for k, v in rules.items()})
    return out


def part_d(datasets, hammer_per_group, n_frames):
    from scripts.hypersim_synthesis_stats import DATA, SAMPLE, SEED, SELECTION, TRAINING
    from tp.data.hypersim import load_frame
    import tp.omnidc.data as tdata
    from tp.sim.corrupt import type_labels
    import yaml
    out = {}
    if sum(n_frames.values()):
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        min_mirror = float(yaml.safe_load(TRAINING.read_text(encoding="utf-8"))["min_mirror_share"])
        frames = [f for f in json.loads((DATA / "frames_v1.json").read_text(encoding="utf-8"))["frames"]
                  if f["role"] != "mirror" or f["mirror"] >= min_mirror]
        rng = np.random.default_rng(SEED)
        sample = []
        for role, n in SAMPLE.items():
            pool = [f for f in frames if f["role"] == role]
            sample += [pool[i] for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))]
        noise = tdata.omnidc_noise
        rows = defaultdict(list)
        taken = defaultdict(int)
        for k, f in enumerate(sample):
            if taken[f["role"]] >= n_frames.get(f["role"], 0):
                continue
            taken[f["role"]] += 1
            fr = tdata.resize_frame(load_frame(DATA, f["scene"], f["camera"], f["frame"],
                                               selection["scenes"][f["scene"]]["camera"]["M_cam_from_uv"]))
            for iid in (True, False):
                tdata.omnidc_noise = noise if iid else (lambda P, V, rng, max_prob=0.05: (P.copy(), V.copy()))
                try:
                    for v in ("f0", "f1"):
                        for rep in range(3):
                            try:
                                smp = tdata.make_sample(fr, v, np.random.default_rng([SEED, k, rep]), p_sparse=0.0)
                            except ValueError:
                                continue
                            rows[f"{v} input, {f['role']} frames, {'with' if iid else 'without'} i.i.d. outliers"].append(
                                label_rates(smp["dep"][0], smp["gt"][0], smp["labels"]))
                finally:
                    tdata.omnidc_noise = noise
        out.update({k: dict(samples=len(v), **{m: float(np.mean([x[m] for x in v])) for m in v[0]}) for k, v in rows.items() if v})
    for ds, per_group in (("nyu", None), ("hammer", hammer_per_group)):
        if ds not in datasets:
            continue
        processed, recs = records(ds, ["train", "val"], per_group, scope="all")
        rows = []
        for r in recs:
            smp = load_sample(processed / r["path"])
            gt = np.where(smp.gt_valid, smp.gt, 0.0)
            med = float(np.median(gt[gt > 0]))
            dep = np.where(smp.prompt_valid, smp.prompt, 0.0) / med
            lab = type_labels(dep.astype(np.float32), smp.prompt_valid, (gt / med).astype(np.float32), smp.gt_valid)
            rows.append(label_rates(dep, gt / med, lab, region=focus_scope(smp)[1]))
        out[f"real {ds} dev images"] = dict(samples=len(rows), **{m: float(np.mean([x[m] for x in rows])) for m in rows[0]})
    return out


def table_a(a):
    lines = ["| source | images | region px | hole | wrong (C) | correct | correct among readings | median image correct share | images with >= 10 % correct |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, x in a.items():
        if isinstance(x, dict) and "images" in x:
            lines.append(f"| {k} | {x['images']} | {x['region_px']} | {x['hole_share']:.3f} | {x['C_share']:.3f} | "
                         f"{x['clean_share']:.3f} | {x['clean_share_of_readings']:.3f} | "
                         f"{'NA' if x['median_image_clean_share'] is None else format(x['median_image_clean_share'], '.3f')} | "
                         f"{x['images_clean_share_ge_10pct']} |")
    return lines


def table_b(name, bins):
    lines = [f"{name}:", "", "| distance to nearest C | pixel share | AbsRel R0 | AbsRel F0 | AbsRel F1 | F1 error behind GT |",
             "|---|---:|---:|---:|---:|---:|"]
    for k, x in bins.items():
        behind = "NA" if x["F1_error_share_behind_gt"] is None else f"{x['F1_error_share_behind_gt']:.2f}"
        lines.append(f"| {k} | {x['pixel_share']:.3f} | {x['absrel_R0']:.4f} | {x['absrel_F0']:.4f} | {x['absrel_F1']:.4f} | {behind} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["A", "B", "D"], choices=["A", "B", "D"])
    ap.add_argument("--label-frames", type=int, nargs=3, default=[20, 10, 10], metavar=("MIRROR", "WINDOW", "CLEAN"))
    ap.add_argument("--datasets", nargs="+", default=["nyu", "hammer"], choices=["nyu", "hammer"])
    ap.add_argument("--hammer-per-group", type=int, default=3)
    ap.add_argument("--hypersim-frames", type=int, nargs=2, default=[40, 20], metavar=("MIRROR", "WINDOW"))
    a = ap.parse_args()
    out = ROOT / "results" / f"p1_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, band=BAND, parts=a.part, datasets=a.datasets, label_frames=a.label_frames, hammer_per_group=a.hammer_per_group,
                                  hypersim_frames=a.hypersim_frames, caches=CACHES, dist_bins=DIST_BINS),
                        [sys.executable, "-m", "scripts.p1_diagnostics", *sys.argv[1:]], 20261025)
    try:
        res = {}
        if "A" in a.part:
            res["A"] = part_a(a.datasets, a.hammer_per_group, *a.hypersim_frames)
        if "B" in a.part:
            res["B"] = part_b(a.datasets, a.hammer_per_group)
        if "D" in a.part:
            res["D"] = part_d(a.datasets, a.hammer_per_group, dict(zip(("mirror", "window", "clean"), a.label_frames)))
        write_json(out / "diagnostics.json", res)
        lines = ["# P1 diagnostics after G3 failed (descriptive)", "",
                 f"Generated by `python -m scripts.p1_diagnostics` ({out.name}). tau = {TAU}; development data only, "
                 "test never read; no decision rule.", ""]
        if "A" in res:
            lines += ["## A. Readings inside the surface region (pixels with valid GT)", "",
                      f"Real: NYU train mirrors; HAMMER train objects ({a.hammer_per_group} frames per scene x sensor). "
                      f"Synthetic: first {res['A'].get('synthetic_frames')} frames of the D3 sample, same seeds.", "",
                      *table_a(res["A"]), ""]
        if "B" in res:
            lines += ["## B. Where the clean-pixel error comes from", "",
                      "Clean = correct readings outside the 5-pixel band around C (the gate's clean region); pixel-pooled AbsRel "
                      "(the gate uses scene means of image means, so the levels differ).", ""]
            for ds, x in res["B"].items():
                lines += [f"### {ds}: {x['images']} images; F1 worse than R0 on clean scope pixels in "
                          f"{x['images_F1_worse_than_R0']} of {x['images_with_scope_clean']}; top 3 images carry "
                          f"{'NA' if x['top3_share_of_increase'] is None else format(100 * x['top3_share_of_increase'], '.0f') + ' %'} "
                          "of the increase", ""]
                lines += table_b("Inside the main scope (the G3 pixels)", x["inside_scope"])
                lines += table_b("Outside the main scope (rest of the image)", x["outside_scope"])
        if "D" in res:
            lines += ["## D. Share of confidence cells labelled noisy", "",
                      f"1/{CELL}-resolution cells with a reading; F0 rule = OMNI-DC's original label, F1 rule = two-sided "
                      "label used to train F1. Synthetic rows are training inputs (dense); real rows are the dev images.", "",
                      "| input | samples | F0 rule | F1 rule | F0 rule outside scope | F1 rule outside scope |",
                      "|---|---:|---:|---:|---:|---:|"]
            for k, x in res["D"].items():
                g = lambda m: "-" if m not in x else f"{x[m]:.3f}"
                lines.append(f"| {k} | {x['samples']} | {g('f0_rule')} | {g('f1_rule')} | {g('f0_rule_outside_scope')} | "
                             f"{g('f1_rule_outside_scope')} |")
            lines.append("")
        (ROOT / "docs" / "omnidc_p1_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", diagnostics_sha256=sha256(out / "diagnostics.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
