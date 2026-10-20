"""Mainline P0-D3 (docs/plan_omnidc.md sec. 7, 13): do the synthesised Hypersim errors look like the real ones?
(CPU; Windows .venv; development diagnostics, test never read)

  python -m scripts.hypersim_synthesis_stats [--previews 8]

Synthetic: a fixed-seed sample of the selected Hypersim frames (data/source_cache/hypersim_v1/frames_v1.json:
300 mirror, 150 window, 50 clean), each through tp.sim.hypersim_condition.condition with its own seed.
Real: NYU (Mirror3D) train split, mirror pixels; HAMMER train split, object pixels, 3 frames per scene x sensor.
Per image, inside the surface region (synthetic: the annotated mirror / window pixels; real: the main scope):
contaminated share C (|P - GT| / GT > tau), far / near share of C, hole share, median relative error of C, and C as
a share of the whole image.
Checks fixed before this script's first run (mirror frames vs NYU mirrors; written after a 24-frame calibration look,
docs/plan_omnidc.md sec. 15):
  R1  pooled far share of C >= 0.7
  R2  pooled median relative error of C within [0.5, 2] x NYU
  R3  hole share of the mirror pixels within +-0.2 of NYU's (real sensors return nothing on about half of a mirror)
  R4  median per-image C share of the image within [0.25, 4] x NYU
  plausible iff R1-R4 hold; otherwise the synthesis is revised before any training. Also reported: the share of
  mirror pixels whose reflection is outside the view (filled by the back-wall model or left as holes).
Window check (descriptive): share of window pixels whose Hypersim depth lies > 10 % behind the plane fitted to the
surrounding wall. A high share means Hypersim's depth already sees through the glass, and the window synthesis
(reading = glass depth x factor) must be revisited.
Writes results/hypersim_synthesis_<time>/ (stats.json, previews/) and docs/hypersim_synthesis.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys

import numpy as np

from tp.data.hypersim import MIRROR, WINDOW, load_frame
from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.sim.corrupt import _ring_plane_behind
from tp.sim.hypersim_condition import condition, regions

TAU = 0.05
SAMPLE = {"mirror": 300, "window": 150, "clean": 50}
SEED = 20261020
DATA = ROOT / "data" / "source_cache" / "hypersim_v1"
SELECTION = ROOT / "configs" / "hypersim" / "selection_v1.json"
REAL = {"nyu": dict(processed="data/processed/mirror3d_dev_v1", per_group=None),
        "hammer": dict(processed="data/processed/hammer_dev_v1", per_group=3)}


def region_stats(P, V, gt, gv, region):
    """Per-image numbers inside `region` (bool) for prompt P/V against GT."""
    both = region & V & gv
    rel = np.zeros(gt.shape); rel[both] = (P[both] - gt[both]) / gt[both]
    c = both & (np.abs(rel) > TAU)
    return dict(region_px=int(region.sum()), c_px=int(c.sum()), far_px=int((c & (rel > 0)).sum()),
                near_px=int((c & (rel < 0)).sum()), hole_px=int((region & gv & ~V).sum()),
                c_image_share=float(c.mean()), rel_err_C=np.abs(rel[c]).astype(np.float32))


def pooled(rows):
    c = sum(r["c_px"] for r in rows)
    errs = np.concatenate([r.pop("rel_err_C") for r in rows]) if rows else np.zeros(0)
    shares = [r["c_image_share"] for r in rows if r["c_px"] > 0]
    return dict(images=len(rows), c_px=c, far_share=sum(r["far_px"] for r in rows) / c if c else None,
                near_share=sum(r["near_px"] for r in rows) / c if c else None,
                hole_share=sum(r["hole_px"] for r in rows) / max(1, sum(r["region_px"] for r in rows)),
                median_rel_err_C=float(np.median(errs)) if errs.size else None,
                median_c_image_share=float(np.median(shares)) if shares else None)


def real_rows(name, d):
    from scripts.cache_completion import select_records
    processed = ROOT / d["processed"]
    records = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"],
                             {"train"}, "all")
    rows = []
    for r in per_group_subsample(records, d["per_group"]):
        s = load_sample(processed / r["path"])
        rows.append(region_stats(s.prompt, s.prompt_valid, s.gt, s.gt_valid, focus_scope(s)[1]))
    return rows


def see_through_share(frame, rng):
    """Window pixels whose depth is > 10 % behind the plane fitted to the wall ring around the window."""
    n_behind = n = 0
    for omega in regions(frame.semantic, WINDOW, frame.valid, 0.002):
        plane, src = _ring_plane_behind(np.where(frame.valid, frame.depth, 0), frame.valid, omega, rng)
        if src != "ring_plane":
            continue
        m = omega & frame.valid
        n += int(m.sum()); n_behind += int((frame.depth[m] > 1.1 * plane[m]).sum())
    return n_behind, n


def preview(path, frame, out):
    from PIL import Image
    lab = out["labels"]
    colours = np.array([[60, 160, 60], [220, 60, 60], [60, 90, 220], [30, 30, 30]], np.uint8)
    lab_rgb = np.where((lab < 4)[..., None], colours[np.minimum(lab, 3)], 255).astype(np.uint8)
    def depth_rgb(z):
        v = np.nan_to_num(z / np.nanpercentile(np.where(z > 0, z, np.nan), 98)).clip(0, 1)
        return (np.stack([v] * 3, -1) * 255).astype(np.uint8)
    top = np.concatenate([frame.rgb, depth_rgb(frame.depth)], 1)
    bottom = np.concatenate([depth_rgb(np.where(out["V_corrupt"], out["P_corrupt"], np.nan)), lab_rgb], 1)
    Image.fromarray(np.concatenate([top, bottom], 0)).reduce(2).save(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--previews", type=int, default=8)
    a = ap.parse_args()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    frames = json.loads((DATA / "frames_v1.json").read_text(encoding="utf-8"))["frames"]
    rng = np.random.default_rng(SEED)
    sample = []
    for role, n in SAMPLE.items():
        pool = [f for f in frames if f["role"] == role]
        sample += [pool[i] for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))]
    out_dir = ROOT / "results" / f"hypersim_synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out_dir, dict(tau=TAU, sample=SAMPLE, seed=SEED, real=REAL,
                                      frames_sha256=sha256(DATA / "frames_v1.json"), selection_sha256=sha256(SELECTION)),
                        [sys.executable, "-m", "scripts.hypersim_synthesis_stats", *sys.argv[1:]], SEED)
    try:
        (out_dir / "previews").mkdir()
        synth = {"mirror": [], "window": []}
        unresolved = resolved = behind = n_window = 0
        shown = dict.fromkeys(SAMPLE, 0)
        for k, f in enumerate(sample):
            fr = load_frame(DATA, f["scene"], f["camera"], f["frame"], selection["scenes"][f["scene"]]["camera"]["M_cam_from_uv"])
            out = condition(fr, np.random.default_rng([SEED, k]))
            for reg in out["info"]["regions"]:
                if reg.get("type") == "reflection" and "resolved" in reg:
                    resolved += reg["resolved"]; unresolved += reg["unresolved"]
            for label, key in ((MIRROR, "mirror"), (WINDOW, "window")):
                region = (fr.semantic == label) & fr.valid
                if region.mean() >= 0.002:
                    synth[key].append(region_stats(out["P_corrupt"], out["V_corrupt"], fr.depth, fr.valid, region))
            b, n = see_through_share(fr, np.random.default_rng([SEED, k, 1]))
            behind += b; n_window += n
            if shown[f["role"]] < a.previews:
                shown[f["role"]] += 1
                preview(out_dir / "previews" / f"{f['role']}_{f['scene']}_{f['camera']}_{f['frame']:04d}.png", fr, out)
            if k % 50 == 0:
                print(f"synthetic {k + 1}/{len(sample)}", flush=True)
        stats = {f"synthetic_{k}": pooled(v) for k, v in synth.items()}
        for name, d in REAL.items():
            stats[f"real_{name}"] = pooled(real_rows(name, d)); print(name, stats[f"real_{name}"], flush=True)
        stats["unresolved_share"] = unresolved / max(1, resolved + unresolved)
        stats["window_see_through_share"] = behind / n_window if n_window else None
        m, nyu = stats["synthetic_mirror"], stats["real_nyu"]
        checks = dict(R1=m["far_share"] is not None and m["far_share"] >= 0.7,
                      R2=None not in (m["median_rel_err_C"], nyu["median_rel_err_C"])
                      and 0.5 <= m["median_rel_err_C"] / nyu["median_rel_err_C"] <= 2,
                      R3=abs(m["hole_share"] - nyu["hole_share"]) <= 0.2,
                      R4=None not in (m["median_c_image_share"], nyu["median_c_image_share"])
                      and 0.25 <= m["median_c_image_share"] / nyu["median_c_image_share"] <= 4)
        stats.update(checks=checks, plausible=all(checks.values()))
        write_json(out_dir / "stats.json", stats)
        fmt = lambda v, p=3: "NA" if v is None else f"{v:.{p}f}"
        lines = ["# Synthesised Hypersim errors vs real sensor errors (mainline P0-D3)", "",
                 f"Generated by `python -m scripts.hypersim_synthesis_stats` ({out_dir.name}). Synthetic: {len(sample)} "
                 "fixed-seed Hypersim frames through tp.sim.hypersim_condition; real: NYU train mirrors, HAMMER train "
                 "objects (3 frames per scene x sensor). Inside the surface region; C = |P - GT| / GT > 0.05. "
                 "Development diagnostics; test never read.", "",
                 "| Source | images | C px | far share of C | near share of C | hole share | median rel. error of C | median C share of image |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, x in stats.items():
            if isinstance(x, dict) and "images" in x:
                lines.append(f"| {key} | {x['images']} | {x['c_px']} | {fmt(x['far_share'])} | {fmt(x['near_share'])} | "
                             f"{fmt(x['hole_share'])} | {fmt(x['median_rel_err_C'])} | {fmt(x['median_c_image_share'], 4)} |")
        lines += ["", f"- unresolved share of mirror pixels: {fmt(stats['unresolved_share'])}",
                  f"- window pixels > 10 % behind the surrounding wall plane in Hypersim's own depth: "
                  f"{fmt(stats['window_see_through_share'])}",
                  "- checks fixed before running: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in checks.items()),
                  f"- **{'plausible: use this synthesis in P1' if stats['plausible'] else 'NOT plausible: revise the synthesis before training'}**",
                  f"- previews (RGB | GT, synthetic prompt | labels: green clean, red far, blue near, black hole): "
                  f"results/{out_dir.name}/previews/", ""]
        (ROOT / "docs" / "hypersim_synthesis.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", stats_sha256=sha256(out_dir / "stats.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out_dir / "receipt.json", receipt)


if __name__ == "__main__":
    main()
