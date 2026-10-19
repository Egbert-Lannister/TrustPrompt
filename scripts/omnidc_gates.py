"""Mainline P0-D1b (docs/plan_omnidc.md sec. 12): through which path do wrong readings reach the OMNI-DC output?
(GPU, Windows, .venvs/baselines; development diagnostics, test never read)

  python -m scripts.omnidc_gates

OMNI-DC v1.1, no retraining. Flagged prompt pixels F (GT flags: region C; deployable: T < conformal threshold)
are switched off on one path set at a time (tp/omnidc/gates.py):
  raw      nothing switched off
  ddi      depth-integration weight 0 on cells with a flagged pixel
  ddi_spn  + the DySPN refinement no longer re-injects flagged pixels
  mask     flagged pixels deleted from the prompt (closes the network-input path too; = tmask / oracle_mask)
Per image: main-scope C AbsRel and clean AbsRel; image -> scene -> dataset means.
Reading (fixed before running): with GT flags, S = (C_raw - C_ddi_spn) / (C_raw - C_mask) is the share of the
removable error closed without touching the network input.
  S >= 0.8  wrong readings act mainly through integration and refinement: module A gates DDI and SPN and keeps the
            prompt as a network input (no input masking);
  S <= 0.5  they also act through the features: module A must also gate the network input;
  between   both.
Clean AbsRel of ddi_spn vs mask is reported (does gating avoid masking's damage to correct pixels?).
Populations: NYU val + train-holdout (30); HAMMER all 10 dev scenes, 3 frames per scene x sensor (90; tp.data.select).
Writes results/omnidc_gates_<time>/ and docs/omnidc_gates.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from collections import defaultdict
from datetime import datetime
import json
import sys

import numpy as np

from baselines.completion import build
from scripts.cache_completion import CONFIG, crc_threshold, flagged, model_input, select_records
from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.consistency import main_region_absrel
from tp.omnidc.gates import cell_keep, gated
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.trust.cache import TrustCache

TAU = 0.05
SETS = {"nyu": dict(scope="holdout", per_group=None, trust="cache/trust_t_v1_real_sim_nofeat_nyu_dev_v1"),
        "hammer": dict(scope="all", per_group=3, trust="cache/trust_t_v1cf_real_sim_nofeat_hammer_dev_v1")}
PATHS = ("raw", "ddi", "ddi_spn", "mask")
SOURCES = {"gt": "oracle_mask", "t": "tmask"}       # flag source -> the masking variant with the same flags


def predict(model, s, path, variant="raw", trust=None, thr=None):
    """Prediction (HxW) of one path setting; `variant` (tmask / oracle_mask) defines the flagged pixels."""
    torch = model.torch
    if path in ("raw", "mask"):
        depth, _, _ = model_input(s, "raw" if path == "raw" else variant, tau=TAU, trust=trust, threshold=thr)
        return model.predict(s.rgb, depth, s.K_rgb)[0]
    depth, _, _ = model_input(s, "raw", tau=TAU)
    flag = flagged(s, variant, tau=TAU, trust=trust, threshold=thr)
    H, W = depth.shape
    ph, pw, _ = model.padding(depth.shape)
    f = torch.nn.functional.pad(torch.from_numpy(flag)[None, None].float(), (0, pw, 0, ph)).to(model.device)
    ddi = cell_keep(f, model.args.backbone_output_downsample_rate)
    spn = 1.0 - f if path == "ddi_spn" else None
    with gated(model.net, ddi_keep=ddi, spn_keep=spn):
        out, _ = model.outputs(s.rgb, depth)
    return out["pred"][0, 0, :H, :W].float().cpu().numpy()


def dataset_mean(rows, key):
    by_scene = defaultdict(list)
    for r in rows:
        if r[key] is not None:
            by_scene[r["scene"]].append(r[key])
    return float(np.mean([np.mean(v) for v in by_scene.values()])) if by_scene else None


def main():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / "results" / f"omnidc_gates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, sets=SETS, paths=PATHS, sources=SOURCES),
                        [sys.executable, "-m", "scripts.omnidc_gates", *sys.argv[1:]], int(cfg["seed"]))
    try:
        model = build("omnidc", device="cuda")
        receipt.update(model_deviations=model.deviations)
        payload = {}
        for ds, d in SETS.items():
            dcfg = cfg["datasets"][ds]
            processed = ROOT / dcfg["processed"]
            records = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"],
                                     {"train", "val"}, d["scope"])
            records = per_group_subsample(records, d["per_group"])
            tc = TrustCache(d["trust"])
            lam = None if tc.per_item else crc_threshold(cfg, dcfg["crc_dataset"])[0]
            rows = []
            for k, r in enumerate(records):
                path = processed / r["path"]; h = sha256(path)
                if h != r["sha256"]:
                    raise ValueError(f"checksum mismatch {path}")
                s = load_sample(path)
                trust = tc.trust(s, h, s.prompt.shape)
                thr = tc.threshold_for(s) if tc.per_item else lam
                row = dict(scene=s.scene, sensor=s.sensor, frame=s.frame, split=s.split)
                raw = predict(model, s, "raw")
                row["raw_C"], row["raw_clean"] = (main_region_absrel(s, raw, reg, TAU) for reg in ("C", "clean"))
                for src, variant in SOURCES.items():
                    for p in PATHS[1:]:
                        pred = predict(model, s, p, variant, trust, thr)
                        row[f"{src}_{p}_C"], row[f"{src}_{p}_clean"] = (main_region_absrel(s, pred, reg, TAU) for reg in ("C", "clean"))
                rows.append(row)
                if k % 20 == 0:
                    print(f"{ds} {k + 1}/{len(records)} {s.key}", flush=True)
            keys = ["raw_C", "raw_clean"] + [f"{src}_{p}_{m}" for src in SOURCES for p in PATHS[1:] for m in ("C", "clean")]
            means = {key: dataset_mean(rows, key) for key in keys}
            share = {}
            for src in SOURCES:
                denom = None if None in (means["raw_C"], means[f"{src}_mask_C"]) else means["raw_C"] - means[f"{src}_mask_C"]
                share[src] = None if not denom else (means["raw_C"] - means[f"{src}_ddi_spn_C"]) / denom
            payload[ds] = dict(images=len(rows), scenes=len({r["scene"] for r in rows}), means=means, share_closed=share, rows=rows)
            print(ds, json.dumps(dict(means=means, share=share)), flush=True)
        write_json(out / "gates.json", payload)
        fmt = lambda v: "NA" if v is None else f"{v:.4f}"
        lines = ["# Where wrong readings enter OMNI-DC (mainline P0-D1b)", "",
                 f"Generated by `python -m scripts.omnidc_gates` ({out.name}). OMNI-DC v1.1 without retraining; flagged "
                 "prompt pixels switched off on the integration (ddi), + refinement (ddi_spn), or deleted from the prompt "
                 "(mask, closes the network input too). Main-scope AbsRel, image -> scene -> dataset mean. Development "
                 "diagnostics; test never read.", ""]
        for ds, p in payload.items():
            m = p["means"]
            lines += [f"## {ds} ({p['images']} images, {p['scenes']} scenes)", "",
                      "| Flags | path | C AbsRel | clean AbsRel |", "|---|---|---:|---:|",
                      f"| - | raw | {fmt(m['raw_C'])} | {fmt(m['raw_clean'])} |"]
            for src in SOURCES:
                for pth in PATHS[1:]:
                    lines.append(f"| {'GT' if src == 'gt' else 'T'} | {pth} | {fmt(m[f'{src}_{pth}_C'])} | {fmt(m[f'{src}_{pth}_clean'])} |")
            lines += ["", "Share of the removable C error closed without touching the network input, "
                      "S = (raw - ddi_spn) / (raw - mask): " + ", ".join(
                          f"{'GT' if k == 'gt' else 'T'} flags {fmt(v)}" for k, v in p["share_closed"].items()), ""]
        lines += ["Reading fixed before running: GT-flag S >= 0.8 -> module A gates integration + refinement and keeps the "
                  "prompt as network input; S <= 0.5 -> module A must gate the network input too; between -> both.", ""]
        (ROOT / "docs" / "omnidc_gates.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", gates_sha256=sha256(out / "gates.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
