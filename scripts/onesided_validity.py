"""Is the one-sided constraint "true depth <= reading" right on the pixels T flags? (CPU; dev diagnostics)

  python -m scripts.onesided_validity [--budget SECONDS]

The guided Marigold-DC variants (baselines/guidance.py; tonesided, tsoft) replace the equality fit on flagged
prompt pixels (T < conformal threshold) by a hinge that only forbids predictions behind the reading. That is right
where the reading is too far (mirror: reflected scene; glass: background) and wrong where it is too near. Per
dataset, on flagged pixels with valid GT, inside the main scope and on the whole image:
  share_le        GT <= P                    the hinge never pushes against the truth
  share_le_tau    GT <= (1 + tau) P          the same within the contamination tolerance
  far / near      P > (1 + tau) GT / P < (1 - tau) GT among the flagged pixels (the rest are clean)
Populations and T as in the guided caches: NYU val + train-holdout scenes with T v1 (global threshold); HAMMER all
dev scenes with the cross-fitted T (per-item thresholds). Pixel shares pooled over images and as the mean of
per-scene shares. Per-image counts go to results/onesided_validity_parts/<dataset>.jsonl, so a run stopped by
--budget resumes there; once every image is counted, writes results/onesided_validity_<time>/ and
docs/onesided_validity.md. Test never read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import sys
import time

import numpy as np

from tp.data.schema import load_sample
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.trust.cache import TrustCache

TAU = 0.05
SETS = {
    "nyu": dict(processed="data/processed/mirror3d_dev_v1", trust="cache/trust_t_v1_real_sim_nofeat_nyu_dev_v1",
                crc=("results/trust_t_v1_crc_d20.json", "mirror3d_nyu", "0.1"), scope="holdout"),
    "hammer": dict(processed="data/processed/hammer_dev_v1", trust="cache/trust_t_v1cf_real_sim_nofeat_hammer_dev_v1",
                   crc=None, scope="all"),
}
KEYS = ("flagged", "le", "le_tau", "far", "near")
PARTS = ROOT / "results" / "onesided_validity_parts"


def counts(s, trust, threshold, where):
    m = s.prompt_valid & s.gt_valid & (trust < threshold) & where
    p, g = s.prompt[m].astype(np.float64), s.gt[m].astype(np.float64)
    return dict(flagged=int(m.sum()), le=int((g <= p).sum()), le_tau=int((g <= (1 + TAU) * p).sum()),
                far=int((p > (1 + TAU) * g).sum()), near=int((p < (1 - TAU) * g).sum()))


def shares(c):
    n = c["flagged"]
    return {k: (c[k] / n if n else None) for k in KEYS[1:]} | dict(flagged=n)


def summarise(rows, region):
    pooled = {k: sum(r[region][k] for r in rows) for k in KEYS}
    by_scene = defaultdict(lambda: dict.fromkeys(KEYS, 0))
    for r in rows:
        for k in KEYS:
            by_scene[r["scene"]][k] += r[region][k]
    per_scene = {sc: shares(c) for sc, c in sorted(by_scene.items())}
    scene_mean = {k: float(np.mean(v)) if (v := [x[k] for x in per_scene.values() if x[k] is not None]) else None
                  for k in KEYS[1:]}
    return dict(pooled=shares(pooled), scene_mean=scene_mean, per_scene=per_scene)


def count_dataset(ds, d, deadline):
    """Per-image rows of one dataset, resumed from and appended to its parts file; None if the budget ran out."""
    from scripts.cache_completion import select_records
    processed = ROOT / d["processed"]
    records = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"]
    records = select_records(records, {"train", "val"}, d["scope"])
    tc = TrustCache(d["trust"])
    lam = None
    if d["crc"]:
        f, key, alpha = d["crc"]
        lam = float(json.loads((ROOT / f).read_text(encoding="utf-8"))["datasets"][key]["alphas"][alpha]["lambda_hat"])
    parts = PARTS / f"{ds}.jsonl"
    done = {}
    if parts.is_file():
        for line in parts.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); done[row["sample_sha256"]] = row
    PARTS.mkdir(parents=True, exist_ok=True)
    with parts.open("a", encoding="utf-8") as fh:
        for r in records:
            if r["sha256"] in done:
                continue
            if time.monotonic() > deadline:
                return None
            path = processed / r["path"]
            h = sha256(path)
            if h != r["sha256"]:
                raise ValueError(f"checksum mismatch {path}")
            s = load_sample(path)
            thr = tc.threshold_for(s) if tc.per_item else lam
            trust = tc.trust(s, h, s.prompt.shape)
            row = dict(sample_sha256=h, scene=s.scene, sensor=s.sensor, frame=s.frame, split=s.split, threshold=thr,
                       trust_index_sha256=tc.receipt["index_sha256"], main=counts(s, trust, thr, focus_scope(s)[1]),
                       full=counts(s, trust, thr, np.ones(s.prompt.shape, bool)))
            fh.write(json.dumps(row) + "\n"); fh.flush(); done[h] = row
    rows = [done[r["sha256"]] for r in records]
    if any(row["trust_index_sha256"] != tc.receipt["index_sha256"] for row in rows):
        raise ValueError(f"{parts} was counted with a different trust cache; rename it and re-run")
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=None, help="seconds")
    a = ap.parse_args()
    deadline = time.monotonic() + (a.budget or float("inf"))
    all_rows = {}
    for ds, d in SETS.items():
        rows = count_dataset(ds, d, deadline)
        if rows is None:
            print(f"PARTIAL: budget used up in {ds}; re-run to continue", flush=True)
            return
        all_rows[ds] = rows
    out = ROOT / "results" / f"onesided_validity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, sets=SETS), [sys.executable, "-m", "scripts.onesided_validity", *sys.argv[1:]], 0)
    try:
        payload = {}
        for ds, rows in all_rows.items():
            payload[ds] = dict(images=len(rows), main=summarise(rows, "main"), full=summarise(rows, "full"), rows=rows)
            print(ds, json.dumps({k: payload[ds][k]["pooled"] for k in ("main", "full")}), flush=True)
        write_json(out / "validity.json", payload)
        fmt = lambda v: "NA" if v is None else f"{100 * v:.1f}%"
        lines = ["# One-sided constraint validity on T-flagged prompt pixels", "",
                 f"Generated by `python -m scripts.onesided_validity` ({out.name}). Flagged = valid prompt and GT, "
                 f"T < threshold. `GT <= P`: the hinge (prediction not behind the reading) never fights the truth; "
                 f"tau = {TAU}. Development diagnostics; test never read.", "",
                 "| Dataset | Region | images | flagged px | GT <= P | GT <= (1+tau)P | far | near | scene-mean GT <= P |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for ds, p in payload.items():
            for region in ("main", "full"):
                x, sm = p[region]["pooled"], p[region]["scene_mean"]
                lines.append(f"| {ds} | {region} | {p['images']} | {x['flagged']} | {fmt(x['le'])} | {fmt(x['le_tau'])} | "
                             f"{fmt(x['far'])} | {fmt(x['near'])} | {fmt(sm['le'])} |")
        lines += ["", "HAMMER per scene (main scope):", "", "| Scene | flagged px | GT <= P | far | near |", "|---|---:|---:|---:|---:|"]
        for sc, x in payload["hammer"]["main"]["per_scene"].items():
            lines.append(f"| {sc} | {x['flagged']} | {fmt(x['le'])} | {fmt(x['far'])} | {fmt(x['near'])} |")
        (ROOT / "docs" / "onesided_validity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", validity_sha256=sha256(out / "validity.json"))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
