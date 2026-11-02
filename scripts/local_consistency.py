"""Supplementary experiment S1 (docs/plan_supplement.md): is the prior's ability to spot wrong readings global or
local? A direct test of the "local self-consistency" explanation in docs/analysis_paper.md sec. 6. CPU; reuses the
frozen, globally aligned UniDepthV2-L prior recorded by the CAPA pre-check (step 0 of
results/tta_precheck_20261101_012932) on the same 125 development images; test never read.

  python -m scripts.local_consistency

For every pixel, the prior D is re-fitted to the readings P inside a w x w window by least squares
(P ~ a D + b over the window's valid readings; box filters), and the local residual r_w = |a D + b - P| / P is used
as the score. w = inf is the global fit already applied (d = |D - P| / P). AUROC: wrong (C) vs correct readings in
the main scope (NYU mirrors, HAMMER objects), tau 0.05. If wrong readings are locally self-consistent, a local fit
explains them about as well as correct readings and the AUROC falls toward 0.5 as the window shrinks.

PRE-REGISTERED RULE (fixed before the run), NYU (large coherent regions):
  L1  AUROC(r_31) <= 0.60 and AUROC(d) - AUROC(r_31) >= 0.15 with scene-bootstrap CI low > 0 (2000 draws, seed 20261103);
  L2  AUROC does not decrease with the window: AUROC(r_15) <= AUROC(r_31) + 0.01 <= ... over w = 15, 31, 63, 127, inf
      (each step may drop by at most 0.01).
  supported = L1 and L2; otherwise not supported. HAMMER (small objects, windows reach the background) and the
  per-sensor split are descriptive, as is the CAPA inheritance-vs-area relation (Spearman of per-image IR at step 100
  against the C area share).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from scripts import tta_precheck as T

TAU, BAND, SEED, DRAWS = 0.05, 5, 20261103, 2000
WINDOWS = (15, 31, 63, 127)
CAPA_RUN = "results/tta_precheck_20261101_012932"


def box_sum(x, w):
    """Sum of x over the w x w window centred at every pixel (zero padding)."""
    r = w // 2
    p = np.pad(x, ((r + 1, r), (r + 1, r)))
    s = p.cumsum(0).cumsum(1)
    h, wd = x.shape
    return s[w:w + h, w:w + wd] - s[0:h, w:w + wd] - s[w:w + h, 0:wd] + s[0:h, 0:wd]


def local_fit_residual(D, P, valid, w, eps=1e-9):
    """|a D + b - P| / P with (a, b) the least-squares fit of P on D over the window's valid readings.
    A window with fewer than 3 readings or a flat prior falls back to a shift-only fit (a = 1)."""
    v = valid.astype(np.float64)
    Dv, Pv = np.where(valid, D, 0.0).astype(np.float64), np.where(valid, P, 0.0).astype(np.float64)
    S1, Sd, Sp = box_sum(v, w), box_sum(Dv, w), box_sum(Pv, w)
    Sdd, Sdp = box_sum(Dv * Dv, w), box_sum(Dv * Pv, w)
    den = S1 * Sdd - Sd * Sd
    ok = (S1 >= 3) & (den > eps * np.maximum(S1 * Sdd, eps))
    a = np.where(ok, (S1 * Sdp - Sd * Sp) / np.where(ok, den, 1.0), 1.0)
    b = np.where(S1 > 0, (Sp - a * Sd) / np.maximum(S1, 1.0), 0.0)
    Ps = np.where(P > 0, P, 1.0)
    return np.abs(a * D + b - P) / Ps


def image_stats(D, P, pv, G, gv, scope):
    R = regions(P, pv, G, gv, tau=TAU, band=BAND)
    valid = pv & (P > 0) & np.isfinite(D) & (D > 0)
    pos, neg = R["C"] & scope & valid, R["clean_all"] & scope & valid
    Ps = np.where(P > 0, P, 1.0)
    scores = {"inf": np.abs(D - P) / Ps}
    for w in WINDOWS:
        scores[str(w)] = local_fit_residual(D, P, valid, w)
    return dict(pos=int(pos.sum()), neg=int(neg.sum()),
                hists={k: dict(pos=T.hist(s[pos]).tolist(), neg=T.hist(s[neg]).tolist()) for k, s in scores.items()})


def auc(xs, k):
    return T.auroc_hist(np.sum([x["stats"]["hists"][k]["pos"] for x in xs], 0),
                        np.sum([x["stats"]["hists"][k]["neg"] for x in xs], 0))


def summarise(rows, capa_rows, rng):
    from scipy.stats import spearmanr
    keys = [str(w) for w in WINDOWS] + ["inf"]
    res = {}
    for ds in T.DATASETS:
        xs = [r for r in rows if r["dataset"] == ds]
        sc = T._by_scene(xs)
        flat = lambda s: [x for g in s for x in g]
        d31 = T.scene_bootstrap(sc, lambda s: (lambda a, b: None if a is None or b is None else a - b)(
            auc(flat(s), "inf"), auc(flat(s), "31")), rng)
        out = dict(images=len(xs), scenes=len(sc), pos=sum(x["stats"]["pos"] for x in xs),
                   neg=sum(x["stats"]["neg"] for x in xs), auroc={k: auc(xs, k) for k in keys}, delta_inf_31=d31)
        if ds == "hammer":
            out["by_sensor"] = {s: {k: auc([x for x in xs if x["sensor"] == s], k) for k in keys}
                                for s in ("d435", "l515", "helios")}
        cr = [c for c in capa_rows if c["dataset"] == ds and c["stats"]["C"] > 0]
        area = [c["stats"]["C"] / c["pixels"] for c in cr]
        ir = [c["stats"]["steps"]["100"]["inherited"] / c["stats"]["C"] for c in cr]
        ir0 = [c["stats"]["steps"]["0"]["inherited"] / c["stats"]["C"] for c in cr]
        out["capa_ir_vs_area"] = dict(images=len(cr), spearman_ir100=float(spearmanr(area, ir).statistic) if len(cr) > 2 else None,
                                      spearman_ir0=float(spearmanr(area, ir0).statistic) if len(cr) > 2 else None,
                                      median_area=float(np.median(area)) if area else None)
        res[ds] = out
    return res


def verdict(res):
    a = res["nyu"]["auroc"]
    p, lo, _ = res["nyu"]["delta_inf_31"]
    L1 = a["31"] is not None and a["31"] <= 0.60 and p is not None and p >= 0.15 and lo > 0
    seq = [a[str(w)] for w in WINDOWS] + [a["inf"]]
    L2 = all(x is not None for x in seq) and all(seq[i + 1] >= seq[i] - 0.01 for i in range(len(seq) - 1))
    return dict(L1=L1, L2=L2, outcome="supported" if L1 and L2 else "not supported")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run = ROOT / CAPA_RUN
    rec = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    if rec.get("stage") != "complete" or sha256(run / "tta_precheck.json") != rec["result_sha256"]:
        raise SystemExit(f"{CAPA_RUN} is not a complete, untouched CAPA run")
    config = dict(tau=TAU, band=BAND, seed=SEED, draws=DRAWS, windows=list(WINDOWS), capa_run=CAPA_RUN,
                  capa_result_sha256=rec["result_sha256"], limit=a.limit, fit="least squares P ~ a D + b, box windows")
    out = ROOT / "results" / f"local_consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, config, [sys.executable, "-m", "scripts.local_consistency", *sys.argv[1:]], SEED)
    try:
        rows, capa_rows = [], []
        todo = [(ds, r) for ds in T.DATASETS for r in T.select(ds, T.DATASETS[ds]["per_group"], a.limit)]
        for i, (ds, r) in enumerate(todo):
            name = f"{ds}__{Path(r['path']).stem}"
            path = ROOT / T.DATASETS[ds]["processed"] / r["path"]
            if sha256(path) != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            s = load_sample(path)
            if s.split == "test":
                raise ValueError("test sample reached the experiment")
            D = np.load(run / "predictions" / f"{name}.npz")["step_0"].astype(np.float64)
            _, scope = focus_scope(s)
            rows.append(dict(dataset=ds, scene=s.scene, sensor=s.sensor, name=name,
                             stats=image_stats(D, s.prompt.astype(np.float64), s.prompt_valid, s.gt, s.gt_valid, scope)))
            c = json.loads((run / "per_image" / f"{name}.json").read_text(encoding="utf-8"))
            capa_rows.append(dict(dataset=ds, stats=c["stats"], pixels=int(s.gt.size)))
            if (i + 1) % 25 == 0:
                print(f"[{i + 1}/{len(todo)}]", flush=True)
        res = summarise(rows, capa_rows, np.random.default_rng(SEED))
        ver = verdict(res) if a.limit is None else dict(outcome="engineering check (no verdict)")
        write_json(out / "local_consistency.json", dict(results=res, verdict=ver))
        f = lambda x: "NA" if x is None else f"{x:.3f}"
        keys = [str(w) for w in WINDOWS] + ["inf"]
        L = ["# S1: is the prior's signal global or local? (pre-registered, development data)", "",
             f"Generated by `python -m scripts.local_consistency` ({out.name}); rule fixed in docs/plan_supplement.md and the "
             f"script docstring before the run. Prior = frozen UniDepthV2-L, globally aligned (step 0 of {CAPA_RUN}). "
             "Test never read.", "", f"**Outcome (NYU rule): {ver['outcome']}**" +
             (f" (L1 {ver['L1']}, L2 {ver['L2']})" if "L1" in ver else "") + ".", "",
             "AUROC of the residual after a local least-squares fit in a w x w window (inf = global fit only), wrong vs "
             "correct readings, main scope:", "",
             "| dataset | wrong / correct readings | " + " | ".join(f"w={k}" for k in keys) + " | AUROC(inf) - AUROC(31) (95 % CI) |",
             "|---|---:|" + "---:|" * (len(keys) + 1)]
        for ds, x in res.items():
            dd = x["delta_inf_31"]
            L.append(f"| {ds} | {x['pos']} / {x['neg']} | " + " | ".join(f(x["auroc"][k]) for k in keys) +
                     f" | {f(dd[0])} ({f(dd[1])} to {f(dd[2])}) |")
        L += ["", "HAMMER by sensor (descriptive):", "", "| sensor | " + " | ".join(f"w={k}" for k in keys) + " |",
              "|---|" + "---:|" * len(keys)]
        for sn, v in res["hammer"].get("by_sensor", {}).items():
            L.append(f"| {sn} | " + " | ".join(f(v[k]) for k in keys) + " |")
        L += ["", "CAPA inheritance vs contaminated area (descriptive; per image, C pixels / image pixels):", "",
              "| dataset | images | Spearman(area, IR step 100) | Spearman(area, IR step 0) | median area |", "|---|---:|---:|---:|---:|"]
        for ds, x in res.items():
            c = x["capa_ir_vs_area"]
            L.append(f"| {ds} | {c['images']} | {f(c['spearman_ir100'])} | {f(c['spearman_ir0'])} | {f(c['median_area'])} |")
        L.append("")
        doc = ROOT / "docs" / ("local_consistency.md" if a.limit is None else "local_consistency_engineering.md")
        doc.write_text("\n".join(L), encoding="utf-8")
        receipt.update(stage="complete" if a.limit is None else "engineering_check_complete", verdict=ver,
                       result_sha256=sha256(out / "local_consistency.json"), doc=doc.relative_to(ROOT).as_posix())
        print("\n".join(L))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
