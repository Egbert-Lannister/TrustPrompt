"""Direction 1 pre-check (docs/plan_tta.md sec. 4): can the frozen prior's own uncertainty bound how far test-time
adaptation may move away from it? GPU for one frozen UniDepthV2-L forward per image (no adaptation); development
data only, test never read. Reuses the CAPA run results/tta_precheck_20261101_012932 (its step-0 and step-100
predictions) and the same 125 images.

  python -m scripts.tr_precheck                  # NYU 95 + HAMMER 30
  python -m scripts.tr_precheck --limit 2        # engineering check

UniDepthV2 outputs Sigma ("confidence" in the code), trained with an L1 loss on the log-space depth error
(UniDepthV2 paper, Fig. 2), i.e. u = predicted |log D - log GT|: larger = less certain.

Trust-region emulation, training free, from the recorded CAPA run (D_p = aligned frozen prior = CAPA step 0,
D_c = CAPA step 100):  D_tr(k) = D_p + clip(D_c - D_p, -k u D_p, +k u D_p).
Control with the same average width but no per-pixel information: D_const(k) uses u replaced by the image median
of u. k = 2 is primary; k = 1, 3 are descriptive.

PRE-REGISTERED RULES (fixed before any run; main scope; tau 0.05, band 5; scene bootstrap 2000 draws, seed 20261101):
  R0 semantics gate: Spearman(u, |D_p - GT| / GT) > 0.1 on both datasets (pixels with valid GT, up to 20000 per image).
  R1 trust region works, on both datasets:
     C reduction  = 1 - AbsRel_C(D_tr2) / AbsRel_C(D_c) >= 0.30 with CI low > 0, and
     clean retention = (AbsRel_clean(D_p) - AbsRel_clean(D_tr2)) / (AbsRel_clean(D_p) - AbsRel_clean(D_c)) >= 0.50.
  R2 per-pixel uncertainty matters, on both datasets:
     1 - AbsRel_C(D_tr2) / AbsRel_C(D_const2) has CI low > 0, and AbsRel_clean(D_tr2) <= 1.10 AbsRel_clean(D_const2).
  go = R0 and R1 and R2;  "fixed_region_enough" = R0 and R1 without R2;  otherwise "fails".
  Descriptive: k = 1, 3; full-image regions; AUROC of d = |D_p - P| / P, u and z = d / u for wrong vs correct readings.
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

from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from scripts import tta_precheck as T

TAU, BAND, SEED, DRAWS = 0.05, 5, 20261101, 2000
KS = (1.0, 2.0, 3.0)
PRIMARY_K = 2.0
CAPA_RUN = "results/tta_precheck_20261101_012932"
SPEARMAN_PER_IMAGE = 20000
VARIANTS = ("prior", "capa", "tr1", "tr2", "tr3", "const2")


# ---------------------------------------------------------------- pure numpy (unit-tested)

def trust_region(Dp, Dc, u, k):
    w = k * u * Dp
    return Dp + np.clip(Dc - Dp, -w, w)


def variants(Dp, Dc, u):
    med = float(np.nanmedian(u[np.isfinite(u)])) if np.isfinite(u).any() else np.nan
    out = dict(prior=Dp, capa=Dc)
    for k in KS:
        out[f"tr{int(k)}"] = trust_region(Dp, Dc, u, k)
    out["const2"] = trust_region(Dp, Dc, np.full_like(u, med), PRIMARY_K)
    return out


def hist_range(x, lo, hi, bins=1200):
    v = np.log10(np.clip(np.asarray(x, np.float64), 10 ** lo, 10 ** hi))
    return np.histogram(v, bins=bins, range=(lo, hi))[0].astype(np.int64)


def image_stats(P, pv, G, gv, scope, preds, u, rng):
    R = regions(P, pv, G, gv, tau=TAU, band=BAND)
    masks = dict(C=R["C"] & scope, clean=R["clean"] & scope, C_full=R["C"], clean_full=R["clean"])
    Gs = np.where(G > 0, G, 1.0)
    out = dict(counts={k: int(m.sum()) for k, m in masks.items()}, absrel={})
    for name, D in preds.items():
        valid = np.isfinite(D) & (D > 0)
        rel = np.abs(D - G) / Gs
        out["absrel"][name] = {k: [float(rel[m & valid].sum()), int((m & valid).sum())] for k, m in masks.items()}
        out["absrel"][name]["invalid"] = int((gv & ~valid).sum())
    Dp = preds["prior"]
    ok = gv & np.isfinite(Dp) & (Dp > 0) & np.isfinite(u) & (u > 0)
    idx = np.flatnonzero(ok)
    if idx.size > SPEARMAN_PER_IMAGE:
        idx = rng.choice(idx, SPEARMAN_PER_IMAGE, replace=False)
    out["spearman_sample"] = dict(u=u.ravel()[idx].astype(np.float32).tolist(),
                                  e=(np.abs(Dp - G) / Gs).ravel()[idx].astype(np.float32).tolist())
    Ps = np.where(P > 0, P, 1.0)
    d = np.abs(Dp - P) / Ps
    pos = masks["C"] & ok
    neg = R["clean_all"] & scope & ok
    out["auroc_hists"] = {}
    for name, s, lo, hi in (("d", d, -5, 1), ("u", u, -5, 1), ("z", d / np.where(u > 0, u, 1.0), -4, 4)):
        out["auroc_hists"][name] = dict(pos=hist_range(s[pos], lo, hi).tolist(), neg=hist_range(s[neg], lo, hi).tolist())
    return out


def _pooled(xs, variant, region):
    s = sum(x["stats"]["absrel"][variant][region][0] for x in xs)
    n = sum(x["stats"]["absrel"][variant][region][1] for x in xs)
    return None if n == 0 else s / n


def _flat(scenes):
    return [x for s in scenes for x in s]


def c_reduction(scenes, a="tr2", b="capa"):
    xs = _flat(scenes)
    A, B = _pooled(xs, a, "C"), _pooled(xs, b, "C")
    return None if A is None or not B else 1 - A / B


def clean_retention(scenes, k="tr2"):
    xs = _flat(scenes)
    p, c, t = _pooled(xs, "prior", "clean"), _pooled(xs, "capa", "clean"), _pooled(xs, k, "clean")
    return None if None in (p, c, t) or p - c <= 0 else (p - t) / (p - c)


def spearman(u, e):
    from scipy.stats import spearmanr
    if len(u) < 3:
        return None
    v = float(spearmanr(u, e).statistic)
    return v if np.isfinite(v) else None


def summarise(rows, rng):
    res = {}
    for ds in T.DATASETS:
        xs = [r for r in rows if r["dataset"] == ds]
        if not xs:
            continue
        sc = T._by_scene(xs)
        u = np.concatenate([x["stats"]["spearman_sample"]["u"] for x in xs])
        e = np.concatenate([x["stats"]["spearman_sample"]["e"] for x in xs])
        auc = {}
        for name in ("d", "u", "z"):
            auc[name] = T.auroc_hist(np.sum([x["stats"]["auroc_hists"][name]["pos"] for x in xs], 0),
                                     np.sum([x["stats"]["auroc_hists"][name]["neg"] for x in xs], 0))
        res[ds] = dict(
            images=len(xs), scenes=len(sc),
            absrel={v: {reg: _pooled(xs, v, reg) for reg in ("C", "clean", "C_full", "clean_full")} for v in VARIANTS},
            invalid={v: sum(x["stats"]["absrel"][v]["invalid"] for x in xs) for v in VARIANTS},
            spearman_u_err=spearman(u, e),
            c_reduction={k: T.scene_bootstrap(sc, lambda s, k=k: c_reduction(s, f"tr{int(k)}"), rng) for k in KS},
            clean_retention={k: clean_retention(sc, f"tr{int(k)}") for k in KS},
            c_vs_const=T.scene_bootstrap(sc, lambda s: c_reduction(s, "tr2", "const2"), rng),
            auroc=auc, align_check_median_rel_diff_max=max(x["align_check"] for x in xs),
            seconds_median=float(np.median([x["seconds"] for x in xs])))
    return res


def verdict(res):
    r0 = {ds: x["spearman_u_err"] is not None and x["spearman_u_err"] > 0.1 for ds, x in res.items()}
    r1 = {}
    r2 = {}
    for ds, x in res.items():
        p, lo, _ = x["c_reduction"][PRIMARY_K]
        ret = x["clean_retention"][PRIMARY_K]
        r1[ds] = p is not None and p >= 0.30 and lo > 0 and ret is not None and ret >= 0.50
        cp, clo, _ = x["c_vs_const"]
        a, b = x["absrel"]["tr2"]["clean"], x["absrel"]["const2"]["clean"]
        r2[ds] = cp is not None and clo > 0 and a is not None and b is not None and a <= 1.10 * b
    full = len(res) == len(T.DATASETS)
    R0, R1, R2 = full and all(r0.values()), full and all(r1.values()), full and all(r2.values())
    return dict(R0=R0, R1=R1, R2=R2, by_dataset=dict(R0=r0, R1=r1, R2=r2),
                outcome="go" if R0 and R1 and R2 else ("fixed_region_enough" if R0 and R1 else "fails"))


# ---------------------------------------------------------------- driver

def report(out, res, ver, args):
    f = lambda x, d=3: "NA" if x is None else f"{x:.{d}f}"
    ci = lambda t: f"{f(t[0])} ({f(t[1])} to {f(t[2])})"
    L = ["# Direction 1 pre-check: prior-uncertainty trust region (pre-registered, development data)", "",
         f"Generated by `python -m scripts.tr_precheck` ({out.name}); rules fixed in docs/plan_tta.md sec. 4 and the "
         f"script docstring before the run. Predictions D_p (step 0) and D_c (step 100) from {CAPA_RUN}; u = frozen "
         "UniDepthV2-L Sigma. Test never read.", "",
         f"**Outcome: {ver['outcome']}** (R0 {ver['R0']}, R1 {ver['R1']}, R2 {ver['R2']}).", "",
         "| dataset | Spearman(u, prior error) | C reduction vs CAPA, k=2 (95 % CI) | clean retention, k=2 | C vs constant width (95 % CI) |",
         "|---|---:|---:|---:|---:|"]
    for ds, x in res.items():
        L.append(f"| {ds} | {f(x['spearman_u_err'])} | {ci(x['c_reduction'][PRIMARY_K])} | {f(x['clean_retention'][PRIMARY_K])} | "
                 f"{ci(x['c_vs_const'])} |")
    L += ["", "AbsRel (main scope C / clean; full image C / clean):", "",
          "| dataset | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    for ds, x in res.items():
        for reg in ("C", "clean", "C_full", "clean_full"):
            L.append(f"| {ds} {reg} | " + " | ".join(f(x["absrel"][v][reg], 4) for v in VARIANTS) + " |")
    L += ["", "Descriptive: k = 1 / 3 C reduction and clean retention; AUROC of d, u, z (main scope, wrong vs correct readings).", "",
          "| dataset | C red. k=1 | ret. k=1 | C red. k=3 | ret. k=3 | AUROC d | AUROC u | AUROC z | invalid px (capa / tr2) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, x in res.items():
        L.append(f"| {ds} | {f(x['c_reduction'][1.0][0])} | {f(x['clean_retention'][1.0])} | {f(x['c_reduction'][3.0][0])} | "
                 f"{f(x['clean_retention'][3.0])} | {f(x['auroc']['d'])} | {f(x['auroc']['u'])} | {f(x['auroc']['z'])} | "
                 f"{x['invalid']['capa']} / {x['invalid']['tr2']} |")
    L += ["", "Engineering: recomputed aligned prior vs CAPA's recorded step 0, worst per-image median relative difference: "
          + ", ".join(f"{ds} {x['align_check_median_rel_diff_max']:.2e}" for ds, x in res.items()) + ".",
          "Caveat: D_tr is a post-hoc emulation on CAPA's recorded outputs, not a trust-region adaptation; it tests "
          "whether the prior's uncertainty marks where adaptation should and should not move the prediction.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-endpoint", default=None)
    a = ap.parse_args()
    import torch
    capa_run = ROOT / CAPA_RUN
    rec = json.loads((capa_run / "receipt.json").read_text(encoding="utf-8"))
    if rec.get("stage") != "complete" or sha256(capa_run / "tta_precheck.json") != rec["result_sha256"]:
        raise SystemExit(f"{CAPA_RUN} is not a complete, untouched CAPA pre-check run")
    config = dict(tau=TAU, band=BAND, seed=SEED, draws=DRAWS, ks=list(KS), primary_k=PRIMARY_K, capa_run=CAPA_RUN,
                  capa_run_result_sha256=rec["result_sha256"], limit=a.limit, spearman_per_image=SPEARMAN_PER_IMAGE)
    out = ROOT / "results" / f"tr_precheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, config, [sys.executable, "-m", "scripts.tr_precheck", *sys.argv[1:]], SEED)
    try:
        info = T.fetch_capa(a.hf_endpoint)
        if info["weights_revision"] != rec["capa_setup"]["capa"]["weights_revision"]:
            raise RuntimeError("UniDepthV2-L weights differ from the CAPA run")
        device = torch.device(a.device)
        proto, _ = T.load_protocol(device, info["weights_dir"])
        from capa.utils.alignment import align_depth_to_condition
        proto.model.reset()
        net = proto.model.model.eval()
        rng = np.random.default_rng(SEED)
        rows = []
        todo = [(ds, r) for ds in T.DATASETS for r in T.select(ds, T.DATASETS[ds]["per_group"], a.limit)]
        for i, (ds, r) in enumerate(todo):
            name = f"{ds}__{Path(r['path']).stem}"
            path = ROOT / T.DATASETS[ds]["processed"] / r["path"]
            if sha256(path) != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            s = load_sample(path)
            if s.split == "test":
                raise ValueError("test sample reached the pre-check")
            rgb, cond, mask = T.tensors(s)
            t1 = time.time()
            with torch.no_grad():
                o = net.infer(rgb.to(device) * 255.0)
                depth = o["depth"][:, 0]
                scale, shift = align_depth_to_condition(depth, cond.to(device), mask.to(device),
                                                        n_points=proto.align_n_points, seed=proto.align_seed)
                Dp = (depth * scale.view(-1, 1, 1) + shift.view(-1, 1, 1))[0].float().cpu().numpy()
                u = o["confidence"][0, 0].float().cpu().numpy()
            seconds = time.time() - t1
            z = np.load(capa_run / "predictions" / f"{name}.npz")
            D0, Dc = z["step_0"].astype(np.float32), z["step_100"].astype(np.float32)
            align_check = float(np.median(np.abs(Dp - D0) / np.maximum(np.abs(D0), 1e-6)))
            _, scope = focus_scope(s)
            stats = image_stats(s.prompt, s.prompt_valid, s.gt, s.gt_valid, scope, variants(Dp, Dc, u), u, rng)
            np.savez_compressed(out / f"u__{name}.npz", u=u.astype(np.float16))
            rows.append(dict(dataset=ds, scene=s.scene, name=name, seconds=seconds, align_check=align_check, stats=stats))
            print(f"[{i + 1}/{len(todo)}] {name}: {seconds:.1f} s, align check {align_check:.1e}", flush=True)
        if max(x["align_check"] for x in rows) > 0.01:
            raise RuntimeError("recomputed aligned prior differs from CAPA's step 0 by > 1 % (median) on some image")
        res = summarise(rows, np.random.default_rng(SEED))
        ver = verdict(res) if a.limit is None else dict(outcome="engineering check (no verdict)", R0=None, R1=None, R2=None)
        write_json(out / "tr_precheck.json", dict(results={ds: {k: v for k, v in x.items()} for ds, x in res.items()}, verdict=ver))
        lines = report(out, res, ver, a)
        doc = ROOT / "docs" / ("tr_precheck.md" if a.limit is None else "tr_precheck_engineering.md")
        doc.write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="complete" if a.limit is None else "engineering_check_complete", verdict=ver,
                       result_sha256=sha256(out / "tr_precheck.json"), doc=doc.relative_to(ROOT).as_posix(),
                       model_version=f"UniDepthV2-L {info['weights_revision']}")
        print("\n".join(lines))
        print("TR PRECHECK DONE" if a.limit is None else "TR ENGINEERING CHECK DONE", flush=True)
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
