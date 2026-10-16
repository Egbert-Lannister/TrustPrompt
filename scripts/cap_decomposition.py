"""R16 / T v2 diagnostic: where does the far-only bound cap gain go when the sign comes from T v2?

  python -m scripts.cap_decomposition            (numpy only; no torch / scipy needed)

On the 192x256 trust grid (centre-grid nearest samples of the full-resolution arrays, the same sampling as
the T training data), over contaminated main-scope pixels (|P - GT| / GT > 0.05; NYU mirrors / HAMMER objects):
  base = PromptDA-L prediction from the GT-mask prompt (oracle_prior cache), no cap
  cap  = min(pred, P) on pixels whose sign is 'far'
Signs: GT (oracle), T v2 argmax (p_far >= p_near), and T v2 confidence rules r = p_far / (p_far + p_near) >= q.
Decomposition of the argmax rule (contributions to the pooled mean AbsRel, relative to no cap):
  gain = far pixels capped (error removed), harm = near pixels capped (error change), lost = far pixels not capped.
The decision rule q* = h / (g + h) uses the mean per-pixel gain g (far, pred > P) and harm h (near, pred > P)
measured on the train split minus the holdout scenes only, then is applied unchanged to holdout and val.
Grid-level numbers approximate the full-resolution evaluation; they explain, they do not replace, bound_results.md.
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from tp.geom.camera import resize_nearest

ROOT = Path(__file__).resolve().parents[1]
GRID = (192, 256)
TAGS = {"mirror3d_nyu": ("nyu", "promptda_vitl_oracle_prior_nyu_dev_v1"),
        "hammer": ("hammer", "promptda_vitl_oracle_prior_hammer_dev_v2")}
MODELS = ("real_sim_nofeat", "real_sim")
QS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
POPS = ("train fit", "train holdout", "val")


def holdout(ds, scene):  # same rule as tp.trust.data.scene_inner_holdout (that module needs scipy)
    return scene in ("scene10", "10") if ds == "hammer" else zlib.crc32(scene.encode("utf-8")) % 5 == 0


def items(ds, tag):
    out = []
    for s in sorted((ROOT / "cache" / f"trust_data_v1_{tag}").glob("shard_*")):
        if not s.is_dir() or "_failed" in s.name:
            continue
        if json.loads((s / "receipt.json").read_text(encoding="utf-8")).get("stage") != "trust_data_verified":
            raise ValueError(f"{s}: unexpected stage")
        for e in json.loads((s / "index.json").read_text(encoding="utf-8")):
            if e.get("variant") == "real":
                out.append(dict(e, file=str(s / e["path"])))
    return out


PART = ROOT / "cache" / "cap_decomposition_items.jsonl"   # per-image sums; resumable (I/O-bound on the bridge)


def per_image(ds, pop, a, pred, tpm):
    p = a["prompt"].astype(np.float64); gt = a["gt"].astype(np.float64)
    ok = a["valid"] & a["gt_valid"] & a["focus"] & (p > 0) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    far = ok & (p > 1.05 * gt); near = ok & (p < 0.95 * gt); con = far | near
    if not con.any():
        return None
    g = np.where(ok, gt, 1.0)
    e0 = np.abs(pred - g) / g; e1 = np.abs(np.minimum(pred, p) - g) / g; d = e1 - e0  # error change if capped
    fo, no = far & (pred > p), near & (pred > p)
    b = dict(n=int(con.sum()), base=float(e0[con].sum()), gt=float(e0[con].sum() + d[far].sum()),
             g=[float(-d[fo].sum()), int(fo.sum())], h=[float(d[no].sum()), int(no.sum())])
    dg = a["dg"].astype(np.float64); okd = far & np.isfinite(dg) & (dg > 0)
    # fill-side headroom: the k0 fill puts D_g on flagged pixels; would capping the fill at P help it?
    b["fill"] = [float((np.abs(dg - g) / g)[okd].sum()), float((np.abs(np.minimum(dg, p) - g) / g)[okd].sum()),
                 int(okd.sum()), int((okd & (dg > p)).sum())]
    for m, r in tpm.items():
        sel = con & (r >= 0.5)
        b[f"{m}:argmax"] = float(e0[con].sum() + d[sel].sum())
        b[f"{m}:gain"] = float(-d[far & sel].sum()); b[f"{m}:harm"] = float(d[near & sel].sum())
        b[f"{m}:lost"] = float(-d[far & ~sel].sum())
        for x in QS:
            b[f"{m}:q{x}"] = float(e0[con].sum() + d[con & (r >= x)].sum())
    return b


def collect(budget):
    done = set()
    if PART.exists():
        done = {json.loads(l)["key"] for l in PART.read_text(encoding="utf-8").splitlines() if l.strip()}
    t0 = time.time(); left = 0
    with PART.open("a", encoding="utf-8") as out:
        for ds, (tag, pcache) in TAGS.items():
            tc = {m: ROOT / "cache" / f"trust_t_v2_{m}_{tag}_dev_v1" for m in MODELS}
            tidx = {m: {e["key"]: e for e in json.loads((tc[m] / "index.json").read_text(encoding="utf-8"))} for m in MODELS}
            for e in items(ds, tag):
                key = f"{e['sensor']}__{e['scene']}__{e['frame']}"
                if f"{ds}/{key}" in done:
                    continue
                if time.time() - t0 > budget:
                    left += 1; continue
                pop = "val" if e["split"] == "val" else ("train holdout" if holdout(ds, e["scene"]) else "train fit")
                pf = ROOT / "cache" / pcache / "predictions" / f"{key}.npz"
                rec = dict(key=f"{ds}/{key}", ds=ds, pop=pop, b=None)
                if pf.exists() and all(key in tidx[m] for m in MODELS):
                    z = np.load(e["file"]); a = {k: z[k] for k in z.files}
                    q = np.load(pf)
                    full = (q["depth_q"].astype(np.float64) * float(q["scale_m"])) if "depth_q" in q.files else q["depth"].astype(np.float64)
                    pred = resize_nearest(full, GRID)  # depth_q == 0 marks an invalid prediction -> excluded
                    tpm = {}
                    for m in MODELS:
                        t = np.load(tc[m] / tidx[m][key]["path"]); a_f = t["p_far"].astype(np.float64); a_n = t["p_near"].astype(np.float64)
                        tpm[m] = a_f / np.maximum(a_f + a_n, 1e-12)
                    rec["b"] = per_image(ds, pop, a, pred, tpm)
                out.write(json.dumps(rec) + "\n"); out.flush()
    return left


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    left = collect(budget)
    if left:
        print(f"PARTIAL: {left} images left; run again"); return
    acc = {}
    for l in PART.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if not r["b"]:
            continue
        b = acc.setdefault((r["ds"], r["pop"]), None)
        if b is None:
            acc[(r["ds"], r["pop"])] = dict(r["b"], imgs=1)
        else:
            for k, v in r["b"].items():
                if isinstance(v, list):
                    b[k] = [x + y for x, y in zip(b[k], v)]
                else:
                    b[k] += v
            b["imgs"] += 1
    qstar = {}
    for ds in TAGS:
        b = acc[(ds, "train fit")]; g = b["g"][0] / max(b["g"][1], 1); h = b["h"][0] / max(b["h"][1], 1)
        qstar[ds] = dict(g=g, h=h, q=h / (g + h) if g + h > 0 else 0.5)
    lines = ["# Far-only cap with T v2 signs: decomposition (192x256 grid)", "",
             "Generated by `python -m scripts.cap_decomposition` (diagnostic; grid-level approximation of the full-resolution "
             "evaluation in bound_results.md). Pooled mean AbsRel over contaminated main-scope pixels; prediction = PromptDA-L "
             "with the GT-mask prompt (oracle mask), so only the sign source varies. 'train fit' = train split minus holdout "
             "scenes (in-sample for T); 'train holdout' and 'val' are out of fit.", "",
             "Decision rule from train fit only: cap where r = p_far/(p_far+p_near) >= q* = h/(g+h); g = mean error removed per "
             "capped far pixel, h = mean error added per capped near pixel (both where pred > P).", ""]
    for ds in TAGS:
        s = qstar[ds]
        lines.append(f"- {ds}: g = {s['g']:.4f}, h = {s['h']:.4f}, h/g = {s['h'] / max(s['g'], 1e-12):.1f}, q* = {s['q']:.3f}")
    lines.append("")
    payload = dict(qstar=qstar, rows={})
    for ds in TAGS:
        for pop in POPS:
            b = acc.get((ds, pop))
            if not b:
                continue
            n = b["n"]; base = b["base"] / n; gtc = b["gt"] / n
            lines += [f"## {ds}, {pop}: {b['imgs']} images, {n} contaminated pixels", "",
                      f"No cap {base:.4f}; GT-sign cap {gtc:.4f} (gain {base - gtc:.4f}). Fill side, far pixels: D_g AbsRel "
                      f"{b['fill'][0] / max(b['fill'][2], 1):.4f}, min(D_g, P) {b['fill'][1] / max(b['fill'][2], 1):.4f}, "
                      f"D_g > P on {b['fill'][3] / max(b['fill'][2], 1):.1%} of them.", "",
                      "| T v2 model | argmax cap | recovered | gain | harm | lost | q* rule | recovered (q*) | " +
                      " | ".join(f"q={x}" for x in QS) + " |", "|---|" + "---:|" * (7 + len(QS))]
            for m in MODELS:
                am = b[f"{m}:argmax"] / n
                qs_ = min(QS, key=lambda x: abs(x - qstar[ds]["q"]))
                # exact q* evaluation needs another pass; report nearest grid value and label it
                qv = b[f"{m}:q{qs_}"] / n
                rec = lambda v: "NA" if base - gtc <= 0 else f"{(base - v) / (base - gtc):.0%}"
                lines.append(f"| {m} | {am:.4f} | {rec(am)} | {b[f'{m}:gain'] / n:.4f} | {b[f'{m}:harm'] / n:.4f} | "
                             f"{b[f'{m}:lost'] / n:.4f} | q={qs_} {qv:.4f} | {rec(qv)} | " +
                             " | ".join(f"{b[f'{m}:q{x}'] / n:.4f}" for x in QS) + " |")
                payload["rows"][f"{ds}/{pop}/{m}"] = dict(n=n, base=base, gt_sign=gtc, argmax=am,
                                                          gain=b[f"{m}:gain"] / n, harm=b[f"{m}:harm"] / n, lost=b[f"{m}:lost"] / n,
                                                          q={str(x): b[f"{m}:q{x}"] / n for x in QS})
            lines.append("")
    (ROOT / "docs" / "cap_decomposition.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "docs" / "cap_decomposition.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
