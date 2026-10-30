"""Layer-2 check: are the wrong readings on HAMMER objects the second layer? (docs/plan_layered.md sec. 2)

The layered reframing says a depth sensor does not misread a transparent object; it measures a different surface, the
one behind it. HAMMER pairs every frame with a "naked" frame (objects removed, same robot pose), so the second layer is
known: `behind`. This script asks what share of the wrong readings on object pixels land on that layer. CPU only;
development scenes (HAMMER train + val, 2-11) only; the test scenes are never read.

  python -m scripts.layer2_check                      # all 1143 development samples
  python -m scripts.layer2_check --per-group 3        # 3 frames per scene x sensor (quick look)
  python -m scripts.layer2_check --resume results/layer2_check_<time> --max-seconds 150

Pixel populations (tau = 0.05, the project's reading tolerance):
  scope      object pixels: valid GT, behind > 0, |GT - behind| > 1 mm (tp.eval.run_eval.focus_scope);
             "interior" = scope eroded by 5 px (no object boundary), a secondary population.
  separable  behind * (1 - tau) > GT * (1 + tau): the two layers' tau-bands do not overlap, so a reading can be
             assigned to at most one layer. Pixels where the layers are too close (thin objects, objects lying on the
             surface) are counted but not classified.
  C          valid reading with |P - GT| / GT > tau (tp.eval.regions).
Every reading in C on a separable pixel falls in exactly one class:
  front      P < GT (1 - tau)                     nearer than the object
  between    GT (1 + tau) < P < behind (1 - tau)  between the two layers (mixed / multipath / interpolated)
  layer2     |P - behind| <= tau * behind         on the second layer
  beyond     P > behind (1 + tau)                 farther than the second layer
Placebo: a made-up third layer as far behind layer 2 as layer 2 is behind the object, B' = 2 behind - GT; the share of
C readings within tau of B'. If readings spread widely, layer 2 and the placebo catch similar shares.
Also t = (P - GT) / (behind - GT) for C readings on separable pixels (0 = object, 1 = second layer), as a histogram.

PRE-REGISTERED READING RULE (fixed before the full run; a single frame per sensor, scene5 traj1_1 000000, was looked at
while timing the loader, see docs/plan_layered.md):
  primary endpoint  layer2 share = layer2 / (C on separable pixels), pooled over the three sensors, scope "objects";
                    95 % CI by scene bootstrap (10 development scenes resampled with replacement, 10000 draws).
  supported         CI low >= 0.50 and layer2 share >= 3 x placebo share.
  leaning           point >= 0.50 but CI low < 0.50 (and >= 3 x placebo).
  sensor-specific   pooled not supported/leaning, but at least one sensor has CI low >= 0.50 and >= 3 x placebo:
                    the direction holds only for that sensing principle.
  not supported     otherwise.
  Descriptive, not part of the rule: per-sensor shares, the interior population, the share of all readings (not only
  C) on separable object pixels that land on layer 2, and the "between" share. A large "between" share points to
  mixtures of the two layers rather than a clean second layer.
Part N (descriptive, no rule): NYU development mirrors (Mirror3D, material 2) have no second-layer GT; we only report
which side of the mirror plane the wrong readings lie on (a virtual image lies behind it).

Writes results/layer2_check_<time>/ (config, receipt, command, diff, per_sample/*.json, layer2_check.json) and
docs/layer2_check.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.provenance import ROOT, sha256, start_run, write_json

TAU, ERODE, SEED, DRAWS = 0.05, 5, 20261029, 10000
T_EDGES = np.round(np.arange(-1.0, 3.0001, 0.1), 6)
CLASSES = ("front", "between", "layer2", "beyond")
SCOPES = ("objects", "interior")
HAMMER = "data/processed/hammer_dev_v1"
NYU = "data/processed/mirror3d_dev_v1"


def erode(mask, r):
    """Square erosion with radius r (a pixel survives if its (2r+1)^2 window lies inside the mask)."""
    if r <= 0:
        return mask.copy()
    p = np.pad(mask.astype(np.int32), r + 1)
    s = p.cumsum(0).cumsum(1)
    h, w = mask.shape
    k = 2 * r + 1
    win = s[k:k + h, k:k + w] - s[0:h, k:k + w] - s[k:k + h, 0:w] + s[0:h, 0:w]
    return mask & (win == k * k)


def classify(P, pv, G, gv, B, scope, tau=TAU):
    """Integer counts for one image and one pixel population (see module docstring). Pure numpy, no I/O."""
    scope = scope & gv & (B > 0)
    v = scope & pv & (P > 0)
    Gs = np.where(G > 0, G, 1.0)
    rel = np.abs(P - G) / Gs
    C = v & (rel > tau)
    sep = scope & (B * (1 - tau) > G * (1 + tau))
    inverted = scope & (B * (1 + tau) < G * (1 - tau))
    c = C & sep
    out = dict(scope=int(scope.sum()), hole=int((scope & ~(pv & (P > 0))).sum()), readings=int(v.sum()),
               correct=int((v & ~C).sum()), C=int(C.sum()), separable=int(sep.sum()),
               C_separable=int(c.sum()), C_inseparable=int((C & ~sep & ~inverted).sum()), C_inverted=int((C & inverted).sum()),
               readings_separable=int((v & sep).sum()), correct_separable=int((v & sep & ~C).sum()))
    # classes built from the same rel / rel2 comparisons as C, so they partition C exactly (no float boundary leaks)
    l2 = c & (np.abs(P - B) / np.where(B > 0, B, 1.0) <= tau)
    cls = dict(front=c & (P < G), between=c & (P > G) & (P < B) & ~l2, layer2=l2, beyond=c & (P >= B) & ~l2)
    for k, m in cls.items():
        out[k] = int(m.sum())
    if sum(out[k] for k in CLASSES) != out["C_separable"]:
        raise AssertionError("classes must partition C on separable pixels")
    Bp = 2 * B - G
    out["placebo"] = int((c & (np.abs(P - Bp) <= tau * Bp)).sum())
    t = (P[c] - G[c]) / (B[c] - G[c])
    hist = np.histogram(np.clip(t, T_EDGES[0], T_EDGES[-1]), bins=T_EDGES)[0]
    out["t_hist"] = hist.astype(int).tolist()
    return out


def hammer_records(per_group):
    rows = json.loads((ROOT / HAMMER / "manifest.json").read_text(encoding="utf-8"))["records"]
    rows = [r for r in rows if r["split"] in {"train", "val"}]   # test scenes locked
    return per_group_subsample(rows, per_group)


def nyu_counts(path, tau=TAU):
    s = load_sample(path)
    m = (s.material == 2) & s.gt_valid
    v = m & s.prompt_valid & (s.prompt > 0)
    rel = np.abs(s.prompt - s.gt) / np.where(s.gt > 0, s.gt, 1.0)
    C = v & (rel > tau)
    return dict(scene=s.scene, mirror=int(m.sum()), hole=int((m & ~v).sum()), correct=int((v & ~C).sum()), C=int(C.sum()),
                C_behind_plane=int((C & (s.prompt > s.gt)).sum()), C_front_of_plane=int((C & (s.prompt < s.gt)).sum()))


def ratio(a, b):
    return None if b == 0 else a / b


def bootstrap(rows, rng):
    """Scene bootstrap of sum(num) / sum(den); rows: list of (scene, num, den). Returns point, lo, hi."""
    scenes = sorted({r[0] for r in rows})
    n = np.array([sum(r[1] for r in rows if r[0] == s) for s in scenes], float)
    d = np.array([sum(r[2] for r in rows if r[0] == s) for s in scenes], float)
    if d.sum() == 0:
        return None, None, None
    idx = rng.integers(0, len(scenes), size=(DRAWS, len(scenes)))
    dn, dd = n[idx].sum(1), d[idx].sum(1)
    vals = dn[dd > 0] / dd[dd > 0]
    return float(n.sum() / d.sum()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def summarise(per_sample, rng):
    res = {}
    for scope in SCOPES:
        groups = {"pooled": per_sample, **{s: [x for x in per_sample if x["sensor"] == s] for s in ("d435", "l515", "helios")}}
        res[scope] = {}
        for g, xs in groups.items():
            tot = {k: sum(x[scope][k] for x in xs) for k in xs[0][scope] if k != "t_hist"} if xs else {}
            if not xs:
                continue
            tot["t_hist"] = np.sum([x[scope]["t_hist"] for x in xs], 0).astype(int).tolist()
            rows = lambda k, d="C_separable": [(x["scene"], x[scope][k], x[scope][d]) for x in xs]
            res[scope][g] = dict(
                samples=len(xs), totals=tot,
                layer2=bootstrap(rows("layer2"), rng), placebo=ratio(tot["placebo"], tot["C_separable"]),
                shares={k: ratio(tot[k], tot["C_separable"]) for k in CLASSES},
                hole_share=ratio(tot["hole"], tot["scope"]), correct_share_of_scope=ratio(tot["correct"], tot["scope"]),
                C_share_of_scope=ratio(tot["C"], tot["scope"]), separable_share_of_C=ratio(tot["C_separable"], tot["C"]),
                layer2_share_of_separable_readings=ratio(tot["layer2"], tot["readings_separable"]),
                correct_share_of_separable_readings=ratio(tot["correct_separable"], tot["readings_separable"]))
    return res


def verdict(objects):
    def ok(x):
        p, lo, _ = x["layer2"]
        return p is not None and x["placebo"] is not None and p >= 3 * x["placebo"], p, lo
    good, p, lo = ok(objects["pooled"])
    if good and lo >= 0.5:
        return "supported"
    if good and p >= 0.5:
        return "leaning"
    hits = [s for s in ("d435", "l515", "helios") if s in objects and ok(objects[s])[0] and ok(objects[s])[2] >= 0.5]
    return f"sensor-specific ({', '.join(hits)})" if hits else "not supported"


def f(x, d=3):
    return "NA" if x is None else f"{x:.{d}f}"


def report(out, res, nyu, args):
    o = res["objects"]
    lines = ["# Layer-2 check on HAMMER (pre-registered, development data)", "",
             f"Generated by `python -m scripts.layer2_check` ({out.name}); rule fixed in docs/plan_layered.md sec. 2 and the "
             f"script docstring before the run. tau = {TAU}; {o['pooled']['samples']} samples "
             f"({'all' if args.per_group is None else str(args.per_group) + ' per scene x sensor'}); test scenes never read.", "",
             f"**Verdict (pre-registered rule): {res['verdict']}.**", "",
             "## Primary: wrong readings on separable object pixels", "",
             "Share of C readings (|P - GT| / GT > tau) on pixels whose two layers are tau-separable; 95 % CI by scene bootstrap.", "",
             "| sensor | samples | C on separable px | front | between | **layer 2** (95 % CI) | beyond | placebo layer |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in SCOPES:
        if scope == "interior":
            lines += ["", "Interior (object eroded by 5 px, no boundary pixels):", "",
                      "| sensor | samples | C on separable px | front | between | **layer 2** (95 % CI) | beyond | placebo layer |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g, x in res[scope].items():
            p, lo, hi = x["layer2"]
            sh = x["shares"]
            lines.append(f"| {g} | {x['samples']} | {x['totals']['C_separable']} | {f(sh['front'])} | {f(sh['between'])} | "
                         f"**{f(p)}** ({f(lo)}-{f(hi)}) | {f(sh['beyond'])} | {f(x['placebo'])} |")
    lines += ["", "## Context: all object pixels", "",
              "| sensor | hole | correct | C | separable share of C | layer 2 among all readings on separable px | correct among them |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for g, x in o.items():
        lines.append(f"| {g} | {f(x['hole_share'])} | {f(x['correct_share_of_scope'])} | {f(x['C_share_of_scope'])} | "
                     f"{f(x['separable_share_of_C'])} | {f(x['layer2_share_of_separable_readings'])} | "
                     f"{f(x['correct_share_of_separable_readings'])} |")
    lines += ["", "## Where the wrong readings sit between the layers", "",
              "t = (P - GT) / (behind - GT): 0 = object surface, 1 = second layer (C readings on separable pixels, "
              "objects scope; bins of 0.1, ends clipped to [-1, 3]). Share per bin:", "",
              "| t bin | " + " | ".join(o) + " |", "|---|" + "---:|" * len(o)]
    for i in range(len(T_EDGES) - 1):
        lo_, hi_ = T_EDGES[i], T_EDGES[i + 1]
        if not (-0.5 <= lo_ < 1.6):
            continue
        cells = []
        for g, x in o.items():
            h = np.array(x["totals"]["t_hist"], float)
            cells.append(f(h[i] / h.sum() if h.sum() else None))
        lines.append(f"| [{lo_:.1f}, {hi_:.1f}) | " + " | ".join(cells) + " |")
    if nyu:
        tot = {k: sum(r[k] for r in nyu) for k in nyu[0] if k != "scene"}
        lines += ["", "## N. NYU development mirrors (descriptive; no second-layer GT)", "",
                  f"{len(nyu)} images, {tot['mirror']} mirror pixels with GT: hole {f(ratio(tot['hole'], tot['mirror']))}, "
                  f"correct {f(ratio(tot['correct'], tot['mirror']))}, C {f(ratio(tot['C'], tot['mirror']))}. "
                  f"Of the C readings, {f(ratio(tot['C_behind_plane'], tot['C']))} lie behind the mirror plane (where a "
                  f"virtual image lies) and {f(ratio(tot['C_front_of_plane'], tot['C']))} in front of it.", ""]
    lines += ["", "Caveats: HAMMER has no material labels, so 'objects' mixes transparent, metallic and diffuse objects; "
              "wrong readings on diffuse objects are mostly boundary errors, which the interior rows remove. The naked "
              "frame is paired by pose (<= 1 mm, <= 0.1 deg), so behind carries the pairing's registration error.", ""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=None, help="frames per scene x sensor (default: all)")
    ap.add_argument("--resume", type=Path, default=None, help="continue a run folder (per-sample results are kept)")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after this long; resume later")
    ap.add_argument("--no-nyu", action="store_true")
    a = ap.parse_args()
    config = dict(tau=TAU, erode_px=ERODE, seed=SEED, bootstrap_draws=DRAWS, per_group=a.per_group, hammer=HAMMER,
                  nyu=None if a.no_nyu else NYU, t_edges=T_EDGES.tolist(), splits=["train", "val"])
    command = [sys.executable, "-m", "scripts.layer2_check", *sys.argv[1:]]
    if a.resume is None:
        out = ROOT / "results" / f"layer2_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, config, command, SEED)
    else:
        out = a.resume if a.resume.is_absolute() else ROOT / a.resume
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        if receipt["config"] != json.loads(json.dumps(config)):
            raise ValueError("resume requires the identical configuration")
        if sha256(ROOT / "scripts" / "layer2_check.py") != receipt["source_sha256"]["scripts/layer2_check.py"]:
            raise ValueError("scripts/layer2_check.py changed since the run started; start a new run")
        receipt.setdefault("resumed", []).append(command)
    t0 = time.time()
    try:
        recs = hammer_records(a.per_group)
        done_dir = out / "per_sample"
        done_dir.mkdir(exist_ok=True)
        for r in recs:
            dst = done_dir / (Path(r["path"]).stem + ".json")
            if dst.is_file():
                continue
            if a.max_seconds is not None and time.time() - t0 > a.max_seconds:
                receipt.update(stage="partial_time_budget")
                print(f"time budget reached; {sum(1 for _ in done_dir.glob('*.json'))} / {len(recs)} done. "
                      f"Resume with: python -m scripts.layer2_check --resume {out.relative_to(ROOT).as_posix()}"
                      + ("" if a.per_group is None else f" --per-group {a.per_group}") + (" --no-nyu" if a.no_nyu else ""))
                return
            path = ROOT / HAMMER / r["path"]
            if sha256(path) != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {r['path']}")
            s = load_sample(path)
            if s.split == "test":
                raise ValueError("test sample reached the layer-2 check")
            if s.behind is None:
                row = dict(sensor=s.sensor, scene=s.scene, frame=s.frame, behind=None)
            else:
                full = s.gt_valid & (s.behind > 0) & (np.abs(s.gt - s.behind) > 1e-3)
                row = dict(sensor=s.sensor, scene=s.scene, frame=s.frame, behind=True,
                           objects=classify(s.prompt, s.prompt_valid, s.gt, s.gt_valid, s.behind, full),
                           interior=classify(s.prompt, s.prompt_valid, s.gt, s.gt_valid, s.behind, erode(full, ERODE)))
            write_json(dst, row)
        per_sample = [json.loads((done_dir / (Path(r["path"]).stem + ".json")).read_text(encoding="utf-8")) for r in recs]
        unpaired = [x["frame"] for x in per_sample if x["behind"] is None]
        per_sample = [x for x in per_sample if x["behind"] is not None]
        nyu = []
        if not a.no_nyu:
            rows = json.loads((ROOT / NYU / "manifest.json").read_text(encoding="utf-8"))["records"]
            for r in rows:
                if r["split"] in {"train", "val"}:
                    nyu.append(nyu_counts(ROOT / NYU / r["path"]))
        rng = np.random.default_rng(SEED)
        res = summarise(per_sample, rng)
        res["verdict"] = verdict(res["objects"])
        res["unpaired_samples"] = unpaired
        res["nyu"] = nyu
        write_json(out / "layer2_check.json", res)
        lines = report(out, res, nyu, a)
        doc = ROOT / "docs" / ("layer2_check.md" if a.per_group is None else "layer2_check_subset.md")
        doc.write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="complete", verdict=res["verdict"], result_sha256=sha256(out / "layer2_check.json"),
                       doc=doc.relative_to(ROOT).as_posix())
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
