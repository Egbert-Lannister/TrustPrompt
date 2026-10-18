"""Marigold-DC guidance track: tables and the pre-registered gate -> docs/marigold_guidance_{nyu,hammer}.md

  python -m scripts.summarize_guidance --stage nyu      # after M1: table, Holm, gate -> results/marigold_guidance_gate.json
  python -m scripts.summarize_guidance --stage hammer   # after M2: table, Holm, confirmation

Gate, ceiling and confirmation rules: configs/baselines/marigold_guidance_v1.yaml (written before any prediction).
Numbers come from the evaluation runs (scripts/summarize_baselines.row) and the latest Holm family run of the stage
(scripts/holm_family). Development diagnostics; test never read.
"""
import argparse
import json

import yaml

from scripts.summarize_baselines import fmt, row
from tp.provenance import ROOT, sha256, write_json

CONFIG = ROOT / "configs" / "baselines" / "marigold_guidance_v1.yaml"
GUIDED = ("tonesided", "tsoft")
DOC = {"nyu": ROOT / "docs" / "marigold_guidance_nyu.md", "hammer": ROOT / "docs" / "marigold_guidance_hammer.md"}
GATE_FILE = ROOT / "results" / "marigold_guidance_gate.json"


def latest_family(family):
    runs = sorted(p for p in (ROOT / "results").glob(f"holm_{family}_*") if (p / "family_results.json").is_file())
    if not runs:
        raise FileNotFoundError(f"no completed Holm run for {family}; run scripts.holm_family first")
    return runs[-1], json.loads((runs[-1] / "family_results.json").read_text(encoding="utf-8"))


def gate_verdict(c_val, c_val_tmask, c_hold, c_hold_tmask, clean_clearly_worse, gate):
    """M1 gate for one guided variant (configs/baselines/marigold_guidance_v1.yaml). Missing numbers fail it."""
    v = dict(val_gain=None if None in (c_val, c_val_tmask) else 1 - c_val / c_val_tmask,
             holdout_not_worse=None if None in (c_hold, c_hold_tmask) else c_hold <= c_hold_tmask,
             clean_clearly_worse=clean_clearly_worse)
    v["passes"] = bool(v["val_gain"] is not None and v["val_gain"] >= gate["min_relative_gain_val"]
                       and (v["holdout_not_worse"] is True or not gate["holdout_not_worse"])
                       and (clean_clearly_worse is False or not gate["clean_not_clearly_worse"]))
    return v


def holm_lines(results):
    lines = ["| Comparison | reference | candidate | difference [95% CI] | p (Holm) | reject | clean clearly worse |",
             "|---|---:|---:|---|---:|---|---|"]
    for r in results.values():
        ci = (f"{fmt(r.get('mean_difference'))} [{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}]"
              if r.get("mean_difference") is not None else r.get("status", "NA"))
        lines.append(f"| {r['model']} | {fmt(r.get('reference_mean'))} | {fmt(r.get('candidate_mean'))} | {ci} | "
                     f"{fmt(r.get('p_holm'))} | {'yes' if r.get('reject_at_alpha') else 'no'} | "
                     f"{(r.get('clean_check') or {}).get('clearly_worse')} |")
    return lines


def table(ds, prefix, scope, runs, splits):
    lines = ["| Variant | population | images | C AbsRel | C full | clean | IR | DOR | invalid imgs |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    got = {}
    for v, run in runs.items():
        for split, hold, tag in splits:
            x = got[(v, tag)] = row(ds, prefix, scope, run, split, hold)
            cells = ([str(x["images"])] + [fmt(x[k]) for k in ("c", "c_full", "clean", "ir", "dor")]
                     + [str(x["images_with_invalid"])]) if x else ["NA", "missing run"] + [""] * 5
            lines.append(f"| {v} | {tag} | " + " | ".join(cells) + " |")
    return lines, got


def stage_nyu(cfg):
    runs = {v: f"bl_marigold_dc_{v}" for v in cfg["nyu"]["variants"]}
    lines, got = table("mirror3d_nyu", "mirror3d_dev", "mirror_only", runs,
                       [("val", False, "val"), ("train", True, "train_holdout")])
    run_dir, fam = latest_family("marigold_nyu_v1")
    g = cfg["gate"]
    c = lambda v, pop: (got.get((v, pop)) or {}).get("c")
    verdicts = {}
    for v in GUIDED:
        cv, ct, mv, mt = c(v, "val"), c("tmask", "val"), c(v, "train_holdout"), c("tmask", "train_holdout")
        clean = fam["primary"].get(f"{v}_vs_tmask/mirror3d_nyu/val", {}).get("clean_check", {}).get("clearly_worse")
        verdicts[v] = gate_verdict(cv, ct, mv, mt, clean, g)
    co, cm = c("oracle_onesided", "val"), c("oracle_mask", "val")
    ceiling = None if None in (co, cm) else 1 - co / cm
    passed = [v for v in GUIDED if verdicts[v]["passes"]]
    if passed:
        reading = f"{', '.join(passed)} pass the gate: run stage M2 (HAMMER) with them."
    elif ceiling is not None and ceiling >= cfg["ceiling"]["oracle_relative_gain"]:
        reading = ("no guided variant passes, but with GT flags one-sided beats masking by "
                   f"{100 * ceiling:.0f}%: the headroom exists, T's flags limit it; no M2.")
    else:
        reading = "no guided variant passes and the GT-flag ceiling is below the bar: one-sided guidance adds nothing here; no M2."
    write_json(GATE_FILE, dict(passed=passed, verdicts=verdicts, oracle_ceiling=ceiling, holm_run=run_dir.name,
                               config_sha256=sha256(CONFIG)))
    out = ["# Marigold-DC guidance track (development diagnostics)", "",
           "Generated by `python -m scripts.summarize_guidance`. Variants: scripts/cache_completion.py; objective: "
           "baselines/guidance.py; rules fixed beforehand in configs/baselines/marigold_guidance_v1.yaml. "
           "Main-scope C AbsRel (NYU mirror pixels); test never read.", "",
           "## Stage M1: NYU (30 images)", ""] + lines + ["", f"Holm, primary family (val; {run_dir.name}):", ""]
    out += holm_lines(fam["primary"]) + ["", "Secondary family (train-holdout scenes):", ""] + holm_lines(fam["secondary"])
    out += ["", "Gate (val relative gain >= {:.0%} vs T mask, holdout not worse, clean not clearly worse):".format(
        g["min_relative_gain_val"]), "", "| Variant | val gain vs T mask | holdout not worse | clean clearly worse | passes |",
            "|---|---:|---|---|---|"]
    for v, x in verdicts.items():
        gain = "NA" if x["val_gain"] is None else f"{100 * x['val_gain']:+.1f}%"
        out.append(f"| {v} | {gain} | {x['holdout_not_worse']} | {x['clean_clearly_worse']} | {x['passes']} |")
    out += ["", f"GT-flag ceiling, 1 - C(oracle_onesided) / C(oracle_mask) on val: "
            f"{'NA' if ceiling is None else f'{100 * ceiling:+.1f}%'}", "", f"**Reading: {reading}**", ""]
    DOC["nyu"].write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def stage_hammer(cfg):
    gate = json.loads(GATE_FILE.read_text(encoding="utf-8"))
    tags = {"raw": "_allpg3", "oracle_mask": "_allpg3"}
    variants = cfg["hammer"]["variants"] + gate["passed"]
    runs = {v: f"bl_marigold_dc_{v}{tags.get(v, '_cfpg3')}" for v in variants}
    lines, _ = table("hammer", "hammer_dev", "objects_only", runs, [("val", False, "val"), ("train", False, "train")])
    run_dir, fam = latest_family("marigold_cf_v1")
    confirmed = []
    for v in gate["passed"]:
        r = fam["crossfit"].get(f"{v}_vs_tmask/hammer/all_dev", {})
        if (r.get("p_holm") is not None and r["p_holm"] < cfg["confirm"]["alpha"] and r.get("mean_difference", 0) < 0
                and (r.get("clean_check") or {}).get("clearly_worse") is False):
            confirmed.append(v)
    out = ["# Marigold-DC guidance track, stage M2: HAMMER (development diagnostics)", "",
           f"Generated by `python -m scripts.summarize_guidance --stage hammer`. Guided variants that passed M1 "
           f"(docs/marigold_guidance_nyu.md): {', '.join(gate['passed'])}. 10 dev scenes, 3 frames per scene x sensor, "
           "cross-fitted T; main-scope (object pixels) C AbsRel; test never read.", ""]
    out += lines + ["", f"Holm, cross-fitted family (all dev scenes, scene bootstrap; {run_dir.name}):", ""]
    out += holm_lines(fam["crossfit"])
    out += ["", f"**Confirmed (Holm p < {cfg['confirm']['alpha']}, better, clean not clearly worse): "
            f"{', '.join(confirmed) if confirmed else 'none'}**", ""]
    DOC["hammer"].write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["nyu", "hammer"], required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    (stage_nyu if a.stage == "nyu" else stage_hammer)(cfg)


if __name__ == "__main__":
    main()
