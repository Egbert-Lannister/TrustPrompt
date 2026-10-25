"""Mainline P1 gate: does training on synthesised structured errors (F1) beat the same fine-tuning without them (F0)?
(CPU; Windows .venv; development diagnostics, test never read)

  python -m scripts.p1_gate [--spec configs/omnidc/p1_gate_v1.yaml]

Rule (G1-G3) and runs: the spec. A run is either a name (results/<prefix>_<name>_<split>_v1, as the R0 / R1
baselines; scripts/holm_family.load_rows) or {split: folder} (the evaluation folders written by
scripts/cache_omnidc_finetuned.py --evaluate for F0 / F1). Each folder needs per_image_<scope>.json and a receipt
with a passed evaluation stage. R0's caches are stored as uint16 (0.2 mm, saturating at 13.107 m) while F0 / F1 are
float32; the clipped pixels of the R0 caches used in G3 are counted and reported next to the gate.
Writes results/p1_gate_<time>/gate.json and docs/omnidc_p1_gate.md.
"""
import argparse
from datetime import datetime
import json
import sys

import yaml

from scripts.holm_family import POP_SPLIT, fmt, load_rows
from tp.eval.family import compare
from tp.provenance import ROOT, sha256, start_run, write_json


OK_STAGES = ("small_scale_validation_passed", "completed_with_invalid_predictions")


def load_dirs(dirs, splits, scope, dataset, population):
    """As holm_family.load_rows, for explicit evaluation folders {split: folder}."""
    rows, hashes, names = [], [], []
    for split in splits:
        if split not in dirs:
            return None, f"no {split} folder given"
        d = ROOT / dirs[split]
        f, rec = d / f"per_image_{scope}.json", d / "receipt.json"
        if not f.is_file() or not rec.is_file():
            return None, str(d)
        stage = json.loads(rec.read_text(encoding="utf-8")).get("stage")
        if stage not in OK_STAGES:
            return None, f"{d} (stage {stage})"
        part = json.loads(f.read_text(encoding="utf-8"))
        if population == "train_holdout":
            from tp.trust.data import scene_inner_holdout
            part = [r for r in part if scene_inner_holdout(dataset, r["scene"])]
        rows += part; hashes.append(sha256(f)); names.append(dirs[split])
    return dict(rows=rows, sha256=hashes if len(hashes) > 1 else hashes[0], run="+".join(names)), "+".join(names)


def load(spec, name, ds, pop):
    d, run = spec["datasets"][ds], spec["runs"][name][ds]
    if isinstance(run, dict):
        return load_dirs(run, POP_SPLIT[pop], d["scope"], ds, pop)
    return load_rows(d["prefix"], run, POP_SPLIT[pop], d["scope"], ds, pop)


def r0_clipping(spec):
    """Clipped pixels (> 13.107 m) in the uint16 R0 caches, from their images.json."""
    out = {}
    for ds, cache in spec.get("r0_caches", {}).items():
        f = ROOT / cache / "images.json"
        if not f.is_file():
            out[ds] = dict(status=f"missing {f}")
            continue
        rows = json.loads(f.read_text(encoding="utf-8"))
        out[ds] = dict(images=len(rows), images_with_clipping=sum(1 for r in rows if r.get("clipped_pixels", 0) > 0),
                       clipped_pixels=sum(int(r.get("clipped_pixels", 0)) for r in rows))
    return out


def comparison(spec, ds, ref, cand, region, population=None):
    pop = population or spec["datasets"][ds]["population"]
    rows = {}
    for name in (ref, cand):
        rows[name], path = load(spec, name, ds, pop)
        if rows[name] is None:
            return dict(status="NA_missing_run", missing=path, reference=ref, candidate=cand, dataset=ds, population=pop, region=region)
    try:
        r = compare(rows[ref]["rows"], rows[cand]["rows"], dataset=ds, region=region, metric="absrel",
                    n_resamples=int(spec["n_resamples"]), seed=int(spec["seed"]), block_length=int(spec["block_length"]))
    except ValueError as e:
        r = dict(status=f"NA_error: {e}")
    rel = (r["mean_difference"] / r["reference_mean"]) if r.get("mean_difference") is not None and r.get("reference_mean") else None
    r.update(reference=ref, candidate=cand, dataset=ds, population=pop, region=region, relative_change=rel,
             reference_run=rows[ref]["run"], candidate_run=rows[cand]["run"],
             reference_sha256=rows[ref]["sha256"], candidate_sha256=rows[cand]["sha256"])
    return r


def clearly_worse(r):
    if r.get("ci_low") is None or r.get("relative_change") is None:
        return None
    return bool(r["ci_low"] > 0 and r["relative_change"] > 0.2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="configs/omnidc/p1_gate_v1.yaml")
    a = ap.parse_args()
    spec_path = ROOT / a.spec
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    out = ROOT / "results" / f"p1_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(spec, spec_sha256=sha256(spec_path)), [sys.executable, "-m", "scripts.p1_gate", *sys.argv[1:]],
                        int(spec["seed"]))
    try:
        g = dict(G1=comparison(spec, "mirror3d_nyu", "F0", "F1", "C"),
                 G2=comparison(spec, "hammer", "F0", "F1", "C"),
                 G3_nyu=comparison(spec, "mirror3d_nyu", "R0", "F1", "clean"),
                 G3_hammer=comparison(spec, "hammer", "R0", "F1", "clean"))
        checks = dict(
            G1=None if g["G1"].get("relative_change") is None else g["G1"]["relative_change"] <= float(spec["g1_max_relative_change"]),
            G2=None if g["G2"].get("mean_difference") is None else g["G2"]["mean_difference"] < 0,
            G3=None if None in (clearly_worse(g["G3_nyu"]), clearly_worse(g["G3_hammer"]))
            else not (clearly_worse(g["G3_nyu"]) or clearly_worse(g["G3_hammer"])))
        desc = {}
        for ds in spec["datasets"]:
            for ref, cand in (("R0", "F0"), ("R0", "F1"), ("R1", "F1")):
                desc[f"{cand}_vs_{ref}/{ds}"] = comparison(spec, ds, ref, cand, "C")
        desc["F1_vs_F0/mirror3d_nyu/train_holdout"] = comparison(spec, "mirror3d_nyu", "F0", "F1", "C", "train_holdout")
        desc["F0_vs_R0/clean/mirror3d_nyu"] = comparison(spec, "mirror3d_nyu", "R0", "F0", "clean")
        desc["F0_vs_R0/clean/hammer"] = comparison(spec, "hammer", "R0", "F0", "clean")
        passed = all(v is True for v in checks.values())
        clip = r0_clipping(spec)
        write_json(out / "gate.json", dict(checks=checks, passed=passed, gate=g, descriptive=desc, r0_clipping=clip))

        def row(name, r):
            ci = (f"{fmt(r.get('mean_difference'))} [{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}]"
                  if r.get("mean_difference") is not None else r.get("status", "NA"))
            rel = "NA" if r.get("relative_change") is None else f"{100 * r['relative_change']:+.1f}%"
            return (f"| {name} | {r['dataset']} {r['population']} | {r['region']} | {r['reference']} {fmt(r.get('reference_mean'))} | "
                    f"{r['candidate']} {fmt(r.get('candidate_mean'))} | {ci} | {rel} | {r.get('scene_count', 'NA')} |")
        head = ["| check | population | region | reference | candidate | difference [95% CI] | relative | scenes |",
                "|---|---|---|---:|---:|---|---:|---:|"]
        lines = ["# Mainline P1 gate: F1 (synthesised structured errors) vs F0 (same fine-tuning without them)", "",
                 f"Generated by `python -m scripts.p1_gate` ({out.name}) from `{a.spec}`. AbsRel, scene means; difference = "
                 "candidate - reference (negative = candidate better); paired scene bootstrap. Development diagnostics; "
                 "test never read.", "", "## Gate (plan sec. 7)", "", *head]
        lines += [row(k, r) for k, r in g.items()]
        lines += ["", f"- G1 (NYU val C, F1 vs F0, relative change <= {100 * float(spec['g1_max_relative_change']):.0f}%): {checks['G1']}",
                  f"- G2 (HAMMER C, F1 lower than F0): {checks['G2']}",
                  f"- G3 (N1: F1 clean pixels not clearly worse than R0 on either dataset): {checks['G3']}",
                  f"- R0 caches (uint16, saturate at 13.107 m; F0 / F1 are float32): {json.dumps(clip)}",
                  f"- **{'P1 passed: go to P2 (F2 and ablations)' if passed else 'P1 NOT passed: stop GPU spending, report, rethink the synthesis (plan sec. 7)'}**",
                  "", "## Descriptive (no decision)", "", *head]
        lines += [row(k, r) for k, r in desc.items()] + [""]
        (ROOT / "docs" / "omnidc_p1_gate.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_comparison_complete", formal_experiment_complete=False, passed=passed,
                       checks=checks, gate_sha256=sha256(out / "gate.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
