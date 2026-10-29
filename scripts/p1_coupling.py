"""Is F1's loss on correct readings coupled to its correction of wrong ones through the trust channel?
(docs/plan_omnidc.md sec. 23; GPU, Windows, .venvs/baselines; inference only, development data, test never read)

  python -m scripts.p1_coupling

OMNI-DC lets a reading act on the depth in three ways: the backbone sees it, the depth integration (DDI) weighs it by
confidence_input x weights_input, and the DySPN refinement re-injects it. P1's F1 learned to lower confidence_input on
wrong readings (NYU AUROC 0.91) and on correct ones too (0.99 -> 0.62), and lost accuracy on correct readings image-wide.
Here confidence_input is replaced at inference (1/4-resolution cells, the head's own grid), nothing is retrained:
  F0, F1               as trained
  F1 | c = 1           F1 with full trust everywhere
  F1 | oracle correct  F1 with c = 1 on cells whose readings are all correct (GT-based; c kept elsewhere)
  F0 | oracle wrong    F0 with c = 0 on cells holding a wrong reading (GT-based; c kept elsewhere)
  F0 | F1's c          F0's network with F1's confidence map for the same image (deployable: two networks)
Weights: the AutoDL F0 / F1 final.pt (SHA from results/autodl_full_20261024/mirror/job/operator.json, checked by
baselines.omnidc_finetuned). Populations: NYU val + train-holdout (30), HAMMER dev 3 frames per scene x sensor (90).
Per condition, pixel-pooled AbsRel on clean pixels (tp.eval.regions) inside / outside the main scope and on C inside
the scope, and the mean confidence_input on all-correct cells (checks the replacement took effect).
Reading, fixed before the first run (NYU; gaps measured as F1 - F0):
  R_a  the clean loss flows through the trust channel: "F1 | oracle correct" closes >= 50 % of the F1 - F0 gap in
       clean-outside-scope AbsRel
  R_b  a separate trust signal keeps the gain: "F0 | F1's c" keeps >= 50 % of F1's C improvement over F0 and its
       clean-outside-scope AbsRel is <= 1.5 x F0's
  R_a and R_b -> coupling through shared trust; decoupling trust from depth is a viable method core (design next).
  not R_a     -> the loss sits in the depth network itself, not in trust; a trust-side fix is not enough.
  R_a, not R_b -> F1's correction needs its depth path as well; decoupling must be trained jointly.
Writes results/p1_coupling_<time>/ and docs/omnidc_p1_coupling.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import contextlib
from datetime import datetime
import json
import sys

import numpy as np

from scripts.cache_completion import model_input, select_records
from tp.data.schema import load_sample
from tp.data.select import per_group_subsample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json

TAU, CELL = 0.05, 4
SETS = {"nyu": dict(processed="data/processed/mirror3d_dev_v1", scope="holdout", per_group=None),
        "hammer": dict(processed="data/processed/hammer_dev_v1", scope="all", per_group=3)}
JOB = ROOT / "results" / "autodl_full_20261024" / "mirror" / "job" / "operator.json"
READING = dict(R_a_min_gap_closed=0.5, R_b_min_gain_kept=0.5, R_b_max_clean_ratio=1.5)


def cell_masks(prompt_valid, wrong, padded_shape, r=CELL):
    """(has a reading, holds a wrong reading) per r x r cell of the padded image (padding = no reading)."""
    H, W = padded_shape
    pad = lambda m: np.pad(m, ((0, H - m.shape[0]), (0, W - m.shape[1])))
    pool = lambda m: m.reshape(H // r, r, W // r, r).any(axis=(1, 3))
    return pool(pad(prompt_valid)), pool(pad(wrong))


@contextlib.contextmanager
def override_confidence(net, fn):
    """Replace OMNI-DC's confidence_input (5th backbone output) by fn(c) during the block."""
    original = net.backbone.forward

    def forward(*args, **kwargs):
        out = list(original(*args, **kwargs))
        out[4] = fn(out[4])
        return tuple(out)
    net.backbone.forward = forward
    try:
        yield
    finally:
        net.backbone.forward = original


def records(ds):
    d = SETS[ds]
    processed = ROOT / d["processed"]
    rows = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"], {"train", "val"}, d["scope"])
    return processed, per_group_subsample(rows, d["per_group"])


class Tally:
    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, values):
        if values.size:
            self.s[key] = self.s.get(key, 0.0) + float(values.sum()); self.n[key] = self.n.get(key, 0) + int(values.size)

    def summary(self):
        return {k: self.s[k] / self.n[k] for k in self.s}


def run_model(model, conditions, stored=None):
    """conditions: {name: maker(cells) -> fn or None}; cells = dict(has, wrong, correct, given). Returns metrics and,
    for the unmodified run, the confidence maps (to hand to the other model)."""
    import torch
    out, maps = {}, {}
    for ds in SETS:
        processed, recs = records(ds)
        tallies = {name: Tally() for name in conditions}
        for r in recs:
            path = processed / r["path"]
            if sha256(path) != r["sha256"]:
                raise ValueError(f"checksum mismatch {path}")
            s = load_sample(path)
            depth, _, _ = model_input(s, "raw", tau=TAU)
            H, W = depth.shape
            ph, pw, _ = model.padding(depth.shape)
            reg = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=TAU, band=5)
            has, wrong = cell_masks(s.prompt_valid, reg["C"], (H + ph, W + pw))
            dev = model.device
            t = lambda m: torch.from_numpy(m)[None, None].to(dev)
            cells = dict(has=t(has), wrong=t(wrong), correct=t(has & ~wrong),
                         given=None if stored is None else stored[(ds, s.key)].to(dev))
            scope = focus_scope(s)[1]
            gt = s.gt.astype(np.float64)
            for name, maker in conditions.items():
                fn = maker(cells)
                ctx = override_confidence(model.net, fn) if fn else contextlib.nullcontext()
                with torch.no_grad(), ctx:
                    o, _ = model.outputs(s.rgb, depth)
                pred = o["pred"][0, 0, :H, :W].float().cpu().numpy().astype(np.float64)
                c = o["confidence_input"]
                if name == "as trained":
                    maps[(ds, s.key)] = c.detach().cpu()
                ok = np.isfinite(pred) & (pred > 0)
                rel = lambda m: np.abs(pred[m] - gt[m]) / gt[m]
                tl = tallies[name]
                tl.add("clean_in_scope", rel(reg["clean"] & scope & ok))
                tl.add("clean_outside_scope", rel(reg["clean"] & ~scope & ok))
                tl.add("C_in_scope", rel(reg["C"] & scope & ok))
                tl.add("conf_correct_cells", c[cells["correct"]].float().cpu().numpy())
                tl.add("conf_wrong_cells", c[cells["wrong"] & cells["has"]].float().cpu().numpy())
        out[ds] = dict(images=len(recs), **{name: t_.summary() for name, t_ in tallies.items()})
        print(ds, json.dumps(out[ds]), flush=True)
    return out, maps


def main():
    import torch
    from baselines.omnidc_finetuned import FineTunedOmniDC
    job = json.loads(JOB.read_text(encoding="utf-8"))
    weights = {v: (job["run_dirs"][v], job["checkpoints"][v]["final"]["sha256"]) for v in ("f0", "f1")}
    out = ROOT / "results" / f"p1_coupling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, cell=CELL, sets=SETS, reading=READING, weights=weights),
                        [sys.executable, "-m", "scripts.p1_coupling", *sys.argv[1:]], 20261029)
    try:
        f1 = FineTunedOmniDC(weights["f1"][0], "final.pt", weights["f1"][1], "f1", device="cuda")
        r1, f1_maps = run_model(f1.base, {
            "as trained": lambda c: None,
            "c = 1": lambda c: (lambda x: torch.ones_like(x)),
            "oracle correct": lambda c: (lambda x: torch.where(c["correct"], torch.ones_like(x), x))})
        del f1; torch.cuda.empty_cache()
        f0 = FineTunedOmniDC(weights["f0"][0], "final.pt", weights["f0"][1], "f0", device="cuda")
        r0, _ = run_model(f0.base, {
            "as trained": lambda c: None,
            "oracle wrong": lambda c: (lambda x: torch.where(c["wrong"] & c["has"], torch.zeros_like(x), x)),
            "F1's c": lambda c: (lambda x: c["given"].to(x.dtype))}, stored=f1_maps)
        del f0; torch.cuda.empty_cache()
        res = {ds: {**{f"F1 {k}": v for k, v in r1[ds].items() if k != "images"},
                    **{f"F0 {k}": v for k, v in r0[ds].items() if k != "images"}, "images": r1[ds]["images"]} for ds in SETS}
        n = res["nyu"]
        f0_, f1_ = n["F0 as trained"], n["F1 as trained"]
        gap = f1_["clean_outside_scope"] - f0_["clean_outside_scope"]
        closed = (f1_["clean_outside_scope"] - n["F1 oracle correct"]["clean_outside_scope"]) / gap if gap > 0 else None
        gain = f0_["C_in_scope"] - f1_["C_in_scope"]
        kept = (f0_["C_in_scope"] - n["F0 F1's c"]["C_in_scope"]) / gain if gain > 0 else None
        clean_ratio = n["F0 F1's c"]["clean_outside_scope"] / f0_["clean_outside_scope"]
        checks = dict(R_a=closed is not None and closed >= READING["R_a_min_gap_closed"],
                      R_b=kept is not None and kept >= READING["R_b_min_gain_kept"] and clean_ratio <= READING["R_b_max_clean_ratio"])
        verdict = ("coupling through shared trust: decoupling trust from depth is a viable method core" if all(checks.values()) else
                   "the clean loss sits in the depth network itself; a trust-side fix is not enough" if not checks["R_a"] else
                   "F1's correction also needs its depth path; decoupling must be trained jointly")
        summary = dict(gap_closed_by_oracle_correct=closed, F1_gain_kept_by_F0_with_F1_c=kept,
                       clean_ratio_F0_with_F1_c=clean_ratio, checks=checks, verdict=verdict)
        write_json(out / "coupling.json", dict(results=res, summary=summary))
        fmt = lambda v, p=4: "NA" if v is None else f"{v:.{p}f}"
        lines = ["# Trust and correction in F1: confidence replaced at inference (P1 diagnostic)", "",
                 f"Generated by `python -m scripts.p1_coupling` ({out.name}). AutoDL F0 / F1 final weights; confidence_input "
                 "replaced on its 1/4-resolution cells, nothing retrained; oracle rows read GT and are diagnostics only. "
                 "Pixel-pooled AbsRel; development data, test never read.", "",
                 "| condition | dataset | AbsRel clean in scope | AbsRel clean outside scope | AbsRel C | conf on correct cells | conf on wrong cells |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for ds, x in res.items():
            for name in [k for k in x if k != "images"]:
                v = x[name]
                lines.append(f"| {name} | {ds} ({x['images']}) | {fmt(v.get('clean_in_scope'))} | {fmt(v.get('clean_outside_scope'))} | "
                             f"{fmt(v.get('C_in_scope'))} | {fmt(v.get('conf_correct_cells'), 3)} | {fmt(v.get('conf_wrong_cells'), 3)} |")
        lines += ["", f"- NYU: oracle trust on correct cells closes {fmt(closed, 2)} of F1's clean-outside gap to F0 "
                  f"(R_a needs >= {READING['R_a_min_gap_closed']}): {checks['R_a']}",
                  f"- NYU: F0 with F1's confidence keeps {fmt(kept, 2)} of F1's C gain, clean-outside {fmt(clean_ratio, 2)} x F0 "
                  f"(R_b needs >= {READING['R_b_min_gain_kept']} and <= {READING['R_b_max_clean_ratio']}): {checks['R_b']}",
                  f"- **{verdict}**", ""]
        (ROOT / "docs" / "omnidc_p1_coupling.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", summary=summary, coupling_sha256=sha256(out / "coupling.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
