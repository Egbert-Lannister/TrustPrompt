"""Mainline P0-D1 (docs/plan_omnidc.md sec. 7): does OMNI-DC's own input-confidence head see the wrong readings?
(GPU, Windows, .venvs/baselines; development diagnostics, test never read)

  python -m scripts.omnidc_confidence

OMNI-DC v1.1 predicts a per-pixel confidence for the input depth (backbone cfi head, sigmoid) and uses it as the
weight of the input-depth term in its depth integration. For every valid prompt pixel with valid GT we score
1 - confidence (the head's map is at 1/4 resolution; nearest upsampling) and ask how well it separates
contaminated pixels (|P - GT| / GT > tau, region C) from correct ones (<= tau): pixel AUROC, pooled per dataset,
in the main scope (NYU mirror / HAMMER object pixels) and on the whole image. The same pixels scored with 1 - T
(our detector: T v1 on NYU, cross-fitted T on HAMMER) give the reference. Also: mean confidence on C and on
correct pixels, and the share of C pixels with confidence < 0.5.
Reading fixed in the plan: AUROC >= 0.8 -> detection is not the bottleneck, weight module B; < 0.7 -> module A is
needed. Populations: NYU val + train-holdout scenes (30 images); HAMMER all 10 dev scenes (1143 images).
Writes results/omnidc_confidence_<time>/ and docs/omnidc_confidence.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from datetime import datetime
import json
import sys

import numpy as np

from baselines.completion import build
from scripts.cache_completion import CONFIG, crc_threshold, model_input, select_records
from tp.data.schema import load_sample
from tp.eval.auroc import ScoreHistogram
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.trust.cache import TrustCache

TAU = 0.05
SETS = {"nyu": dict(scope="holdout", trust="cache/trust_t_v1_real_sim_nofeat_nyu_dev_v1"),
        "hammer": dict(scope="all", trust="cache/trust_t_v1cf_real_sim_nofeat_hammer_dev_v1")}


def full_res(conf, shape, rate):
    """Head map [1,1,h,w] on the padded image downsampled by `rate` -> HxW (nearest), cropped to the image."""
    c = conf[0, 0].float().cpu().numpy()
    return np.repeat(np.repeat(c, rate, 0), rate, 1)[:shape[0], :shape[1]]


class Tally:
    def __init__(self):
        self.omni, self.t = ScoreHistogram(), ScoreHistogram()
        self.sum_c = self.n_c = self.sum_ok = self.n_ok = self.low_c = 0

    def add(self, conf, trust, wrong, where):
        self.omni.add(1 - conf[where], wrong[where]); self.t.add(1 - trust[where], wrong[where])
        c, ok = conf[where & wrong], conf[where & ~wrong]
        self.sum_c += float(c.sum()); self.n_c += c.size; self.low_c += int((c < 0.5).sum())
        self.sum_ok += float(ok.sum()); self.n_ok += ok.size

    def summary(self):
        return dict(auroc_omnidc=self.omni.auroc, auroc_T=self.t.auroc, pixels=self.omni.counts(),
                    mean_conf_C=self.sum_c / self.n_c if self.n_c else None,
                    mean_conf_correct=self.sum_ok / self.n_ok if self.n_ok else None,
                    share_C_conf_below_half=self.low_c / self.n_c if self.n_c else None)


def main():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / "results" / f"omnidc_confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(tau=TAU, sets=SETS), [sys.executable, "-m", "scripts.omnidc_confidence", *sys.argv[1:]],
                        int(cfg["seed"]))
    try:
        model = build("omnidc", device="cuda")
        receipt.update(model_deviations=model.deviations, model_load_info=model.load_info)
        payload = {}
        for ds, d in SETS.items():
            dcfg = cfg["datasets"][ds]
            processed = ROOT / dcfg["processed"]
            records = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"],
                                     {"train", "val"}, d["scope"])
            tc = TrustCache(d["trust"])
            lam = None if tc.per_item else crc_threshold(cfg, dcfg["crc_dataset"])[0]
            tallies = {"main": Tally(), "full": Tally()}
            for k, r in enumerate(records):
                path = processed / r["path"]; h = sha256(path)
                if h != r["sha256"]:
                    raise ValueError(f"checksum mismatch {path}")
                s = load_sample(path)
                depth, _, _ = model_input(s, "raw", tau=TAU)
                outputs, _ = model.outputs(s.rgb, depth)
                conf = full_res(outputs["confidence_input"], depth.shape, model.args.backbone_output_downsample_rate)
                trust = tc.trust(s, h, s.prompt.shape)
                both = s.prompt_valid & s.gt_valid
                wrong = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=TAU, band=5)["C"]
                tallies["main"].add(conf, trust, wrong, both & focus_scope(s)[1])
                tallies["full"].add(conf, trust, wrong, both)
                if k % 50 == 0:
                    print(f"{ds} {k + 1}/{len(records)} {s.key}", flush=True)
            payload[ds] = dict(images=len(records), trust_cache=d["trust"], threshold=lam,
                               **{reg: t.summary() for reg, t in tallies.items()})
            print(ds, json.dumps({reg: payload[ds][reg] for reg in tallies}), flush=True)
        write_json(out / "confidence.json", payload)
        fmt = lambda v, p=3: "NA" if v is None else f"{v:.{p}f}"
        lines = ["# OMNI-DC's input confidence on wrong prompt pixels (mainline P0-D1)", "",
                 f"Generated by `python -m scripts.omnidc_confidence` ({out.name}). Positive = contaminated prompt pixel "
                 f"(|P - GT| / GT > {TAU}), negative = correct prompt pixel; score 1 - confidence. T: our detector on the same "
                 "pixels (NYU T v1, HAMMER cross-fitted). Development diagnostics; test never read.", "",
                 "| Dataset | Region | images | C px | correct px | AUROC OMNI-DC | AUROC T | mean conf C | mean conf correct | C with conf < 0.5 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for ds, p in payload.items():
            for reg in ("main", "full"):
                x = p[reg]
                lines.append(f"| {ds} | {reg} | {p['images']} | {x['pixels']['positive']} | {x['pixels']['negative']} | "
                             f"{fmt(x['auroc_omnidc'])} | {fmt(x['auroc_T'])} | {fmt(x['mean_conf_C'])} | "
                             f"{fmt(x['mean_conf_correct'])} | {fmt(x['share_C_conf_below_half'])} |")
        lines += ["", "Reading fixed in docs/plan_omnidc.md sec. 7: main-scope AUROC >= 0.8 -> detection is not the "
                  "bottleneck (weight module B); < 0.7 -> module A is needed; in between, both.", ""]
        (ROOT / "docs" / "omnidc_confidence.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="descriptive_diagnostic_complete", confidence_sha256=sha256(out / "confidence.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
