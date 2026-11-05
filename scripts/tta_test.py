"""Formal test, secondary and descriptive (docs/plan_test.md): CAPA on all readings vs CAPA on the readings T v1
trusts, as in S2 (scripts.tta_tmask), on the test split. GPU.

  python -m scripts.tta_test [--max-minutes 120]          # then: python -m scripts.tta_test --resume latest

Images: NYU every test image; HAMMER 1 frame per scene x sensor (9 images, about 12 min per image per variant).
An image on which CAPA returns NaN after its own retries is recorded as failed and left out of all variants (listed in
the report), instead of stopping the run. Same CAPA code, weights, config and per-image seed as the CAPA pre-check; T maps from the test trust caches with the
frozen conformal lambda (alpha 0.1). The prior is CAPA's own step 0. No rule: reported descriptively.
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
from tp.data.select import per_group_subsample
from tp.eval.run_eval import focus_scope
from tp.formal import guard
from tp.provenance import ROOT, sha256, start_run, write_json
from scripts import tta_precheck as T
from scripts.patch_precheck import image_metrics
from scripts.tta_tmask import pooled, reduction

DATA = {"nyu": dict(processed="data/processed/mirror3d_test_v1", per_group=None, crc="mirror3d_nyu",
                    trust="cache/trust_t_v1_real_sim_nofeat_nyu_test_v1"),
        "hammer": dict(processed="data/processed/hammer_test_v1", per_group=1, crc="hammer",
                       trust="cache/trust_t_v1_real_sim_nofeat_hammer_test_v1")}
CRC, ALPHA, SEED_BOOT = "results/trust_t_v1_crc_d20.json", "0.1", 20261105


def records(ds):
    rows = json.loads((ROOT / DATA[ds]["processed"] / "manifest.json").read_text(encoding="utf-8"))["records"]
    return per_group_subsample([r for r in rows if r["split"] == "test"], DATA[ds]["per_group"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    formal = guard()
    import torch
    from tp.trust.cache import TrustCache
    crc = json.loads((ROOT / CRC).read_text(encoding="utf-8"))
    lam = {ds: float(crc["datasets"][d["crc"]]["alphas"][ALPHA]["lambda_hat"]) for ds, d in DATA.items()}
    config = dict(data=DATA, crc=CRC, alpha=ALPHA, lambdas=lam, seed=T.SEED, align_points=T.ALIGN_POINTS, capa=T.CAPA)
    command = [sys.executable, "-m", "scripts.tta_test", *sys.argv[1:]]
    if a.resume is None:
        out = ROOT / "results" / f"tta_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        receipt = start_run(out, dict(config, formal_test=formal), command, SEED_BOOT)
    else:
        out = sorted((ROOT / "results").glob("tta_test_*"))[-1] if str(a.resume) == "latest" else (a.resume if a.resume.is_absolute() else ROOT / a.resume)
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        if {k: v for k, v in receipt["config"].items() if k != "formal_test"} != json.loads(json.dumps(config)):
            raise ValueError("resume requires the identical configuration")
        receipt.setdefault("resumed", []).append(command)
    t0 = time.time()
    try:
        info = T.fetch_capa()
        device = torch.device(a.device)
        proto, _ = T.load_protocol(device, info["weights_dir"])
        receipt["model_version"] = f"CAPA {T.CAPA['commit']} + UniDepthV2-L {info['weights_revision']}"
        tcache = {ds: TrustCache(ROOT / DATA[ds]["trust"]) for ds in DATA}
        per = out / "per_image"
        per.mkdir(exist_ok=True)
        todo = [(ds, r) for ds in DATA for r in records(ds)]
        for i, (ds, r) in enumerate(todo):
            name = f"{ds}__{Path(r['path']).stem}"
            if (per / f"{name}.json").is_file():
                continue
            if a.max_minutes is not None and time.time() - t0 > 60 * a.max_minutes:
                receipt.update(stage="partial_time_budget")
                print("time budget reached; continue with: python -m scripts.tta_test --resume latest", flush=True)
                return
            path = ROOT / DATA[ds]["processed"] / r["path"]
            h = sha256(path)
            if h != r["sha256"]:
                raise ValueError(f"sample hash mismatch: {path}")
            s = load_sample(path)
            rgb, cond, mask = T.tensors(s)
            trust = tcache[ds].trust(s, h, s.prompt.shape)
            keep = mask & torch.from_numpy(~(s.prompt_valid & (trust < lam[ds])))[None]
            preds, secs, failed = {}, {}, None
            for v, m in (("capa_raw", mask), ("tmask", keep)):
                torch.manual_seed(T.SEED)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(T.SEED)
                t1 = time.time()
                try:
                    pred, rec, _ = T.run_with_retries(proto, rgb, cond, m)
                except RuntimeError as e:      # CAPA returned NaN after its own retries: record, do not drop silently
                    if "NaN" not in str(e):
                        raise
                    failed = v
                    break
                secs[v] = time.time() - t1
                preds[v] = pred[0].float().cpu().numpy()
                if v == "capa_raw":
                    preds["prior"] = rec[0]
            if failed is not None:
                write_json(per / f"{name}.json", dict(dataset=ds, scene=s.scene, sensor=s.sensor, seconds=secs,
                                                      failed=failed, metrics=None))
                print(f"[{i + 1}/{len(todo)}] {name}: CAPA NaN after retries in variant {failed}; recorded as failed", flush=True)
                continue
            _, scope = focus_scope(s)
            metrics = {k: image_metrics(D, s.prompt, s.prompt_valid, s.gt, s.gt_valid, scope) for k, D in preds.items()}
            write_json(per / f"{name}.json", dict(dataset=ds, scene=s.scene, sensor=s.sensor, seconds=secs, metrics=metrics))
            print(f"[{i + 1}/{len(todo)}] {name}: " + ", ".join(f"{k} {v:.0f} s" for k, v in secs.items()), flush=True)
        rows = [json.loads((per / f"{ds}__{Path(r['path']).stem}.json").read_text(encoding="utf-8")) for ds, r in todo]
        rng = np.random.default_rng(SEED_BOOT)
        f = lambda x, d=4: "NA" if x is None else f"{x:.{d}f}"
        L = ["# Formal test, secondary: CAPA with and without the T v1 mask (descriptive)", "",
             f"Generated by `python -m scripts.tta_test` ({out.name}); test split read once after the method v1 freeze.", "",
             "| dataset | images / scenes | variant | AbsRel C | AbsRel clean | reduction vs CAPA raw (95 % CI, scene bootstrap) |",
             "|---|---:|---|---:|---:|---:|"]
        res = {}
        for ds in DATA:
            allx = [x for x in rows if x["dataset"] == ds]
            xs = [x for x in allx if x.get("metrics") is not None]
            failed = [f"{x['scene']}/{x['sensor']} ({x['failed']})" for x in allx if x.get("metrics") is None]
            sc = T._by_scene(xs)
            red = T.scene_bootstrap(sc, lambda g: reduction(g, "tmask"), rng)
            res[ds] = dict(images=len(xs), scenes=len(sc), failed_images=failed, absrel_C={k: pooled(xs, k) for k in ("prior", "capa_raw", "tmask")},
                           absrel_clean={k: pooled(xs, k, "clean") for k in ("prior", "capa_raw", "tmask")}, reduction=red)
            for k in ("prior", "capa_raw", "tmask"):
                r_ = f"{f(red[0], 3)} ({f(red[1], 3)} to {f(red[2], 3)})" if k == "tmask" else "-"
                L.append(f"| {ds} | {len(xs)} / {len(sc)} | {k} | {f(res[ds]['absrel_C'][k])} | {f(res[ds]['absrel_clean'][k])} | {r_} |")
        L.append("")
        for ds in DATA:
            if res[ds]["failed_images"]:
                L.append(f"{ds}: CAPA returned NaN after its retries on {len(res[ds]['failed_images'])} image(s), "
                         "left out of every variant above (same images for all variants): " + ", ".join(res[ds]["failed_images"]))
        L.append("")
        write_json(out / "tta_test.json", res)
        (ROOT / "docs" / "tta_test.md").write_text("\n".join(L), encoding="utf-8")
        receipt.update(stage="complete", result_sha256=sha256(out / "tta_test.json"))
        print("\n".join(L))
    except BaseException as e:
        if receipt.get("stage") in {"started", None}:
            receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
