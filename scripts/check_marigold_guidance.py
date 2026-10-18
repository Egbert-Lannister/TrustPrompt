"""Stage M0 of the Marigold-DC guidance track (docs/marigold_guidance_plan.md): is the re-implemented loop faithful,
and what does one image cost? (GPU, Windows, .venvs/baselines)

  python -m scripts.check_marigold_guidance [--n 3]

On n NYU val images (evenly spaced, raw prompt):
  A   upstream MarigoldDepthCompletionPipeline (the cached raw / tmask / oracle_mask variants use this path)
  A2  the same again (GPU noise floor)
  B   baselines.guidance.complete with Guide.equality() (the loop the guided variants use)
  G   baselines.guidance.complete with the tonesided guide (cost of the guided objective; not compared)
Predictions are compared after the cache's 0.2 mm quantisation. Rule (fixed before running, same as the LingBot
check): the loop is faithful iff  max_image |C AbsRel(B) - C AbsRel(A)| < 1e-3  and  pooled p99 of
|B - A| / A < 0.5 %. Seconds and peak GPU memory per call are recorded for the M1 budget.
Writes results/marigold_guidance_check_<time>/ and docs/marigold_guidance_check.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
import json
import sys
import time

import numpy as np

from baselines.completion import build, registry
from baselines.guidance import complete
from scripts.cache_completion import CONFIG, crc_threshold, model_input, select_records
from tp.data.schema import load_sample
from tp.eval.consistency import main_c_absrel, quantize_cache, relative_difference
from tp.provenance import ROOT, sha256, start_run, write_json

TAU = 0.05
MAX_DELTA_C = 1e-3
MAX_P99_REL = 5e-3


def timed(torch, fn):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 2 ** 30


def main():
    import torch
    import yaml
    from PIL import Image
    from tp.trust.cache import TrustCache
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = registry()["models"]["marigold_dc"]["settings"]
    args = {k: settings[k] for k in ("num_inference_steps", "ensemble_size", "processing_resolution", "seed")}
    out = ROOT / "results" / f"marigold_guidance_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(n=a.n, tau=TAU, max_delta_c=MAX_DELTA_C, max_p99_rel=MAX_P99_REL, settings=settings),
                        [sys.executable, "-m", "scripts.check_marigold_guidance", *sys.argv[1:]], int(cfg["seed"]))
    try:
        model = build("marigold_dc", device="cuda")
        receipt["gpu"] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                              total_gb=torch.cuda.get_device_properties(0).total_memory / 2 ** 30)
        d = cfg["datasets"]["nyu"]
        processed = ROOT / d["processed"]
        records = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"],
                                 {"val"}, "holdout")
        records = [records[i] for i in np.linspace(0, len(records) - 1, min(a.n, len(records))).round().astype(int)]
        tcache, (lam, _) = TrustCache(d["trust_cache"]), crc_threshold(cfg, d["crc_dataset"])
        rows, pooled = [], []
        for r in records:
            path = processed / r["path"]; h = sha256(path)
            if h != r["sha256"]:
                raise ValueError(f"checksum mismatch {path}")
            s = load_sample(path)
            depth, _, _ = model_input(s, "raw", tau=TAU)
            rgb = Image.fromarray(s.rgb)
            upstream = lambda: model.pipe(image=rgb, sparse_depth=depth, **args)
            A, tA, mA = timed(torch, upstream)
            A2, _, _ = timed(torch, upstream)
            B, tB, mB = timed(torch, lambda: complete(model.pipe, rgb, depth, None, **args))
            gdepth, guide, ginfo = model_input(s, "tonesided", tau=TAU, trust=tcache.trust(s, h, s.prompt.shape), threshold=lam)
            _, tG, mG = timed(torch, lambda: complete(model.pipe, rgb, gdepth, guide, **args))
            A, A2, B = (quantize_cache(np.asarray(x, np.float32)) for x in (A, A2, B))
            rel, noise = relative_difference(A, B), relative_difference(A, A2)
            cA, cB = main_c_absrel(s, A, TAU), main_c_absrel(s, B, TAU)
            rows.append(dict(sensor=s.sensor, scene=s.scene, frame=s.frame, shape=list(depth.shape),
                             b_vs_a=dict(p99=float(np.quantile(rel, 0.99)), max=float(rel.max())),
                             noise_floor=dict(p99=float(np.quantile(noise, 0.99)), max=float(noise.max())),
                             c_absrel_a=cA, c_absrel_b=cB, delta_c=None if cA is None or cB is None else abs(cA - cB),
                             seconds=dict(upstream=tA, equality_loop=tB, tonesided_loop=tG),
                             peak_gb=dict(upstream=mA, equality_loop=mB, tonesided_loop=mG), tonesided_flagged=ginfo["flagged"]))
            pooled.append(np.random.default_rng(0).choice(rel, min(50_000, rel.size), replace=False))
            print(f"{s.key}: p99 |B-A|/A {rows[-1]['b_vs_a']['p99']:.2e} (noise {rows[-1]['noise_floor']['p99']:.2e}), "
                  f"dC {rows[-1]['delta_c']}, {tA:.0f}s / {tB:.0f}s / {tG:.0f}s, peak {mA:.1f} / {mB:.1f} / {mG:.1f} GB", flush=True)
            write_json(out / "check_partial.json", rows)
        pooled = np.concatenate(pooled)
        dcs = [x["delta_c"] for x in rows if x["delta_c"] is not None]
        sec = {k: float(np.mean([x["seconds"][k] for x in rows])) for k in rows[0]["seconds"]}
        peak = {k: float(max(x["peak_gb"][k] for x in rows)) for k in rows[0]["peak_gb"]}
        summary = dict(images=len(rows), pooled_p99=float(np.quantile(pooled, 0.99)), pooled_max=float(pooled.max()),
                       max_delta_c=float(max(dcs)) if dcs else None, images_with_c=len(dcs),
                       noise_floor_p99_max=float(max(x["noise_floor"]["p99"] for x in rows)),
                       mean_seconds=sec, max_peak_gb=peak)
        summary["faithful"] = bool(summary["max_delta_c"] is not None and summary["max_delta_c"] < MAX_DELTA_C
                                   and summary["pooled_p99"] < MAX_P99_REL)
        write_json(out / "check.json", dict(summary=summary, rows=rows))
        lines = ["# Marigold-DC guidance loop vs upstream (stage M0)", "",
                 f"Generated by `python -m scripts.check_marigold_guidance` ({out.name}); {receipt['gpu']['name']}. "
                 f"{len(rows)} NYU val images, raw prompt, settings {args}. Predictions compared after 0.2 mm quantisation.", "",
                 f"- |B - A| / A (re-implemented loop, equality guide vs upstream): p99 {summary['pooled_p99']:.2e}, max {summary['pooled_max']:.2e}",
                 f"- noise floor (upstream run twice), worst p99: {summary['noise_floor_p99_max']:.2e}",
                 f"- main-scope C AbsRel |B - A| per image ({summary['images_with_c']} images): max {summary['max_delta_c']}",
                 f"- seconds per image (upstream / equality loop / tonesided loop): "
                 + " / ".join(f"{v:.1f}" for v in sec.values()),
                 "- peak GPU memory GB (same order): " + " / ".join(f"{v:.2f}" for v in peak.values()),
                 f"- rule (fixed beforehand): faithful iff max |delta C AbsRel| < {MAX_DELTA_C} and pooled p99 < {MAX_P99_REL}",
                 f"- **verdict: {'faithful, guided variants may run' if summary['faithful'] else 'NOT faithful, do not run the guided variants'}**", ""]
        (ROOT / "docs" / "marigold_guidance_check.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="check_complete", summary=summary, check_sha256=sha256(out / "check.json"))
        print("\n".join(lines))
        if not summary["faithful"]:
            raise SystemExit(2)
    except BaseException as e:
        if not isinstance(e, SystemExit):
            receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
