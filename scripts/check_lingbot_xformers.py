"""Does the xFormers-free LingBot path change the cached predictions? (GPU, Windows, .venvs/baselines with xformers)

  python -m scripts.check_lingbot_xformers [--n 30]

The 6 LingBot caches (cache/bl_lingbot_*) were made without xFormers: nested-tensor blocks ran each sample's
sequence through the standard block and attention used torch SDPA (baselines.completion.patch_nested_blocks).
This script re-runs the official xFormers path on n NYU + n HAMMER images of the raw-prompt caches and compares,
image by image, with the cached predictions (same uint16 0.2 mm storage):
  pixel level   |d_xf - d_cache| / d_xf over valid pixels: median, p99, max per image; pooled over a
                fixed 50k-pixel sample per image
  metric level  main-scope C AbsRel (vs GT) with each prediction; |difference| per image
As a noise floor the fallback path is also re-run in the same process and compared with the cache.
Decision (fixed before running, docs/baselines.md): the differences are negligible, and the cached LingBot results
stand, iff  max_image |delta C AbsRel| < 1e-3  and  pooled p99 pixel relative difference < 0.5 %.
Otherwise the 6 LingBot caches are re-made with xFormers.
Writes results/lingbot_xformers_check_<time>/ (receipt, check.json) and docs/lingbot_xformers_check.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

import numpy as np

from baselines.cached_prior import CachedPrior
from baselines.completion import build, patch_nested_blocks, registry, source_dir
from scripts.cache_completion import CONFIG, model_input
from tp.data.schema import load_sample
from tp.eval.consistency import main_c_absrel, quantize_cache
from tp.provenance import ROOT, sha256, start_run, write_json

TAU = 0.05
MAX_DELTA_C = 1e-3
MAX_P99_REL = 5e-3


def xformers_status():
    import torch
    out = dict(import_ok=False)
    try:
        import xformers
        import xformers.ops as xops
        out.update(import_ok=True, version=xformers.__version__)
        q = torch.randn(1, 64, 4, 32, device="cuda", dtype=torch.float16)
        o = xops.memory_efficient_attention(q, q, q)
        ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, q, q))).transpose(1, 2)
        out.update(cuda_kernel_ok=bool(torch.isfinite(o).all()), kernel_max_abs_diff=float((o - ref).abs().max()))
    except Exception as e:                       # import or CUDA-build mismatch: report, do not hide
        out.update(error=repr(e))
    return out


def main():
    import torch
    import yaml
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    label = registry()["models"]["lingbot"]["label"]
    out = ROOT / "results" / f"lingbot_xformers_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(n=a.n, max_delta_c=MAX_DELTA_C, max_p99_rel=MAX_P99_REL, tau=TAU),
                        [sys.executable, "-m", "scripts.check_lingbot_xformers", *sys.argv[1:]], int(cfg["seed"]))
    try:
        xs = xformers_status()
        receipt["xformers"] = xs
        write_json(out / "receipt.json", receipt)
        if not xs.get("cuda_kernel_ok"):
            raise RuntimeError(f"xFormers is not usable here: {xs}")
        model = build("lingbot", device="cuda")             # puts the LingBot checkout on sys.path
        import mdm.model.dinov2_rgbd.layers.attention as la
        import mdm.model.dinov2_rgbd.layers.block as lb
        if not (la.XFORMERS_AVAILABLE and lb.XFORMERS_AVAILABLE) or model.load_info.get("xformers") is not True:
            raise RuntimeError("LingBot did not pick up xFormers")
        official_forward = lb.NestedTensorBlock.forward      # unpatched: xFormers is available
        rows = []
        for ds in ("nyu", "hammer"):
            d = cfg["datasets"][ds]
            cache = CachedPrior(ROOT / f"cache/bl_lingbot_raw_{ds}_dev_v1", expected_model=label)
            processed = ROOT / d["processed"]
            recs = {(r["dataset"], r["sensor"], r["scene"], r["frame"]): r
                    for r in json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"]}
            keys = sorted(cache.rows)
            keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(a.n, len(keys))).round().astype(int)]
            for key in keys:
                r = recs[key]; path = processed / r["path"]; h = sha256(path)
                s = load_sample(path)
                depth_in, _, _ = model_input(s, "raw", tau=TAU)
                cached = cache.predict(s, h).astype(np.float64)
                torch.manual_seed(int(cfg["seed"]))                  # official xFormers path
                xf = quantize_cache(model.predict(s.rgb, depth_in, s.K_rgb)[0])
                la.XFORMERS_AVAILABLE = lb.XFORMERS_AVAILABLE = False      # the path the caches used
                patch_nested_blocks(lb)
                torch.manual_seed(int(cfg["seed"]))
                try:
                    fb = quantize_cache(model.predict(s.rgb, depth_in, s.K_rgb)[0])
                finally:                                                    # back to the official path
                    lb.NestedTensorBlock.forward = official_forward
                    lb.NestedTensorBlock._trustprompt_patched = False
                    la.XFORMERS_AVAILABLE = lb.XFORMERS_AVAILABLE = True
                ok = np.isfinite(xf) & np.isfinite(cached) & np.isfinite(fb)
                rel_xc = np.abs(xf[ok] - cached[ok]) / xf[ok]; rel_fc = np.abs(fb[ok] - cached[ok]) / fb[ok]
                cx, cc = main_c_absrel(s, xf, TAU), main_c_absrel(s, cached, TAU)
                rows.append(dict(dataset=ds, sensor=s.sensor, scene=s.scene, frame=s.frame,
                                 xf_vs_cache=dict(median=float(np.median(rel_xc)), p99=float(np.quantile(rel_xc, 0.99)),
                                                  max=float(rel_xc.max())),
                                 fallback_vs_cache=dict(median=float(np.median(rel_fc)), p99=float(np.quantile(rel_fc, 0.99)),
                                                        max=float(rel_fc.max())),
                                 c_absrel_xf=cx, c_absrel_cache=cc,
                                 delta_c=None if cx is None or cc is None else abs(cx - cc),
                                 _rel=np.random.default_rng(0).choice(rel_xc, min(50_000, rel_xc.size), replace=False)))
                print(f"{ds} {s.key}: p99 rel xf-cache {rows[-1]['xf_vs_cache']['p99']:.2e}, "
                      f"fallback-cache {rows[-1]['fallback_vs_cache']['p99']:.2e}, dC {rows[-1]['delta_c']}", flush=True)
                write_json(out / "check_partial.json", [{k: v for k, v in x.items() if k != "_rel"} for x in rows])
        pooled = np.concatenate([x.pop("_rel") for x in rows])
        dcs = [x["delta_c"] for x in rows if x["delta_c"] is not None]
        summary = dict(images=len(rows), pooled_p50=float(np.median(pooled)), pooled_p99=float(np.quantile(pooled, 0.99)),
                       pooled_max=float(pooled.max()), max_delta_c=float(max(dcs)) if dcs else None,
                       mean_delta_c=float(np.mean(dcs)) if dcs else None, images_with_c=len(dcs),
                       noise_floor_p99_max=float(max(x["fallback_vs_cache"]["p99"] for x in rows)))
        summary["negligible"] = bool(summary["max_delta_c"] is not None and summary["max_delta_c"] < MAX_DELTA_C
                                     and summary["pooled_p99"] < MAX_P99_REL)
        write_json(out / "check.json", dict(summary=summary, rows=rows))
        lines = ["# LingBot: xFormers path vs the cached xFormers-free path", "",
                 f"Generated by `python -m scripts.check_lingbot_xformers` ({out.name}). xFormers {xs.get('version')}; "
                 f"{len(rows)} raw-prompt images (NYU + HAMMER, evenly spaced over each cache). Predictions compared after "
                 "the cache's 0.2 mm quantisation.", "",
                 f"- pixel |xF - cache| / xF: median {summary['pooled_p50']:.2e}, p99 {summary['pooled_p99']:.2e}, max {summary['pooled_max']:.2e}",
                 f"- main-scope C AbsRel, |xF - cache| per image ({summary['images_with_c']} images with C): "
                 f"max {summary['max_delta_c']}, mean {summary['mean_delta_c']}",
                 f"- noise floor (fallback re-run vs cache), worst p99: {summary['noise_floor_p99_max']:.2e}",
                 f"- rule (fixed beforehand): negligible iff max |delta C AbsRel| < {MAX_DELTA_C} and pooled p99 < {MAX_P99_REL}",
                 f"- **verdict: {'negligible, cached LingBot results stand' if summary['negligible'] else 'NOT negligible, re-make the 6 LingBot caches with xFormers'}**", ""]
        (ROOT / "docs" / "lingbot_xformers_check.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="check_complete", summary=summary, check_sha256=sha256(out / "check.json"))
        print("\n".join(lines))
    except BaseException as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
