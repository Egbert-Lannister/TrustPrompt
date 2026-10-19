"""Stage M0 (v2) of the Marigold-DC guidance track (docs/marigold_guidance_plan.md): is the re-implemented loop the
upstream loop, and does a production image fit the card? (GPU, Windows, .venvs/baselines)

  python -m scripts.check_marigold_guidance [--n 3]

Why v2 (decisions sec. 20, M0 v1 result): v1 compared the two loops at production settings with a fixed tolerance
(p99 < 0.5 %), but the upstream pipeline run twice already differs by p99 up to 1.6 % (non-deterministic CUDA
kernels in bf16, amplified over 50 guided steps), so v1 could not separate the code from GPU noise.

Part 1, exactness. Deterministic CUDA (torch.use_deterministic_algorithms, warn_only; cudnn deterministic; cuBLAS
workspace :4096:8), the production model and dtype, a small setting (384 px, 10 steps) so it is quick. On n NYU val
images (raw prompt): A upstream pipeline, A2 upstream again, B baselines.guidance.complete with Guide.equality().
  If A == A2 bitwise (determinism attained): exact iff B == A bitwise.
  Otherwise: exact iff pooled p99 |B - A| / A <= 2 x pooled p99 |A2 - A| / A (B indistinguishable from a re-run).
Part 2, cost. Production settings (registry, deterministic mode off), one image: upstream and the tonesided guide.
  fits iff the peak allocated GPU memory of both is < 95 % of the card (no paging into shared memory).
Passed iff exact and fits; the script then prints the M1 estimate (186 calls x mean seconds).
Writes results/marigold_guidance_check_<time>/ and docs/marigold_guidance_check.md.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # must precede CUDA initialisation
import argparse
from datetime import datetime
import json
import sys
import time
import warnings

import numpy as np

from baselines.completion import build, registry
from baselines.guidance import complete
from scripts.cache_completion import CONFIG, crc_threshold, model_input, select_records
from tp.data.schema import load_sample
from tp.eval.consistency import relative_difference
from tp.provenance import ROOT, sha256, start_run, write_json

TAU = 0.05
EXACT_SETTINGS = dict(num_inference_steps=10, processing_resolution=384)
NOISE_FACTOR = 2.0
MAX_MEMORY_SHARE = 0.95
M1_CALLS = 30 * 6 + 6                      # 6 variants x 30 NYU images + one preflight image per variant


def timed(torch, fn):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return np.asarray(out, np.float32), time.perf_counter() - t, torch.cuda.max_memory_allocated() / 2 ** 30


def deterministic(torch, on):
    torch.use_deterministic_algorithms(on, warn_only=True)
    torch.backends.cudnn.deterministic = on
    torch.backends.cudnn.benchmark = False


def nyu_val(cfg, n):
    d = cfg["datasets"]["nyu"]
    processed = ROOT / d["processed"]
    records = select_records(json.loads((processed / "manifest.json").read_text(encoding="utf-8"))["records"],
                             {"val"}, "holdout")
    for i in np.linspace(0, len(records) - 1, min(n, len(records))).round().astype(int):
        path = processed / records[i]["path"]; h = sha256(path)
        if h != records[i]["sha256"]:
            raise ValueError(f"checksum mismatch {path}")
        yield load_sample(path), h


def main():
    import torch
    import yaml
    from PIL import Image
    from tp.trust.cache import TrustCache
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = registry()["models"]["marigold_dc"]["settings"]
    prod = {k: settings[k] for k in ("num_inference_steps", "ensemble_size", "processing_resolution", "seed")}
    small = dict(prod, **EXACT_SETTINGS)
    out = ROOT / "results" / f"marigold_guidance_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    receipt = start_run(out, dict(version=2, n=a.n, exact_settings=small, production_settings=prod, noise_factor=NOISE_FACTOR,
                                  max_memory_share=MAX_MEMORY_SHARE), [sys.executable, "-m", "scripts.check_marigold_guidance",
                                                                       *sys.argv[1:]], int(cfg["seed"]))
    try:
        model = build("marigold_dc", device="cuda")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        receipt["gpu"] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda, total_gb=total_gb)
        receipt["model_deviations"] = model.deviations
        upstream = lambda rgb, depth, args: model.pipe(image=rgb, sparse_depth=depth, **args)

        # Part 1: exactness
        deterministic(torch, True)
        rows, nondet_warnings = [], set()
        for s, _ in nyu_val(cfg, a.n):
            depth, _, _ = model_input(s, "raw", tau=TAU)
            rgb = Image.fromarray(s.rgb)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                A, tA, _ = timed(torch, lambda: upstream(rgb, depth, small))
                A2, _, _ = timed(torch, lambda: upstream(rgb, depth, small))
                B, tB, _ = timed(torch, lambda: complete(model.pipe, rgb, depth, None, **small))
            nondet_warnings |= {str(w.message).split("\n")[0][:200] for w in caught if "deterministic" in str(w.message)}
            rows.append(dict(scene=s.scene, frame=s.frame, a_equals_a2=bool(np.array_equal(A, A2, equal_nan=True)),
                             b_equals_a=bool(np.array_equal(B, A, equal_nan=True)),
                             p99_b_vs_a=float(np.quantile(relative_difference(A, B), 0.99)),
                             p99_a2_vs_a=float(np.quantile(relative_difference(A, A2), 0.99)), seconds=dict(upstream=tA, loop=tB)))
            print(f"[exact] {s.key}: A==A2 {rows[-1]['a_equals_a2']}, B==A {rows[-1]['b_equals_a']}, "
                  f"p99 B-A {rows[-1]['p99_b_vs_a']:.2e}, p99 A2-A {rows[-1]['p99_a2_vs_a']:.2e}", flush=True)
            write_json(out / "check_partial.json", dict(exact=rows))
        determinism = all(r["a_equals_a2"] for r in rows)
        if determinism:
            exact = all(r["b_equals_a"] for r in rows)
            exact_rule = "A == A2 on every image, so B == A bitwise is required"
        else:
            p_ba, p_aa = max(r["p99_b_vs_a"] for r in rows), max(r["p99_a2_vs_a"] for r in rows)
            exact = p_ba <= NOISE_FACTOR * p_aa
            exact_rule = f"determinism not attained; max p99 B-A {p_ba:.2e} <= {NOISE_FACTOR} x max p99 A2-A {p_aa:.2e} required"
        deterministic(torch, False)

        # Part 2: cost at production settings
        tcache, (lam, _) = TrustCache(cfg["datasets"]["nyu"]["trust_cache"]), crc_threshold(cfg, cfg["datasets"]["nyu"]["crc_dataset"])
        s, h = next(nyu_val(cfg, 1))
        depth, _, _ = model_input(s, "raw", tau=TAU)
        gdepth, guide, _ = model_input(s, "tonesided", tau=TAU, trust=tcache.trust(s, h, s.prompt.shape), threshold=lam)
        rgb = Image.fromarray(s.rgb)
        _, tU, mU = timed(torch, lambda: upstream(rgb, depth, prod))
        _, tG, mG = timed(torch, lambda: complete(model.pipe, rgb, gdepth, guide, **prod))
        cost = dict(image=s.key, seconds=dict(upstream=tU, tonesided=tG), peak_gb=dict(upstream=mU, tonesided=mG))
        fits = max(mU, mG) < MAX_MEMORY_SHARE * total_gb
        m1_hours = M1_CALLS * (tU + tG) / 2 / 3600
        summary = dict(determinism_attained=determinism, exact=exact, exact_rule=exact_rule, fits_memory=fits,
                       total_gb=total_gb, cost=cost, m1_estimate_hours=m1_hours, nondeterministic_ops=sorted(nondet_warnings))
        summary["passed"] = bool(exact and fits)
        write_json(out / "check.json", dict(summary=summary, exact=rows))
        lines = ["# Marigold-DC guidance loop vs upstream (stage M0, v2)", "",
                 f"Generated by `python -m scripts.check_marigold_guidance` ({out.name}); {receipt['gpu']['name']} "
                 f"({total_gb:.1f} GB). Rules fixed before this run (decisions sec. 20); v1 result: see sec. 20.", "",
                 f"Part 1, exactness ({len(rows)} NYU val images, deterministic CUDA, {small}):", "",
                 "| image | A == A2 | B == A | p99 B-A | p99 A2-A |", "|---|---|---|---:|---:|"]
        lines += [f"| {r['scene']}/{r['frame']} | {r['a_equals_a2']} | {r['b_equals_a']} | {r['p99_b_vs_a']:.2e} | {r['p99_a2_vs_a']:.2e} |" for r in rows]
        lines += ["", f"- {exact_rule}: **{'exact' if exact else 'NOT exact'}**",
                  f"- ops without a deterministic CUDA kernel (warnings): {', '.join(sorted(nondet_warnings)) or 'none'}", "",
                  f"Part 2, cost at production settings {prod} ({s.key}):", "",
                  f"- seconds: upstream {tU:.1f}, tonesided {tG:.1f}; peak GB: upstream {mU:.2f}, tonesided {mG:.2f} "
                  f"(card {total_gb:.1f}): **{'fits' if fits else 'does NOT fit'}**",
                  f"- M1 estimate: {M1_CALLS} calls, about {m1_hours:.1f} h", "",
                  f"**Verdict: {'passed, the guided variants may run' if summary['passed'] else 'NOT passed, stop and report'}**", ""]
        (ROOT / "docs" / "marigold_guidance_check.md").write_text("\n".join(lines), encoding="utf-8")
        receipt.update(stage="check_complete", summary=summary, check_sha256=sha256(out / "check.json"))
        print("\n".join(lines))
        if not summary["passed"]:
            raise SystemExit(2)
    except BaseException as e:
        if not isinstance(e, SystemExit):
            receipt.update(stage="failed", error=repr(e))
        raise
    finally:
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
