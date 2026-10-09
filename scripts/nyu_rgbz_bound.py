"""Measure the NYU registered-Z vs RGB-axis-Z gap (plan revision R3).

Usage (toolbox ZIP downloaded once from the official NYU page):
  python -m scripts.nyu_rgbz_bound --toolbox data/source_cache/nyu_toolbox/toolbox_nyu_depth_v2.zip \
      --manifest data/processed/mirror3d_smoke_v1/manifest.json --output results/nyu_rgbz_bound_v1

Reads only prompt/validity/K of non-test samples. Decision rule fixed in
docs/plan_revision_v2.md: worst-case relative gap <= 0.5% -> rgb_z_bounded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tp.data.nyu_rgbz import extrinsics, read_toolbox, relative_gap
from tp.data.schema import load_sample
from tp.provenance import ROOT, sha256, start_run, write_json

THRESHOLD = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolbox", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config = dict(toolbox=str(args.toolbox), manifest=str(args.manifest), threshold=THRESHOLD,
                  splits=["train", "val"], rule="worst pixel-corner ray, valid prompt pixels")
    receipt = start_run(args.output, config, [sys.executable, "-m", "scripts.nyu_rgbz_bound", *sys.argv[1:]], 0)
    texts = read_toolbox(args.toolbox)
    ext = extrinsics(texts)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames, pooled = [], []
    for r in manifest["records"]:
        if r["split"] not in config["splits"]:
            continue                                    # test samples are never opened
        path = (args.manifest.parent / r["path"]).resolve()
        if sha256(path) != r["sha256"]:
            raise ValueError("sample checksum mismatch")
        s = load_sample(path)
        results = {}
        for name, K in (("registered_K", s.K_prompt),
                        ("physical_rgb_K_plus1px", s.K_prompt + np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]]))):
            gap = relative_gap(s.prompt, s.prompt_valid, K, ext["R"], ext["T"])
            g = gap[s.prompt_valid]
            results[name] = dict(max=float(g.max()), p999=float(np.quantile(g, .999)),
                                 p99=float(np.quantile(g, .99)), median=float(np.median(g)), count=int(g.size))
            if name == "registered_K":
                pooled.append(g)
        frames.append(dict(scene=s.scene, frame=s.frame, split=s.split, sample_sha256=r["sha256"], **results))
    allg = np.concatenate(pooled)
    worst = max(max(f[k]["max"] for k in ("registered_K", "physical_rgb_K_plus1px")) for f in frames)
    report = dict(extrinsics={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ext.items()},
                  toolbox_sha256=sha256(args.toolbox), frames=frames,
                  pooled_registered_K=dict(max=float(allg.max()), p999=float(np.quantile(allg, .999)),
                                           median=float(np.median(allg)), count=int(allg.size)),
                  worst_over_both_K=worst, threshold=THRESHOLD,
                  decision="rgb_z_bounded" if worst <= THRESHOLD else "rewrite_to_rgb_z_required")
    write_json(args.output / "rgbz_bound.json", report)
    receipt.update(stage="completed", decision=report["decision"], worst=worst)
    write_json(args.output / "receipt.json", receipt)
    print(json.dumps(dict(decision=report["decision"], worst=worst, pooled=report["pooled_registered_K"]), indent=1))


if __name__ == "__main__":
    main()
