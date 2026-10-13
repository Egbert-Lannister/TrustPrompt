"""Build T v0 arrays (CPU, no torch): real prompts of the train/val splits plus simulated
variants of the train split, on the 192x256 prompt grid.

  python -m scripts.build_trust_data --eval-config configs/hammer_dev_eval.yaml \
      --output cache/trust_data_v0_hammer --sim-variants 4 [--shard 0 --shards 4]

Each shard is an immutable run directory OUTPUT/shard_IofN with items/*.npz, index.json and a
receipt (stage trust_data_verified). Test samples are never read. D_g = global robust alignment of
the cached UniDepth prior to the (real or simulated) prompt, same code and seed as the evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zlib

import numpy as np
import yaml

from baselines.cached_prior import CachedPrior
from tp.align.robust import align_depth
from tp.data.schema import load_sample
from tp.geom import resize_nearest
from tp.provenance import ROOT, sha256, start_run, write_json
from tp.sim.generate import simulate
from tp.eval.run_eval import focus_scope
from tp.trust.data import labels_from, sim_labels, to_grid


def run(eval_config, output, sim_variants=4, shard=0, shards=1, splits=("train", "val"), limit=None, version="v0"):
    cfg = yaml.safe_load(Path(eval_config).read_text(encoding="utf-8"))
    if "test" in splits:
        raise ValueError("test samples are locked until method freeze")
    out = Path(output); out = out if out.is_absolute() else ROOT / out
    run_dir = out / f"shard_{shard}of{shards}"
    config = dict(eval_config=str(eval_config), eval_config_content=cfg, sim_variants=sim_variants, shard=shard,
                  shards=shards, splits=list(splits), limit=limit, grid=[192, 256], version=version,
                  label_rule="0 clean |P-GT|/GT<=tau, 1 contaminated, 255 invalid prompt or GT", tau=cfg["tau"])
    receipt = start_run(run_dir, config, [sys.executable, "-m", "scripts.build_trust_data", *sys.argv[1:]], int(cfg["seed"]))
    log = (run_dir / "run.log").open("w", encoding="utf-8")
    try:
        manifest = Path(cfg["manifest"]); manifest = manifest if manifest.is_absolute() else ROOT / manifest
        records = [r for r in json.loads(manifest.read_text(encoding="utf-8"))["records"] if r["split"] in splits]
        records = [r for i, r in enumerate(records) if i % shards == shard][:limit]
        pc = Path(cfg["prior_cache"]); prior = CachedPrior(pc if pc.is_absolute() else ROOT / pc)
        receipt.update(manifest_sha256=sha256(manifest), prior_cache_receipt_sha256=sha256(prior.root / "receipt.json"),
                       model_version=prior.receipt["model_version"])
        (run_dir / "items").mkdir()
        index = []
        for n, r in enumerate(records):
            path = (manifest.parent / r["path"]).resolve()
            sample_hash = sha256(path)
            if sample_hash != r["sha256"]:
                raise ValueError("sample checksum mismatch")
            s = load_sample(path)
            if s.split not in splits:
                raise ValueError("split mismatch")
            prior_depth = prior.predict(s, sample_hash)
            rgb_hash = hashlib.sha256(s.rgb.tobytes()).hexdigest()
            key = f"{s.sensor}__{s.scene}__{s.frame}"
            variants = [("real", None)]
            if s.split == "train":
                variants += [(f"sim{k}", k) for k in range(sim_variants)]
            for name, k in variants:
                focus = focus_scope(s)[1]
                if k is None:
                    p = resize_nearest(s.prompt, s.gt.shape); pv = resize_nearest(s.prompt_valid, s.gt.shape)
                    lab = labels_from(p, pv, s.gt, s.gt_valid, tau=cfg["tau"]); sim_info = None
                else:
                    rng = np.random.default_rng([int(cfg["seed"]), zlib.crc32(key.encode("utf-8")), k])
                    o = simulate(s, rng, tau=cfg["tau"])
                    p, pv, lab = o["P_corrupt"], o["V_corrupt"], sim_labels(o["labels"])
                    focus = focus | (o["applied"] > 0)          # simulated contamination is in scope too
                    sim_info = dict(corrupted=o["info"]["corrupted"],
                                    regions=[dict(type=x.get("type"), status=x.get("status", "ok"),
                                                  area_frac=x.get("area_frac")) for x in o["info"]["regions"]])
                dg = align_depth(prior_depth, p, pv, seed=int(cfg["seed"]), compute_local=False,
                                 **cfg.get("robust", {}))["global_depth"]
                arrays = to_grid(p, pv, dg, s.rgb, lab) if version == "v0" else \
                    to_grid(p, pv, dg, s.rgb, lab, gt=s.gt, gt_valid=s.gt_valid, focus=focus)
                item = run_dir / "items" / f"{key}__{name}.npz"
                with item.open("xb") as f:
                    np.savez_compressed(f, **arrays)
                lab_lr = arrays["label"]
                index.append(dict(dataset=s.dataset, split=s.split, scene=s.scene, frame=s.frame, sensor=s.sensor,
                                  variant=name, rgb_sha256=rgb_hash, sample_sha256=sample_hash,
                                  path=f"items/{item.name}", sha256=sha256(item), sim=sim_info,
                                  clean=int((lab_lr == 0).sum()), contaminated=int((lab_lr == 1).sum())))
            line = f"{n + 1}/{len(records)} {s.dataset}/{key} variants={len(variants)}"
            print(line, flush=True); log.write(line + "\n")
        write_json(run_dir / "index.json", index)
        receipt.update(stage="trust_data_verified", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       items=len(index), index_sha256=sha256(run_dir / "index.json"))
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); traceback.print_exc(file=log); raise
    finally:
        log.close(); write_json(run_dir / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--sim-variants", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int); ap.add_argument("--version", choices=["v0", "v1"], default="v0")
    a = ap.parse_args()
    run(a.eval_config, a.output, a.sim_variants, a.shard, a.shards, limit=a.limit, version=a.version)
