"""Run a depth-completion baseline on a processed manifest and cache verified predictions (GPU, Windows).

  python -m scripts.cache_completion --model priorda --dataset hammer --variant raw [--limit 3] [--output DIR]

Models: baselines/completion.py (pinned sources and weights, configs/baselines/registry.json).
Variants (configs/baselines/cache.yaml):
  raw          sensor prompt as is (holes = 0), deployable
  tmask        prompt pixels with T v1 trust < the conformal threshold (alpha 0.1, method v1) set to 0; the baseline
               completes them itself. Deployable: no GT is read.
  oracle_mask  every contaminated prompt pixel C (|P - GT| / GT > tau) set to 0 [oracle; reads GT]
Populations: val and the train-split holdout scenes only (T is out of fit on both); test is never admitted.
The cache layout and receipt match scripts/cache_promptda.py, so tp.eval.run_eval --method cached_prediction
evaluates it (expected_model = the registry label, train_scope = holdout).
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")      # see scripts/cache_promptda.py (OMP Error #15)
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from baselines.completion import build, registry, source_dir, verify_weights, weight_files
from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.provenance import ROOT, sha256, start_run, write_json

VARIANTS = ("raw", "tmask", "oracle_mask")
CONFIG = ROOT / "configs" / "baselines" / "cache.yaml"


def git_source(source, *args):
    safe = Path(source).resolve().as_posix()
    p = subprocess.run(["git", "-c", f"safe.directory={safe}", "-C", str(source), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def holdout(dataset, scene):
    from tp.trust.data import scene_inner_holdout
    return scene_inner_holdout(dataset, scene)


def select_records(records, splits, train_scope):
    if "test" in splits:
        raise ValueError("test samples are locked until method freeze")
    out = [r for r in records if r["split"] in splits]
    if train_scope == "holdout":
        out = [r for r in out if r["split"] != "train" or holdout(r["dataset"], r["scene"])]
    elif train_scope != "all":
        raise ValueError("train_scope must be all or holdout")
    return out


def crc_threshold(cfg, crc_dataset):
    p = ROOT / cfg["crc"]["file"]
    d = json.loads(p.read_text(encoding="utf-8"))
    lam = d["datasets"][crc_dataset]["alphas"][str(cfg["crc"]["alpha"])]["lambda_hat"]
    if lam is None:
        raise ValueError(f"no conformal threshold for {crc_dataset} at alpha {cfg['crc']['alpha']}")
    return float(lam), sha256(p)


def model_input(s, variant, *, tau, trust=None, threshold=None):
    """Depth handed to the baseline on the RGB grid (float32 m, 0 = missing) and the removal record."""
    if s.prompt.shape != s.rgb.shape[:2]:
        raise ValueError("baselines need the prompt on the RGB grid (true for the NYU and HAMMER dev sets)")
    valid = s.prompt_valid.copy()
    info = dict(variant=variant, native_valid=int(valid.sum()))
    if variant == "tmask":
        remove = valid & (trust < threshold)
        info.update(threshold=float(threshold))
    elif variant == "oracle_mask":
        remove = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=tau, band=5)["C"] & valid
    elif variant == "raw":
        remove = np.zeros_like(valid)
    else:
        raise ValueError(f"unknown variant {variant}")
    keep = valid & ~remove
    depth = np.where(keep, s.prompt, 0.0).astype(np.float32)
    info.update(removed=int(remove.sum()), kept=int(keep.sum()))
    return depth, info


def store(path, depth, storage):
    with path.open("xb") as f:
        if storage == "uint16_0p2mm":
            finite = np.isfinite(depth) & (depth > 0)
            q = np.zeros(depth.shape, np.uint16)
            q[finite] = np.clip(np.round(depth[finite] / 2e-4), 1, 65535).astype(np.uint16)
            np.savez_compressed(f, depth_q=q, scale_m=np.float64(2e-4))
            return int((finite & (depth > 65535 * 2e-4)).sum())
        np.savez_compressed(f, depth=depth)
        return 0


def run(model_key, dataset, variant, limit=None, output=None, device="cuda", trust_cache=None, train_scope=None, tag=""):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    reg = registry()["models"][model_key]
    ds = cfg["datasets"][dataset]
    processed = ROOT / ds["processed"]
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf8"))
    train_scope = train_scope or cfg["train_scope"]
    records = select_records(manifest["records"], set(cfg["splits"]), train_scope)
    if model_key == "cdm":           # one camera model on the GPU at a time
        records = sorted(records, key=lambda r: r.get("sensor", ""))
    records = records[:limit] if limit else records
    output = Path(output or cfg["output"].format(model=model_key, variant=variant + tag, dataset=dataset))
    output = output if output.is_absolute() else ROOT / output
    config = dict(cfg, model=model_key, dataset=dataset, variant=variant, limit=limit, device=device,
                  train_scope=train_scope, trust_cache_override=trust_cache, tag=tag,
                  oracle=variant == "oracle_mask", registry_entry=reg)
    receipt = start_run(output, config, [sys.executable, "-m", "scripts.cache_completion", *sys.argv[1:]],
                        int(cfg["seed"]))
    log = (output / "run.log").open("w", encoding="utf8")
    try:
        import torch
        torch.manual_seed(int(cfg["seed"])); np.random.seed(int(cfg["seed"]))
        src = source_dir(model_key)
        receipt["model_version"] = dict(
            name=reg["label"], key=model_key, variant=variant, oracle=variant == "oracle_mask",
            repo=reg["repo"], pinned_commit=reg["commit"], source=str(src),
            source_commit=git_source(src, "rev-parse", "HEAD"), source_status=git_source(src, "status", "--porcelain=v1"),
            source_hashes={p.relative_to(src).as_posix(): sha256(p) for p in sorted(src.rglob("*.py"))
                           if ".git" not in p.parts and "__pycache__" not in p.parts},
            code_license=reg["code_license"], weights_license=reg["weights_license"], settings=reg.get("settings"))
        if receipt["model_version"]["source_commit"] != reg["commit"]:
            raise ValueError(f"{src} is at {receipt['model_version']['source_commit']}, registry pins {reg['commit']}")
        receipt["model_version"]["weights"] = verify_weights(model_key)
        receipt["gpu"] = dict(name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                              torch=torch.__version__, cuda=torch.version.cuda)
        write_json(output / "receipt.json", receipt)
        model = build(model_key, device=device)
        receipt["model_version"].update(deviations=model.deviations, load_info=model.load_info)
        tcache = thr = None
        if variant == "tmask":
            from tp.trust.cache import TrustCache
            tcache = TrustCache(trust_cache or ds["trust_cache"])
            if tcache.per_item:        # cross-fitted T: each scene's map and conformal threshold come from its own fold
                thr, crc_hash = "per_item", None
            else:
                thr, crc_hash = crc_threshold(cfg, ds["crc_dataset"])
            receipt.update(trust_cache=str(tcache.root), trust_cache_receipt_sha256=sha256(tcache.root / "receipt.json"),
                           trust_threshold=thr, crc_file_sha256=crc_hash)
        write_json(output / "receipt.json", receipt)
        (output / "predictions").mkdir()
        rows = []
        for k, r in enumerate(records):
            path = (processed / r["path"]).resolve()
            sample_hash = sha256(path)
            if sample_hash != r["sha256"]:
                raise ValueError("sample checksum mismatch")
            s = load_sample(path)
            trust = tcache.trust(s, sample_hash, s.prompt.shape) if tcache is not None else None
            depth_in, pinfo = model_input(s, variant, tau=float(cfg["tau"]), trust=trust,
                                          threshold=tcache.threshold_for(s) if thr == "per_item" else thr)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t = time.perf_counter()
            if pinfo["kept"] < model.min_points:
                depth, report = np.full(depth_in.shape, np.nan, np.float32), dict(status="too_few_prompt_points")
            else:
                kw = dict(sensor=s.sensor) if model_key == "cdm" else {}
                depth, report = model.predict(s.rgb, depth_in, s.K_rgb, **kw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t
            name = f"{s.sensor}__{s.scene}__{s.frame}.npz"
            clipped = store(output / "predictions" / name, depth, cfg["storage"])
            rows.append(dict(dataset=s.dataset, sensor=s.sensor, scene=s.scene, frame=s.frame, split=s.split,
                             sample_npz_sha256=sample_hash, rgb_sha256=hashlib.sha256(s.rgb.tobytes()).hexdigest(),
                             prompt_in_sha256=hashlib.sha256(depth_in.tobytes()).hexdigest(),
                             prediction_path=str(output / "predictions" / name),
                             prediction_sha256=sha256(output / "predictions" / name), seconds=seconds,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                             prompt=pinfo, report=report, storage=cfg["storage"], clipped_pixels=clipped,
                             invalid_pixels=int((~(np.isfinite(depth) & (depth > 0))).sum())))
            line = f"{k + 1}/{len(records)} {s.key} {seconds:.3f}s removed={pinfo['removed']} invalid={rows[-1]['invalid_pixels']}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            write_json(output / "images.json", rows)
        receipt.update(stage="real_dev_verified", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       images_sha256=sha256(output / "images.json"), frame_count=len(rows),
                       inputs=[{k: r[k] for k in ("dataset", "sensor", "scene", "frame", "split", "rgb_sha256",
                                                  "sample_npz_sha256")} for r in rows])
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); traceback.print_exc(file=log); raise
    finally:
        log.close(); write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(registry()["models"]))
    ap.add_argument("--dataset", required=True, choices=["nyu", "hammer"])
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--limit", type=int); ap.add_argument("--output"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust-cache", default=None, help="tmask: T cache to use instead of configs/baselines/cache.yaml "
                    "(a cross-fitted cache carries per-item thresholds)")
    ap.add_argument("--train-scope", choices=["holdout", "all"], default=None)
    ap.add_argument("--tag", default="", help="appended to the variant in the default output name")
    a = ap.parse_args()
    run(a.model, a.dataset, a.variant, a.limit, a.output, a.device, a.trust_cache, a.train_scope, a.tag)
