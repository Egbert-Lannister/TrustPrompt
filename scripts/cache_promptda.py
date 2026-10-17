"""Run PromptDA on a processed manifest and cache verified predictions (GPU, Windows).

  python -m scripts.cache_promptda --config configs/promptda/nyu_vitl_raw.yaml [--limit 3] [--output DIR]

Only train/val samples are admitted. The model receives RGB and the prepared
192x256 prompt only. Variants focus_prior / oracle_prior additionally read the
GT-derived focus or contamination mask to build the prompt; they are oracle
analyses and are labelled so in every record. Variant detector_prior removes the
pixels flagged by the DEPTHOR++-style detector (no GT; deployable baseline). D_g for those variants comes from
the verified UniDepth cache + global robust alignment (same code/seed as the
RobustAlign baseline). Outputs are immutable; the run records the checkpoint
SHA256, the source commit and hashes, and per-sample input/output hashes.
"""
import os
# Anaconda numpy uses MKL; its threaded layer loads Intel OpenMP (libiomp5md) on the first LAPACK call
# (np.linalg in module C), next to the copy torch already loaded -> OMP Error #15. The sequential MKL
# layer loads no OpenMP runtime. Must be set before numpy initialises MKL. (KMP_DUPLICATE_LIB_OK is not used.)
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
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

from baselines.promptda import DEPLOYABLE, VARIANTS, PromptDAPredictor, prepare_prompt
from tp.data.schema import load_sample
from tp.eval.regions import regions
from tp.eval.run_eval import focus_scope
from tp.provenance import ROOT, sha256, start_run, write_json


def git_source(source, *args):
    # The checkout may live on a drive without ownership records (exFAT); trust it for this call only.
    safe = Path(source).resolve().as_posix()
    p = subprocess.run(["git", "-c", f"safe.directory={safe}", "-C", str(source), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def run(config_path, limit=None, output=None, overrides=()):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
    for item in overrides:                      # --set KEY=VALUE (YAML value), recorded in config.json
        key, _, value = item.partition("=")
        config[key] = yaml.safe_load(value)
    if config["variant"] not in VARIANTS:
        raise ValueError("unknown variant")
    allowed = set(config.get("splits", ["train", "val"]))
    if "test" in allowed:
        raise ValueError("test samples are locked until method freeze")
    processed = Path(config["processed"]); processed = processed if processed.is_absolute() else ROOT / processed
    records = [r for r in json.loads((processed / "manifest.json").read_text(encoding="utf8"))["records"] if r["split"] in allowed]
    records = records[:limit] if limit else records
    output = Path(output or config["output"]); output = output if output.is_absolute() else ROOT / output
    receipt = start_run(output, dict(config, limit=limit), [sys.executable, "-m", "scripts.cache_promptda", *sys.argv[1:]],
                        int(config.get("seed", 0)))
    log = (output / "run.log").open("w", encoding="utf8")
    try:
        import torch
        source, ckpt = Path(config["source"]), Path(config["ckpt"])
        receipt["model_version"] = dict(
            name=config["model"], encoder=config["encoder"], variant=config["variant"],
            oracle=config["variant"] not in DEPLOYABLE or config.get("mask_source") == "oracle_c", checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt),
            model_card=config.get("model_card"), source=str(source), source_commit=git_source(source, "rev-parse", "HEAD"),
            source_status=git_source(source, "status", "--porcelain=v1"),
            source_hashes={str(p.relative_to(source)): sha256(p) for p in sorted((source / "promptda").rglob("*.py"))},
            recipe="official: RGB/255, longer side<=1008 INTER_AREA, multiples of 14; prompt 192x256 m",
            deviations="non-multiple-of-14 images resized to nearest multiple (bilinear); output resized back bilinear; "
                       "prompt holes nearest-filled on native grid before nearest resize")
        receipt["gpu"] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
        write_json(output / "receipt.json", receipt)
        torch.manual_seed(int(config.get("seed", 0)))
        predictor = PromptDAPredictor(source, ckpt, config["encoder"], device=config.get("device", "cuda"))
        receipt["model_version"]["load_mode"] = predictor.load_mode
        prior = None
        if config["variant"] != "raw":
            from baselines.cached_prior import CachedPrior
            from tp.align.robust import align_depth
            pc = Path(config["prior_cache"]); prior = CachedPrior(pc if pc.is_absolute() else ROOT / pc)
            receipt["prior_cache_receipt_sha256"] = sha256(prior.root / "receipt.json")
        (output / "predictions").mkdir()
        rows = []; tcache = None
        for k, r in enumerate(records):
            path = (processed / r["path"]).resolve()
            sample_hash = sha256(path)
            if sample_hash != r["sha256"]:
                raise ValueError("sample checksum mismatch")
            s = load_sample(path)
            remove = fill = det_info = None
            if config["variant"] != "raw":
                d = align_depth(prior.predict(s, sample_hash), s.prompt, s.prompt_valid, seed=int(config["seed"]),
                                compute_local=False, **config.get("robust", {}))
                fill = d["global_depth"]
                if config["variant"] == "focus_prior":
                    remove = focus_scope(s)[1]
                elif config["variant"] == "gate_prior":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        remove = s.prompt_valid & np.isfinite(fill) & (np.abs(s.prompt - fill) / fill > float(config["tau_gate"]))
                    det_info = dict(tau_gate=float(config["tau_gate"]), flagged=int(remove.sum()))
                elif config["variant"] == "trust_prior":
                    if tcache is None:
                        from tp.trust.cache import TrustCache
                        tcache = TrustCache(config["trust_cache"])
                        receipt["trust_cache_receipt_sha256"] = sha256(tcache.root / "receipt.json")
                    tt = config.get("trust_threshold")
                    thr = tcache.threshold_for(s) if tt == "per_item" else float(tt if tt is not None else tcache.threshold)
                    remove = s.prompt_valid & (tcache.trust(s, sample_hash, s.prompt.shape) < thr)
                    det_info = dict(threshold=thr, flagged=int(remove.sum()))
                elif config["variant"] == "repair_prior":
                    from tp.models.fill import fill_regions
                    if config["mask_source"] == "oracle_c":
                        remove = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=float(config["tau"]), band=5)["C"]
                    elif config["mask_source"] == "trust":
                        if tcache is None:
                            from tp.trust.cache import TrustCache
                            tcache = TrustCache(config["trust_cache"])
                            receipt["trust_cache_receipt_sha256"] = sha256(tcache.root / "receipt.json")
                        thr = float(config.get("trust_threshold") or tcache.threshold)
                        remove = s.prompt_valid & (tcache.trust(s, sample_hash, s.prompt.shape) < thr)
                    else:
                        raise ValueError("mask_source must be oracle_c or trust")
                    remove = remove & s.prompt_valid
                    pd = prior.predict(s, sample_hash)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pinv = np.where(np.isfinite(pd) & (pd > 0), 1.0 / pd, np.nan)
                    filled, recs = fill_regions(fill, pinv, s.prompt, s.prompt_valid & ~remove, remove,
                                                rule=config.get("fill_rule", "v1"), seed=int(config["seed"]), **config.get("fill", {}))
                    fill = filled
                    det_info = dict(mask_source=config["mask_source"], flagged=int(remove.sum()), regions=len(recs),
                                    choice_area={k: sum(x["area"] for x in recs if x["choice"] == k) for k in ("k0", "k1", "k2", "k3")})
                elif config["variant"] == "detector_prior":
                    from tp.geom import resize_nearest
                    from tp.models.depthor_det import detect_for_sample
                    an, _, det_info = detect_for_sample(s, prior.predict(s, sample_hash), config.get("detector", {}))
                    remove = an if an.shape == s.prompt.shape else resize_nearest(an, s.prompt.shape)
                else:
                    remove = regions(s.prompt, s.prompt_valid, s.gt, s.gt_valid, tau=float(config["tau"]), band=5)["C"]
            prompt_hw, pinfo = prepare_prompt(s.prompt, s.prompt_valid, variant=config["variant"], remove=remove, fill_depth=fill)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
            depth, report = predictor.predict(s.rgb, prompt_hw)
            torch.cuda.synchronize(); seconds = time.perf_counter() - t
            name = f"{s.sensor}__{s.scene}__{s.frame}.npz"
            with (output / "predictions" / name).open("xb") as f:
                if config.get("storage", "float32") == "uint16_0p2mm":
                    # Fixed-point metres, step 0.2 mm (|error| <= 0.1 mm), 0 = invalid, max 13.107 m.
                    finite = np.isfinite(depth) & (depth > 0)
                    q = np.zeros(depth.shape, np.uint16)
                    q[finite] = np.clip(np.round(depth[finite] / 2e-4), 1, 65535).astype(np.uint16)
                    clipped = int((finite & (depth > 65535 * 2e-4)).sum())
                    np.savez_compressed(f, depth_q=q, scale_m=np.float64(2e-4))
                else:
                    clipped = 0
                    np.savez_compressed(f, depth=depth)
            rows.append(dict(dataset=s.dataset, sensor=s.sensor, scene=s.scene, frame=s.frame, split=s.split,
                             sample_npz_sha256=sample_hash, rgb_sha256=hashlib.sha256(s.rgb.tobytes()).hexdigest(),
                             prompt_hw_sha256=hashlib.sha256(prompt_hw.tobytes()).hexdigest(),
                             prediction_path=str(output / "predictions" / name),
                             prediction_sha256=sha256(output / "predictions" / name), seconds=seconds,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(), prompt=pinfo, report=report,
                             invalid_pixels=int((~(np.isfinite(depth) & (depth > 0))).sum()),
                             storage=config.get("storage", "float32"), clipped_pixels=clipped, detector=det_info))
            line = f"{k + 1}/{len(records)} {s.key} {seconds:.3f}s invalid={rows[-1]['invalid_pixels']}"
            print(line, flush=True); log.write(line + "\n")
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
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--output")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(); run(a.config, a.limit, a.output, a.set)
