"""Frozen encoder features for T v0 (GPU, Windows): one file per unique RGB image.

  python -m scripts.cache_encoder_features --config configs/promptda/hammer_vits_raw.yaml \
      --output cache/encoder_vits_hammer_dev_v1 [--limit 3]

Encoder = PromptDA-S checkpoint's ViT-S/14 (DINOv2-S as fine-tuned by Depth Anything V2; plan
deviation recorded in docs/decisions.md sec. 12). Train/val only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import yaml

from baselines.promptda import PromptDAPredictor, encoder_tokens, prepare_prompt
from scripts.cache_promptda import git_source
from tp.data.schema import load_sample
from tp.provenance import ROOT, sha256, start_run, write_json


def run(config_path, output, limit=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
    splits = set(config.get("splits", ["train", "val"]))
    if "test" in splits:
        raise ValueError("test samples are locked until method freeze")
    processed = Path(config["processed"]); processed = processed if processed.is_absolute() else ROOT / processed
    records = [r for r in json.loads((processed / "manifest.json").read_text(encoding="utf8"))["records"] if r["split"] in splits]
    records = records[:limit] if limit else records
    out = Path(output); out = out if out.is_absolute() else ROOT / out
    receipt = start_run(out, dict(config, limit=limit, purpose="T v0 encoder features"),
                        [sys.executable, "-m", "scripts.cache_encoder_features", *sys.argv[1:]], int(config.get("seed", 0)))
    log = (out / "run.log").open("w", encoding="utf8")
    try:
        import torch
        source, ckpt = Path(config["source"]), Path(config["ckpt"])
        receipt["model_version"] = dict(name=config["model"] + " encoder", encoder=config["encoder"], checkpoint=str(ckpt),
                                        checkpoint_sha256=sha256(ckpt), source_commit=git_source(source, "rev-parse", "HEAD"))
        write_json(out / "receipt.json", receipt)
        torch.manual_seed(int(config.get("seed", 0)))
        pred = PromptDAPredictor(source, ckpt, config["encoder"], device=config.get("device", "cuda"))
        (out / "features").mkdir()
        index = {}
        for k, r in enumerate(records):
            path = (processed / r["path"]).resolve()
            if sha256(path) != r["sha256"]:
                raise ValueError("sample checksum mismatch")
            s = load_sample(path)
            h = hashlib.sha256(s.rgb.tobytes()).hexdigest()
            if h in index:
                index[h]["samples"].append(s.key); continue
            prompt_hw, _ = prepare_prompt(s.prompt, s.prompt_valid, variant="raw")
            feat, info = encoder_tokens(pred, s.rgb, prompt_hw)
            f = out / "features" / f"{h}.npy"
            np.save(f, feat, allow_pickle=False)
            index[h] = dict(path=f"features/{f.name}", sha256=sha256(f), samples=[s.key], info=info)
            line = f"{k + 1}/{len(records)} {s.key} tokens={info['token_grid']} hook_calls={info['hook_calls']}"
            print(line, flush=True); log.write(line + "\n")
        write_json(out / "index.json", index)
        receipt.update(stage="features_verified", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                       unique_rgb=len(index), index_sha256=sha256(out / "index.json"), load_mode=pred.load_mode)
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); traceback.print_exc(file=log); raise
    finally:
        log.close(); write_json(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int); a = ap.parse_args(); run(a.config, a.output, a.limit)
