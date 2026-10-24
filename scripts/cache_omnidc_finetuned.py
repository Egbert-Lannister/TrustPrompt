"""Cache F0/F1 own-training checkpoints and optionally evaluate on development data.

Example (SHA from the training host/transfer record, not an untrusted download):
  python -m scripts.cache_omnidc_finetuned --run-dir results/p1_f0_RUN --variant f0 \
    --checkpoint last.pt --checkpoint-sha256 SHA --dataset nyu --split val \
    --limit 1 --output cache/f0_pilot_nyu --evaluate

Requires receipt.json, config.json, training_frames.json and checkpoints/<name>.
Emits eval_<split>.yaml/commands.txt for the existing fixed-region metric-Z,
no-scale evaluator. Test and formal results are never admitted by this entry.

Comparison caution: old R0 caches may use uint16_0p2mm and saturate >13.107 m;
this cache is float32 with invalid predictions preserved. Audit old per-image
clipped_pixels before interpreting a comparison; no old baseline is rerun here.

Legacy-reader compatibility: CachedPrior only accepts two stage enum values,
both named real_*. For synthetic CPU fixtures, stage=real_train_smoke_verified
is solely a reader compatibility token, NOT a real-data claim. The authoritative
labels are synthetic=true and validation_stage=synthetic_fixture_verified.
The existing evaluator preserves synthetic=true in its rows and qualification.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from baselines.omnidc_finetuned import FineTunedOmniDC, ROLE
from scripts.cache_completion import model_input, select_records, store
from tp.data.schema import load_sample, validate_scene_splits
from tp.data.select import per_group_subsample
from tp.provenance import ROOT, sha256, start_run, write_json

CONFIG = ROOT / "configs/omnidc/finetuned_eval.yaml"
IDENTITY = ("dataset", "sensor", "scene", "frame", "split")
CAUTION = "Old R0 uint16_0p2mm caches may clip >13.107 m; inspect clipped_pixels before comparing. No formal comparison established."


def check_request(splits, role=ROLE):
    if not splits or not set(splits) <= {"train", "val"}:
        raise ValueError("test is locked; only nonempty train/val development splits are allowed")
    if role != ROLE:
        raise ValueError("fine-tuned checkpoint entry is development-only, never formal results")


def evaluation_configs(output, manifest, rows, variant, seed):
    """A selected manifest keeps limit/per-group smoke cohorts identical at evaluation."""
    configs, commands = {}, []
    for split in sorted({r["split"] for r in rows}):
        check_request([split])
        cfg = dict(method="cached_prediction", method_label=f"{variant}_raw", seed=seed,
                   manifest=str(manifest), prediction_cache=str(output), split=split,
                   expected_model=f"OMNI-DC-v1.1-{variant.upper()}", train_scope="all",
                   tau=0.05, band=5, alignment="none", visualizations=3,
                   save_predictions=3, save_alignment_maps=0, experiment_role=ROLE)
        path = output / f"eval_{split}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-m", "tp.eval.run_eval", "--config", str(path),
                   "--output", str(output / f"evaluation_{split}")]
        configs[split] = cfg
        commands.append(dict(argv=command, windows_command=subprocess.list2cmdline(command)))
    write_json(output / "evaluation_commands.json", commands)
    (output / "commands.txt").write_text("\n".join(c["windows_command"] for c in commands) + "\n", encoding="utf-8")
    return configs


def run(run_dir, variant, checkpoint, checkpoint_sha256, dataset, output, *, splits=("val",),
        limit=None, per_group=None, device="cuda", config_path=CONFIG, evaluate=False):
    # This guard intentionally precedes config, manifest, checkpoint and model reads.
    check_request(splits)
    if variant not in {"f0", "f1"} or dataset not in {"nyu", "hammer"}:
        raise ValueError("expected f0/f1 and nyu/hammer")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be positive")
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    check_request(splits, cfg["experiment_role"])
    if cfg.get("storage") != "float32" or cfg.get("alignment") != "none":
        raise ValueError("only unclipped float32/no-alignment output is supported")
    ds = cfg["datasets"][dataset]
    manifest_path = ROOT / ds["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_scene_splits(manifest["records"])
    records = per_group_subsample(select_records(manifest["records"], set(splits), ds["train_scope"]), per_group)
    records = records[:limit] if limit is not None else records
    if not records:
        raise ValueError("empty requested development cohort")
    output = (ROOT / output).resolve()
    config = dict(cfg, run_dir=str(Path(run_dir).resolve()), variant=variant, checkpoint=checkpoint,
                  checkpoint_sha256=checkpoint_sha256, dataset=dataset, splits=list(splits),
                  limit=limit, per_group=per_group, device=device, config_file_sha256=sha256(config_path))
    receipt = start_run(output, config, [sys.executable, "-m", "scripts.cache_omnidc_finetuned", *sys.argv[1:]], int(cfg["seed"]))
    receipt.update(synthetic=manifest["synthetic"], formal_experiment_complete=False, comparison_caution=CAUTION,
                   manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path))
    log = (output / "run.log").open("w", encoding="utf-8")
    eval_configs = None
    try:
        import torch
        torch.manual_seed(int(cfg["seed"])); np.random.seed(int(cfg["seed"]))
        model = FineTunedOmniDC(run_dir, checkpoint, checkpoint_sha256, variant, device=device)
        receipt["model_version"] = model.model_version
        receipt["runtime_device"] = dict(device=device, torch=torch.__version__, cuda=torch.version.cuda)
        write_json(output / "receipt.json", receipt)
        (output / "predictions").mkdir()
        rows, selected = [], []
        use_cuda = str(device).startswith("cuda")
        for i, record in enumerate(records):
            if record["split"] not in splits or record["split"] == "test":
                raise ValueError("selected sample split is not authorized")
            path = (manifest_path.parent / record["path"]).resolve()
            sample_hash = sha256(path)
            if sample_hash != record["sha256"]:
                raise ValueError("sample checksum mismatch")
            s = load_sample(path)
            if any(getattr(s, k) != record[k] for k in IDENTITY if k != "sensor"):
                raise ValueError("sample identity differs from manifest")
            if "sensor" in record and s.sensor != record["sensor"]:
                raise ValueError("sample sensor differs from manifest")
            if s.metadata["synthetic"] != manifest["synthetic"]:
                raise ValueError("real/synthetic labeling mismatch")
            depth_in, guide, pinfo = model_input(s, "raw", tau=0.05)
            assert guide is None
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            if pinfo["kept"] < model.min_points:
                depth, report = np.full(depth_in.shape, np.nan, np.float32), dict(status="too_few_prompt_points")
            else:
                depth, report = model.predict(s.rgb, depth_in, s.K_rgb)
            if use_cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            depth = np.asarray(depth)
            if depth.dtype != np.float32 or depth.shape != s.rgb.shape[:2]:
                raise ValueError("model must return float32 depth on the RGB grid")
            prediction = output / "predictions" / f"{i:06d}.npz"
            store(prediction, depth, "float32")
            rows.append(dict(**{k: getattr(s, k) for k in IDENTITY}, sample_npz_sha256=sample_hash,
                             rgb_sha256=hashlib.sha256(s.rgb.tobytes()).hexdigest(),
                             prompt_in_sha256=hashlib.sha256(depth_in.tobytes()).hexdigest(),
                             prediction_path=str(prediction), prediction_sha256=sha256(prediction),
                             seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated() if use_cuda else None,
                             prompt=pinfo, report=report, storage="float32", clipped_pixels=0,
                             invalid_pixels=int((~(np.isfinite(depth) & (depth > 0))).sum())))
            selected.append(dict(record, path=str(path), sensor=s.sensor))
            line = f"{i + 1}/{len(records)} {s.key} seconds={seconds:.3f} invalid={rows[-1]['invalid_pixels']}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            write_json(output / "images.json", rows)
        selected_path = output / "selected_manifest.json"
        write_json(selected_path, dict(schema=manifest["schema"], synthetic=manifest["synthetic"], records=selected,
                                       source_manifest_sha256=sha256(manifest_path)))
        eval_configs = evaluation_configs(output, selected_path, rows, variant, int(cfg["seed"]))
        receipt.update(stage="real_train_smoke_verified" if manifest["synthetic"] else "real_dev_verified",
                       stage_semantics="legacy_reader_compatibility_only; not a real/synthetic data classification",
                       validation_stage="synthetic_fixture_verified" if manifest["synthetic"] else "development_cache_verified",
                       cache_qualification="synthetic engineering fixture" if manifest["synthetic"] else "development only",
                       completed_at_utc=datetime.now(timezone.utc).isoformat(), images_sha256=sha256(output / "images.json"),
                       selected_manifest_sha256=sha256(selected_path), frame_count=len(rows),
                       inputs=[{k: r[k] for k in (*IDENTITY, "rgb_sha256", "sample_npz_sha256")} for r in rows])
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); traceback.print_exc(file=log); raise
    finally:
        log.close(); write_json(output / "receipt.json", receipt)
    if evaluate:
        from tp.eval.run_eval import evaluate_manifest
        for split, eval_cfg in eval_configs.items():
            evaluate_manifest(eval_cfg, output / f"evaluation_{split}",
                              [sys.executable, "-m", "tp.eval.run_eval", "--config", str(output / f"eval_{split}.yaml"),
                               "--output", str(output / f"evaluation_{split}")])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=["f0", "f1"])
    parser.add_argument("--checkpoint", default="last.pt", choices=["last.pt", "final.pt"])
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset", required=True, choices=["nyu", "hammer"])
    parser.add_argument("--split", action="append", choices=["train", "val"], dest="splits")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-group", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--evaluate", action="store_true")
    a = parser.parse_args()
    run(a.run_dir, a.variant, a.checkpoint, a.checkpoint_sha256, a.dataset, a.output,
        splits=a.splits or ["val"], limit=a.limit, per_group=a.per_group, device=a.device,
        config_path=a.config, evaluate=a.evaluate)


if __name__ == "__main__":
    main()
