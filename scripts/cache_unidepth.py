"""Cache 1-frame then 5-frame RGB-only UniDepth priors without training."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import subprocess
import time
import traceback
import numpy as np
import yaml
from tp.provenance import start_run, sha256, write_json
from baselines.unidepth import UniDepthPredictor


def load_rgb_training_npz(path: Path, allowed_splits=frozenset({"train"})):
    """Decode only metadata and RGB members; GT is never decoded or passed onward."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("split") not in allowed_splits:
            raise ValueError(f"This cache admits only training/development splits {sorted(allowed_splits)}; test refused")
        if metadata.get("metadata", {}).get("synthetic") is not False:
            raise ValueError("This smoke cache requires real RGB")
        rgb = archive["rgb"].copy()
    return rgb, {k: metadata[k] for k in ("dataset", "sensor", "scene", "frame", "split")}


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text); stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def git_source(source, *args):
    p = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, encoding="utf8")
    return p.stdout.strip() if p.returncode == 0 else None


def run(config_path, limit=None, output=None):
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    # Plan revision v2: split "dev" = train+val development set; test is never admitted.
    if config.get("split") not in {"train", "dev"} or config.get("input") != "rgb_only" or config.get("camera_input") != "none":
        raise ValueError("Expected train/dev RGB-only no-camera configuration")
    dev = config["split"] == "dev"
    allowed_splits = frozenset({"train", "val"} if dev else {"train"})
    frames = config["frames"]
    if frames == "manifest":
        records = json.loads((ROOT / config["processed"] / "manifest.json").read_text(encoding="utf8"))["records"]
        # (label, relative sample path); several sensor samples may share one RGB frame.
        items = [(r["frame"], r["path"]) for r in records if r["split"] in allowed_splits]
    else:
        items = [(str(f), f"{f}.npz") for f in frames]
    items = items[:limit] if limit is not None else items
    max_samples = int(config.get("max_samples", 95 if dev else 5))
    if not items or len(items) > max_samples:
        raise ValueError(f"Sample count outside bounds (1..{max_samples})")
    frames = [label for label, _ in items]
    output = Path(output or config["output"])
    if not output.is_absolute(): output = ROOT / output
    output = output.resolve()
    if not output.is_relative_to((ROOT / "cache").resolve()):
        raise ValueError("Cache output must be inside TrustPrompt/cache")
    source, weights = Path(config["source"]).resolve(), Path(config["weights"]).resolve()
    actual_config = {**config, "frames": frames, "output": str(output), "config_file": str(config_path)}
    receipt = start_run(output, actual_config, [sys.executable, *sys.argv], config["seed"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(output / "runtime_cache/torch")
    os.environ["HF_HOME"] = str(output / "runtime_cache/huggingface")
    with (output / "run.log").open("w", encoding="utf8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            import torch
            if config["device"] != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("This smoke run requires the verified local CUDA device")
            random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
            torch.set_num_threads(config["torch_threads"])
            torch.backends.cudnn.benchmark = False
            source_files = sorted((source / "unidepth").rglob("*.py"))
            receipt["model_version"] = {
                "name": "UniDepthV2-ViT-S14", "source_directory": str(source),
                "source_commit": git_source(source, "rev-parse", "HEAD"),
                "source_status": git_source(source, "status", "--porcelain=v1"),
                "source_hashes": {str(p.relative_to(source)): sha256(p) for p in source_files},
                "weight_directory": str(weights),
                "weight_hashes": {p.name: sha256(p) for p in weights.iterdir() if p.name in {"config.json", "model.safetensors", "pytorch_model.bin"}},
                "license_source": "Third-party UniDepth source header: CC-BY-NC 4.0",
                "infer_entry": "UniDepthV2.infer(uint8_RGB), no camera argument",
                "native_depth_evidence": "unidepth/models/unidepthv2/unidepthv2.py: out['depth'] = points[:, -1:]",
            }
            old_loader = Path("D:/Python_Script/GLARE/experiments/generate_i3_transphy3d_unidepth_cache.py")
            if old_loader.is_file():
                receipt["ported_reference"] = {"path": str(old_loader), "sha256": sha256(old_loader),
                    "functions": ["_wandb_import_stub", "_module_is_within_source", "_import_local_unidepth_class", "_LocalUniDepthPredictor"]}
            receipt["gpu"] = {"name": torch.cuda.get_device_name(), "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                              "torch": torch.__version__, "cuda": torch.version.cuda}
            receipt["rgb_only_access_contract"] = {"decoded_npz_fields": ["metadata_json", "rgb"],
                "other_npz_bytes_hashed_opaquely": True, "gt_prompt_material_camera_used_by_model": False}
            write_json(output / "receipt.json", receipt)
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            predictor = UniDepthPredictor(source, weights, config["device"], config["resolution_level"])
            torch.cuda.synchronize()
            receipt["load_seconds"] = time.perf_counter() - begin
            rows = []
            (output / "predictions").mkdir()
            done_rgb = {}   # rgb sha256 -> (prediction_path, report): RGB-only model, infer once per image
            for frame, rel in items:
                path = (ROOT / config["processed"] / rel).resolve()
                rgb, identities = load_rgb_training_npz(path, allowed_splits)
                if identities["frame"] != str(frame): raise ValueError("Frame identity mismatch")
                rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                if rgb_hash in done_rgb:
                    prediction_path, report = done_rgb[rgb_hash]
                    seconds, peak_a, peak_r, shared = None, None, None, True
                else:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    arrays, report = predictor.predict_rgb(rgb)
                    torch.cuda.synchronize(); seconds = time.perf_counter() - begin
                    prediction_path = output / "predictions" / f"{identities['scene']}__{frame}.npz"
                    if not config.get("store_points", True):
                        # points Z == depth is verified in `report` before the point cloud is dropped.
                        arrays = {k: v for k, v in arrays.items() if k != "points"}
                    with prediction_path.open("xb") as stream: np.savez_compressed(stream, **arrays)
                    peak_a, peak_r, shared = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), False
                    done_rgb[rgb_hash] = (prediction_path, report)
                row = {**identities, "sample_path": str(path), "sample_npz_sha256": sha256(path),
                    "rgb_sha256": rgb_hash, "rgb_shape": list(rgb.shape),
                    "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
                    "seconds": seconds, "prediction_shared_with_same_rgb": shared,
                    "peak_allocated_bytes": peak_a, "peak_reserved_bytes": peak_r, "validation": report}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                write_json(output / "images.json", rows)
            receipt.update(stage="real_dev_verified" if dev else "real_train_smoke_verified", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                           images_sha256=sha256(output / "images.json"),
                           frame_count=len(rows), all_points_z_equal_depth=all(r["validation"]["points_z_exactly_equal_depth"] for r in rows),
                           inputs=[{k:r[k] for k in ("dataset","sensor","scene","frame","split","rgb_sha256","sample_npz_sha256")} for r in rows])
        except BaseException as error:
            receipt.update(stage="failed", error=repr(error))
            traceback.print_exc()
            raise
        finally:
            write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unidepth_smoke.yaml")
    parser.add_argument("--limit", type=int, choices=range(1, 6))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.config, args.limit, args.output)
