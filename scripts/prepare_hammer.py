"""Materialize a strided HAMMER development subset as Sample NPZ (train/val only).

  python -m scripts.prepare_hammer --root E:/TrustPrompt_data/hammer_subset_v1 \
      --output data/processed/hammer_dev_v1 --stride-train 20 --stride-val 5

Frames are taken where index % stride == 0 within each object sequence; all three
sensors per frame. Test scenes (12-14) are refused. Every sample records the
SHA256 of each source file; the manifest records stray/incomplete files and the
naked-pairing outcome. Outputs are immutable (exclusive create).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tp.data.hammer import SENSORS, list_frames, load_sample
from tp.data.schema import save_sample
from tp.provenance import sha256, start_run, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stride-train", type=int, default=20)
    ap.add_argument("--stride-val", type=int, default=5)
    args = ap.parse_args()
    config = dict(root=str(args.root), stride_train=args.stride_train, stride_val=args.stride_val,
                  sensors=sorted(SENSORS), splits=["train", "val"], seed=20261010)
    receipt = start_run(args.output, config, [sys.executable, "-m", "scripts.prepare_hammer", *sys.argv[1:]], 20261010)
    try:
        rows, stray = list_frames(args.root, splits=("train", "val"))
        chosen = [r for r in rows if r["index"] % (args.stride_train if r["split"] == "train" else args.stride_val) == 0]
        records, pairing = [], Counter()
        for k, r in enumerate(chosen):
            for sensor in sorted(SENSORS):
                s = load_sample(args.root, r["sequence"], r["index"], sensor)
                path = args.output / "samples" / f"{s.sensor}__{s.scene}__{s.frame}.npz"
                save_sample(s, path)
                pairing[s.metadata["naked_pairing"]["status"]] += 1
                records.append(dict(path=f"samples/{path.name}", sha256=sha256(path), dataset=s.dataset,
                                    scene=s.scene, frame=s.frame, split=s.split, sensor=s.sensor,
                                    sequence=r["sequence"], index=r["index"]))
            if k % 20 == 0:
                print(json.dumps(dict(progress=f"{k + 1}/{len(chosen)}", samples=len(records))), flush=True)
        write_json(args.output / "manifest.json", dict(
            schema="trustprompt.manifest.v1", synthetic=False, records=records, seed=20261010,
            gt_scope="rendered_first_surface_full_image", strict_rgb_z_qualified=True,
            source_config=config, stray_or_incomplete=stray, naked_pairing=dict(pairing),
            frames_available={s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val")},
            frames_selected={s: sum(1 for r in chosen if r["split"] == s) for s in ("train", "val")}))
        receipt.update(stage="complete_real_data_preparation", samples=len(records), naked_pairing=dict(pairing))
    except Exception as e:
        receipt.update(stage="failed", error=repr(e)); raise
    finally:
        write_json(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
