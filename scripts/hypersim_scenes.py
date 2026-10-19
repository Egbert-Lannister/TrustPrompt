"""Pick the Hypersim scenes for the structured-error synthesis (docs/plan_omnidc.md sec. 3-4) from the metadata
in an apple/ml-hypersim checkout; no image data is read. -> configs/hypersim/selection_v1.json

  python -m scripts.hypersim_scenes --repo PATH_TO_ml-hypersim

Rule: scenes whose mesh annotation (evermotion_dataset/scenes/<scene>/_detail/mesh/mesh_objects_si.hdf5) contains
an object labelled NYU40 19 (mirror). Per scene: whether NYU40 9 (window) objects exist, the public-release frames
per camera with their official split (metadata_images_split_scene_v1.csv), and the per-scene camera matrices
(contrib/mikeroberts3000/metadata_camera_parameters.csv: M_cam_from_uv for ray casting, M_proj, image size).
Which frames actually show a mirror is decided later from the downloaded semantic maps (scripts/hypersim_fetch.py).
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from tp.provenance import ROOT

MIRROR, WINDOW = 19, 9
OUT = ROOT / "configs" / "hypersim" / "selection_v1.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True, type=Path)
    a = ap.parse_args()
    repo = a.repo
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    split_csv = repo / "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
    cam_csv = repo / "contrib/mikeroberts3000/metadata_camera_parameters.csv"
    frames = defaultdict(lambda: defaultdict(list)); split = {}
    for r in csv.DictReader(split_csv.open(encoding="utf-8")):
        if r["included_in_public_release"] == "True":
            frames[r["scene_name"]][r["camera_name"]].append(int(r["frame_id"]))
            split[r["scene_name"]] = r["split_partition_name"]
    cams = {r["scene_name"]: r for r in csv.DictReader(cam_csv.open(encoding="utf-8"))}
    scenes = {}
    for mesh in sorted((repo / "evermotion_dataset/scenes").glob("ai_*/_detail/mesh/mesh_objects_si.hdf5")):
        name = mesh.parts[-4]
        with h5py.File(mesh, "r") as h:
            labels = np.asarray(h["dataset"])
        if not (labels == MIRROR).any() or name not in frames:
            continue
        c = cams[name]
        scenes[name] = dict(
            split=split[name], has_window=bool((labels == WINDOW).any()),
            frames={cam: sorted(ids) for cam, ids in sorted(frames[name].items())},
            camera=dict(height=int(float(c["settings_output_img_height"])), width=int(float(c["settings_output_img_width"])),
                        meters_per_asset_unit=float(c["settings_units_info_meters_scale"]),
                        M_cam_from_uv=[[float(c[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)],
                        M_proj=[[float(c[f"M_proj_{i}{j}"]) for j in range(4)] for i in range(4)]))
    out = dict(schema="trustprompt.hypersim_selection.v1",
               rule="mesh annotation contains NYU40 19 (mirror); public-release frames only",
               source=dict(repo="https://github.com/apple/ml-hypersim", commit=commit,
                           files={str(p.relative_to(repo)).replace("\\", "/"): sha256(p) for p in (split_csv, cam_csv)}),
               url_template="https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip",
               scenes=scenes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    n = sum(len(v) for s in scenes.values() for v in s["frames"].values())
    print(f"{len(scenes)} scenes, {n} public frames, splits "
          f"{dict(sorted({k: sum(1 for s in scenes.values() if s['split'] == k) for k in ('train', 'val', 'test')}.items()))} -> {OUT}")


if __name__ == "__main__":
    main()
