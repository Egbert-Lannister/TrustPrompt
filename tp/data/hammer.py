"""HAMMER loader (plan section 2.2 / revision v2 R7).

Uses only the polarization-camera view of the official processed release:
RGB, the three sensors' depth already warped into that view by the dataset
authors (depth_d435 / depth_l515 / depth_tof), rendered GT, instance map,
camera pose and intrinsics. One Sample per (frame, sensor).

Verified facts (docs/hammer_audit.md): 16-bit PNG depth in millimetres, 0 =
no measurement; 832x1088 grid; intrinsics 3x3 text; poses 4x4 text; naked
(objects removed) sequences repeat the robot trajectory.

Not provided by HAMMER: per-instance material/category labels. material is
therefore 0 (unknown) for every pixel until an audited instance->material
table exists (configs/hammer_materials_v*.csv); it never enters region
definitions or method inputs.

Split by scene: 2-10 train, 11 val (revision v2 2.2), 12-14 test (official).
Test samples are refused unless allow_test=True (formal evaluation only).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import Sample

SENSORS = {"d435": "depth_d435", "l515": "depth_l515", "helios": "depth_tof"}
SENSOR_PRINCIPLE = {"d435": "active_stereo", "l515": "direct_tof", "helios": "indirect_tof"}
TRAIN_SCENES = set(range(2, 11))
VAL_SCENES = {11}
TEST_SCENES = {12, 13, 14}
SEQ_RE = re.compile(r"^scene(\d+)_traj(\d+)_(\d+)$")
NAKED_RE = re.compile(r"^scene(\d+)_traj(\d+)_naked_(\d+)$")
FRAME_RE = re.compile(r"^(\d{6})\.png$")
# Naked pairing tolerance: the robot repeats the trajectory; pairs farther
# apart than this are not used as "behind" depth.
PAIR_MAX_TRANSLATION_M = 1e-3
PAIR_MAX_ROTATION_DEG = 0.1


def scene_split(scene_number: int) -> str:
    if scene_number in TRAIN_SCENES:
        return "train"
    if scene_number in VAL_SCENES:
        return "val"
    if scene_number in TEST_SCENES:
        return "test"
    raise ValueError(f"unknown HAMMER scene {scene_number}")


def read_matrix(path, shape):
    m = np.loadtxt(path, dtype=np.float64)
    if m.shape != shape or not np.isfinite(m).all():
        raise ValueError(f"{path}: expected finite {shape} matrix")
    return m


def read_depth_mm(path):
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim != 2 or a.dtype != np.uint16:
        raise ValueError(f"{path}: expected 16-bit single-channel depth PNG, got {a.dtype} {a.shape}")
    return a


def pose_gap(a, b):
    """Translation (m) and rotation (deg) difference between two 4x4 poses."""
    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    r = a[:3, :3].T @ b[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))
    return dt, angle


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_frames(root, splits=("train", "val")):
    """Enumerate complete frames of object sequences in the requested splits.
    Returns rows with sequence, frame index, split and naked pairing status.
    Stray files (e.g. a literal '{06d}.png' in the release) are ignored and
    reported."""
    root = Path(root)
    rows, stray = [], []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = SEQ_RE.match(seq_dir.name)
        if not m:
            continue
        scene, traj, part = map(int, m.groups())
        split = scene_split(scene)
        if split not in splits:
            continue
        pol = seq_dir / "polarization"
        sets = {}
        for sub in ("rgb", "_gt", "_instance", "_pose", *SENSORS.values()):
            names = set()
            for f in (pol / sub).iterdir():
                stem_ok = FRAME_RE.match(f.name) or re.match(r"^\d{6}\.txt$", f.name)
                (names.add(int(f.name[:6])) if stem_ok else stray.append(str(f.relative_to(root))))
            sets[sub] = names
        complete = set.intersection(*sets.values())
        naked = root / f"scene{scene}_traj{traj}_naked_{part}" / "polarization"
        for idx in sorted(complete):
            rows.append(dict(sequence=seq_dir.name, scene=f"scene{scene}", traj=f"traj{traj}_{part}",
                             index=idx, split=split,
                             naked_gt_exists=(naked / "_gt" / f"{idx:06d}.png").is_file()))
        incomplete = set.union(*sets.values()) - complete
        if incomplete:
            stray.extend(f"{seq_dir.name}: incomplete frame {i:06d}" for i in sorted(incomplete))
    return rows, stray


def load_sample(root, sequence, index, sensor, *, allow_test=False, with_behind=True):
    root = Path(root)
    m = SEQ_RE.match(sequence)
    if not m or sensor not in SENSORS:
        raise ValueError("bad sequence or sensor")
    scene, traj, part = map(int, m.groups())
    split = scene_split(scene)
    if split == "test" and not allow_test:
        raise ValueError("HAMMER test scenes are locked until method freeze")
    pol = root / sequence / "polarization"
    name = f"{index:06d}"
    files = {"rgb": pol / "rgb" / f"{name}.png", "prompt": pol / SENSORS[sensor] / f"{name}.png",
             "gt": pol / "_gt" / f"{name}.png", "instance": pol / "_instance" / f"{name}.png",
             "pose": pol / "_pose" / f"{name}.txt", "intrinsics": pol / "intrinsics.txt"}
    with Image.open(files["rgb"]) as im:
        rgb = np.array(im.convert("RGB"))
    prompt_mm = read_depth_mm(files["prompt"])
    gt_mm = read_depth_mm(files["gt"])
    if prompt_mm.shape != rgb.shape[:2] or gt_mm.shape != rgb.shape[:2]:
        raise ValueError("HAMMER grids differ")
    K = read_matrix(files["intrinsics"], (3, 3))
    pose = read_matrix(files["pose"], (4, 4))
    prompt = prompt_mm.astype(np.float32) / 1000.0
    gt = gt_mm.astype(np.float32) / 1000.0
    behind, pairing = None, dict(status="not_requested")
    if with_behind:
        naked = root / f"scene{scene}_traj{traj}_naked_{part}" / "polarization"
        ng, npose = naked / "_gt" / f"{name}.png", naked / "_pose" / f"{name}.txt"
        if ng.is_file() and npose.is_file():
            dt, dr = pose_gap(pose, read_matrix(npose, (4, 4)))
            ok = dt <= PAIR_MAX_TRANSLATION_M and dr <= PAIR_MAX_ROTATION_DEG
            pairing = dict(status="paired" if ok else "pose_mismatch", translation_m=dt, rotation_deg=dr)
            if ok:
                behind = read_depth_mm(ng).astype(np.float32) / 1000.0
                files["behind"], files["behind_pose"] = ng, npose
        else:
            pairing = dict(status="naked_frame_missing")
    sample = Sample(
        rgb=rgb, prompt=prompt, prompt_valid=prompt_mm > 0, gt=gt, gt_valid=gt_mm > 0,
        material=np.zeros(rgb.shape[:2], np.uint8), K_rgb=K.copy(), K_prompt=K.copy(),
        sensor=sensor, dataset="hammer", scene=f"scene{scene}", frame=f"traj{traj}_{part}_{name}",
        split=split, behind=behind,
        metadata=dict(
            depth_unit="m", depth_convention="axial_z", synthetic=False,
            source="HAMMER _dataset_processed.zip, polarization view", sequence=sequence, index=index,
            sensor_principle=SENSOR_PRINCIPLE[sensor], prompt_source=f"polarization/{SENSORS[sensor]} (authors' warp)",
            prompt_encoding="uint16 mm, 0=invalid", gt_source="polarization/_gt rendered from registered meshes",
            gt_scope="rendered_first_surface_full_image", material_source="unknown: HAMMER has no material labels",
            naked_pairing=pairing, camera_reference="polarization camera; GT axial-Z audited in docs/hammer_audit.md",
            strict_rgb_z_qualified=True,
            source_sha256={k: sha256_file(v) for k, v in files.items()}))
    return sample.validate()


def load_instance(root, sequence, index):
    with Image.open(Path(root) / sequence / "polarization" / "_instance" / f"{index:06d}.png") as im:
        return np.array(im)
