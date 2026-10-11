"""HAMMER loader on a synthetic directory with the real layout (no real data)."""
import numpy as np
import pytest
from PIL import Image

from tp.data.hammer import list_frames, load_sample, scene_split


def write_seq(root, name, n, depth_mm=800, pose_shift=0.0, objects=True):
    pol = root / name / "polarization"
    for sub in ("rgb", "_gt", "_instance", "_pose", "depth_d435", "depth_l515", "depth_tof"):
        (pol / sub).mkdir(parents=True, exist_ok=True)
    (pol / "intrinsics.txt").write_text("700 0 32\n0 700 24\n0 0 1\n")
    for i in range(n):
        f = f"{i:06d}"
        Image.fromarray(np.full((48, 64, 3), 100, np.uint8)).save(pol / "rgb" / f"{f}.png")
        gt = np.full((48, 64), depth_mm, np.uint16)
        if objects:
            gt[10:20, 10:20] = 500                       # an object in front of the background
        Image.fromarray(gt).save(pol / "_gt" / f"{f}.png")
        Image.fromarray(np.full((48, 64), 20, np.uint8)).save(pol / "_instance" / f"{f}.png")
        pose = np.eye(4); pose[0, 3] = pose_shift
        np.savetxt(pol / "_pose" / f"{f}.txt", pose)
        for sub in ("depth_d435", "depth_l515", "depth_tof"):
            d = gt.copy(); d[:5] = 0                      # holes
            d[10:20, 10:20] = depth_mm                   # sensor sees through the object
            Image.fromarray(d).save(pol / sub / f"{f}.png")


def test_loader_units_pairing_and_test_lock(tmp_path):
    write_seq(tmp_path, "scene3_traj1_1", 3)
    write_seq(tmp_path, "scene3_traj1_naked_1", 3, objects=False)
    write_seq(tmp_path, "scene12_traj1_1", 1)
    (tmp_path / "scene3_traj1_1/polarization/depth_d435/{06d}.png").write_bytes(b"x")
    rows, stray = list_frames(tmp_path)
    assert [r["index"] for r in rows] == [0, 1, 2] and all(r["split"] == "train" for r in rows)
    assert any("{06d}" in s for s in stray)
    s = load_sample(tmp_path, "scene3_traj1_1", 1, "l515")
    assert s.prompt.dtype == np.float32 and s.prompt[30, 30] == pytest.approx(0.8)
    assert not s.prompt_valid[0].any() and s.prompt_valid[30].all()
    assert s.gt[15, 15] == pytest.approx(0.5) and s.behind[15, 15] == pytest.approx(0.8)
    assert s.metadata["naked_pairing"]["status"] == "paired" and s.frame == "traj1_1_000001"
    assert s.material.max() == 0                         # HAMMER has no material labels
    with pytest.raises(ValueError, match="locked"):
        load_sample(tmp_path, "scene12_traj1_1", 0, "d435")
    assert load_sample(tmp_path, "scene12_traj1_1", 0, "d435", allow_test=True).split == "test"


def test_pose_mismatch_disables_behind(tmp_path):
    write_seq(tmp_path, "scene5_traj2_1", 1)
    write_seq(tmp_path, "scene5_traj2_naked_1", 1, objects=False, pose_shift=0.01)
    s = load_sample(tmp_path, "scene5_traj2_1", 0, "helios")
    assert s.behind is None and s.metadata["naked_pairing"]["status"] == "pose_mismatch"


def test_split_rule():
    assert [scene_split(i) for i in (2, 10, 11, 12, 14)] == ["train", "train", "val", "test", "test"]


def test_prepare_and_evaluate_write_object_scope(tmp_path, monkeypatch):
    import json, sys
    from scripts import prepare_hammer
    from tp.eval.run_eval import evaluate_manifest
    root = tmp_path / "hammer"
    write_seq(root, "scene3_traj1_1", 4); write_seq(root, "scene3_traj1_naked_1", 4, objects=False)
    write_seq(root, "scene11_traj1_1", 2); write_seq(root, "scene11_traj1_naked_1", 2, objects=False)
    write_seq(root, "scene13_traj1_1", 2)                       # test scene: must not be materialized
    out = tmp_path / "proc"
    monkeypatch.setattr(sys, "argv", ["prepare_hammer", "--root", str(root), "--output", str(out),
                                      "--stride-train", "2", "--stride-val", "1"])
    prepare_hammer.main()
    m = json.loads((out / "manifest.json").read_text())
    assert {r["split"] for r in m["records"]} == {"train", "val"} and len(m["records"]) == (2 + 2) * 3
    cfg = dict(method="raw_prompt", split="val", manifest=str(out / "manifest.json"), tau=0.05, band=5,
               visualizations=1, experiment_role="development_diagnostic_not_final_test", seed=1)
    evaluate_manifest(cfg, tmp_path / "run", ["t"])
    rows = json.loads((tmp_path / "run" / "per_image_objects_only.json").read_text())
    assert all(r["gt_scope"] == "objects_vs_naked_gt_gt_1mm" for r in rows)
    assert rows[0]["population"]["C"] == 100 and rows[0]["metrics"]["IR"]["value"] == 1.0   # sees through object
    assert not (tmp_path / "run" / "per_image_mirror_only.json").exists()
