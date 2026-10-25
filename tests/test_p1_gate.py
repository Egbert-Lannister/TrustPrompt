import json

import pytest
import yaml

import scripts.holm_family as hf
import scripts.p1_gate as gate

SCENES = {"mirror3d_nyu": ["s1", "s2", "s3", "s4"], "hammer": ["h1", "h2", "h3"]}
SPLIT = {"mirror3d_nyu": "val", "hammer": "val"}


def write_run(root, prefix, run, split, ds, c, clean):
    d = root / (run[split] if isinstance(run, dict) else f"results/{prefix}_{run}_{split}_v1")
    d.mkdir(parents=True)
    rows = [dict(dataset=ds, sensor="d435", scene=sc, frame=f"{i:04d}", split=split,
                 metrics=dict(regions=dict(C=dict(status="ok", count=10, absrel=c * (1 + 0.01 * k + 0.001 * i)),
                                           clean=dict(status="ok", count=10, absrel=clean * (1 + 0.01 * k)))))
            for k, sc in enumerate(SCENES[ds]) for i in range(2)]
    scope = "mirror_only" if ds == "mirror3d_nyu" else "objects_only"
    (d / f"per_image_{scope}.json").write_text(json.dumps(rows))
    (d / "receipt.json").write_text(json.dumps(dict(stage="small_scale_validation_passed")))


@pytest.mark.parametrize("f1_c, f1_clean, expect", [(0.5, 0.1, True), (0.8, 0.1, False), (0.5, 0.2, False)])
def test_gate(tmp_path, monkeypatch, f1_c, f1_clean, expect):
    spec = yaml.safe_load((gate.ROOT / "configs" / "omnidc" / "p1_gate_v1.yaml").read_text())
    spec["n_resamples"] = 500
    for name, c, clean in (("R0", 1.0, 0.1), ("R1", 0.6, 0.1), ("F0", 1.0, 0.1), ("F1", f1_c, f1_clean)):
        for ds, d in spec["datasets"].items():
            for split in ("val", "train"):        # HAMMER all_dev reads both splits
                write_run(tmp_path, d["prefix"], spec["runs"][name][ds], split, ds, c, clean)
    monkeypatch.setattr(hf, "ROOT", tmp_path); monkeypatch.setattr(gate, "ROOT", tmp_path)
    g1 = gate.comparison(spec, "mirror3d_nyu", "F0", "F1", "C")
    assert g1["relative_change"] == pytest.approx(f1_c - 1.0, abs=1e-6)
    g3 = gate.comparison(spec, "hammer", "R0", "F1", "clean")
    assert gate.clearly_worse(g3) == (f1_clean > 0.15)
    ok = g1["relative_change"] <= spec["g1_max_relative_change"] and not gate.clearly_worse(g3)
    assert ok == expect


def test_missing_run_is_na(tmp_path, monkeypatch):
    spec = yaml.safe_load((gate.ROOT / "configs" / "omnidc" / "p1_gate_v1.yaml").read_text())
    monkeypatch.setattr(hf, "ROOT", tmp_path); monkeypatch.setattr(gate, "ROOT", tmp_path)
    r = gate.comparison(spec, "hammer", "F0", "F1", "C")
    assert r["status"] == "NA_missing_run" and gate.clearly_worse(r) is None


def test_r0_clipping(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "images.json").write_text(json.dumps([dict(clipped_pixels=0), dict(clipped_pixels=5)]))
    out = gate.r0_clipping(dict(r0_caches=dict(hammer="c", mirror3d_nyu="missing")))
    assert out["hammer"] == dict(images=2, images_with_clipping=1, clipped_pixels=5) and "missing" in out["mirror3d_nyu"]["status"]
