import json
import sys

import numpy as np
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


def synthetic_evaluation_runs(root, spec, monkeypatch, case):
    """Actual NPZ evaluator -> run artifacts; only the model prediction is synthetic.

    All inputs and outputs live in the isolated test directory. No checkpoint,
    real dataset, test split, or GPU is opened. This does not measure model quality.
    """
    import baselines.raw_prompt as raw
    import tp.eval.run_eval as evaluator
    import tp.provenance as provenance
    from tp.data.schema import Sample, save_sample

    monkeypatch.setattr(provenance, "ROOT", root)
    monkeypatch.setattr(evaluator, "ROOT", root)
    loaded = []
    original_load = evaluator.load_sample

    def load_train_or_val(path):
        assert path.resolve().is_relative_to(root.resolve())
        sample = original_load(path)
        assert sample.split in {"train", "val"} and sample.metadata["synthetic"] is True
        loaded.append(sample.key)
        return sample

    monkeypatch.setattr(evaluator, "load_sample", load_train_or_val)
    outputs = {}
    for ds, definition in spec["datasets"].items():
        records = []
        # Disjoint scene IDs, including between train/val, exercise scene grouping.
        for split in ("train", "val"):
            for scene_index in range(2):
                h, w = 8, 12
                gt = np.full((h, w), 2.0, np.float32)
                prompt = gt.copy(); prompt[:, :w // 2] = 4.0
                sample = Sample(
                    rgb=np.zeros((h, w, 3), np.uint8), prompt=prompt,
                    prompt_valid=np.ones((h, w), bool), gt=gt, gt_valid=np.ones((h, w), bool),
                    material=np.full((h, w), 2, np.uint8), K_rgb=np.eye(3), K_prompt=np.eye(3),
                    sensor="synthetic", dataset=ds, scene=f"synthetic_{ds}_{split}_{scene_index}",
                    frame="0000", split=split, behind=np.full((h, w), 3.0, np.float32),
                    metadata=dict(synthetic=True, depth_unit="m", depth_convention="axial_z",
                                  gt_scope="synthetic_exact", strict_rgb_z_qualified=True))
                path = root / "samples" / f"{sample.scene}.npz"
                save_sample(sample, path)
                records.append(dict(dataset=ds, scene=sample.scene, sensor=sample.sensor, frame=sample.frame,
                                    split=split, path=str(path.relative_to(root)), sha256=gate.sha256(path)))
        manifest = root / f"{ds}_synthetic_manifest.json"
        provenance.write_json(manifest, dict(synthetic=True, records=records))
        for model in ("R0", "R1", "F0", "F1"):
            c_error, clean_error = (0.6 if model == "R1" else 1.0), 0.125
            if model == "F1":
                c_error = 0.5
                if case == "g1_fail" and ds == "mirror3d_nyu":
                    c_error = 0.8
                if case == "g2_tie" and ds == "hammer":
                    c_error = 1.0
                if case == "g3_fail" and ds == "hammer":
                    clean_error = 0.25

            def prediction(sample, c=c_error, clean=clean_error, name=model, dataset=ds):
                pred = sample.gt * np.where(sample.prompt > sample.gt, 1 + c, 1 + clean)
                if name == "F1" and dataset == "hammer" and sample.split == "train":
                    if case == "invalid_c":
                        pred[0, 0] = np.nan
                    if case == "invalid_clean":
                        pred[0, -1] = np.nan
                return pred.astype(np.float32)

            monkeypatch.setattr(raw, "predict", prediction)
            for split in ("train", "val"):
                run = spec["runs"][model][ds]
                folder = root / (run[split] if isinstance(run, dict)
                                 else f"results/{definition['prefix']}_{run}_{split}_v1")
                config = dict(seed=7, manifest=str(manifest), split=split, method="raw_prompt", method_label=model,
                              alignment="none", tau=0.05, band=1, visualizations=0, save_predictions=0,
                              experiment_role="synthetic_gate_engineering_test")
                evaluator.evaluate_manifest(config, folder, ["pytest", "synthetic_gate_engineering_test"])
                receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
                assert receipt["synthetic"] is True and receipt["formal_experiment_complete"] is False
                scope_rows = json.loads((folder / f"per_image_{definition['scope']}.json").read_text(encoding="utf-8"))
                assert all(row["synthetic"] is True and row["population"]["C"] == 48 for row in scope_rows)
                outputs[model, ds, split] = folder
    assert len(loaded) == 32
    return outputs


@pytest.mark.parametrize("case, expected", [
    ("pass", {"G1": True, "G2": True, "G3": True}),
    ("g1_fail", {"G1": False, "G2": True, "G3": True}),
    ("g2_tie", {"G1": True, "G2": False, "G3": True}),
    ("g3_fail", {"G1": True, "G2": True, "G3": False}),
    ("invalid_c", {"G1": True, "G2": None, "G3": True}),
    ("invalid_clean", {"G1": True, "G2": True, "G3": None}),
    ("missing_run", {"G1": True, "G2": None, "G3": None}),
    ("partial_run", {"G1": True, "G2": None, "G3": None}),
])
def test_gate_cli_from_synthetic_evaluation_runs(tmp_path, monkeypatch, capsys, case, expected):
    """Exercise G1 AND G2 AND both-dataset N1 through the real gate entry point.

    The earlier comparison-only test could pass if main() ignored G2 or treated
    invalid/partial/missing evaluations as successful. These fixtures distinguish
    each branch using real fixed-denominator metrics from evaluate_manifest().
    """
    spec = yaml.safe_load((gate.ROOT / "configs/omnidc/p1_gate_v1.yaml").read_text(encoding="utf-8"))
    spec.update(n_resamples=100, synthetic=True, experiment_role="synthetic_gate_engineering_test")
    outputs = synthetic_evaluation_runs(tmp_path, spec, monkeypatch, case)
    if case == "missing_run":
        spec["runs"]["F1"]["hammer"]["train"] = "absent_synthetic_evaluation"
    if case == "partial_run":
        receipt_file = outputs["F1", "hammer", "train"] / "receipt.json"
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        receipt["stage"] = "partial_time_budget"
        receipt_file.write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    spec_path = tmp_path / "synthetic_gate.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(hf, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["scripts.p1_gate", "--spec", spec_path.name])
    gate.main()
    [output] = list((tmp_path / "results").glob("p1_gate_*"))
    result = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert result["checks"] == expected
    assert result["passed"] is (case == "pass")
    assert receipt["checks"] == expected and receipt["passed"] is result["passed"]
    assert receipt["stage"] == "descriptive_comparison_complete" and receipt["formal_experiment_complete"] is False
    assert receipt["config"]["synthetic"] is True
    assert receipt["gate_sha256"] == gate.sha256(output / "gate.json")
    assert receipt["config"]["spec_sha256"] == gate.sha256(spec_path)
    report = (tmp_path / "docs/omnidc_p1_gate.md").read_text(encoding="utf-8")
    assert ("P1 passed: go to P2" if case == "pass" else "P1 NOT passed: stop GPU spending") in report
    assert result["gate"]["G1"]["scene_count"] == 2
    if case not in {"missing_run", "partial_run"}:
        compared = result["gate"]["G2"]
        assert len(compared["candidate_sha256"]) == 2  # HAMMER train + val, not val alone.
        if case == "invalid_c":
            assert compared["status"] == "NA_invalid_or_missing" and compared["n_invalid"] == 2
        else:
            assert compared["scene_count"] == 4
    if case == "invalid_clean":
        assert result["gate"]["G3_hammer"]["status"] == "NA_invalid_or_missing"
    capsys.readouterr()


@pytest.mark.parametrize("ci_low, relative, expected", [
    (0.0, 0.4, False), (0.01, 0.2, False), (0.01, 0.2001, True), (None, 0.4, None),
])
def test_n1_requires_both_strict_thresholds(ci_low, relative, expected):
    assert gate.clearly_worse(dict(ci_low=ci_low, relative_change=relative)) is expected
