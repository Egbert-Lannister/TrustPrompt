import json
import pytest
from scripts.make_synthetic import make_sample
from tp.data.schema import save_sample
from tp.provenance import sha256,write_json
from tp.eval.run_eval import evaluate_manifest


def fixture(tmp_path,qualified=True):
    s=make_sample()
    s.metadata["strict_rgb_z_qualified"]=qualified
    path=tmp_path/"sample.npz";save_sample(s,path)
    row=dict(path=path.name,sha256=sha256(path),dataset=s.dataset,scene=s.scene,frame=s.frame,split=s.split,sensor=s.sensor)
    manifest=tmp_path/"manifest.json"
    write_json(manifest,dict(synthetic=True,records=[row]))
    return dict(seed=0,manifest=str(manifest),method="raw_prompt",split="train",tau=.05,band=1,
                alignment="none",visualizations=0,experiment_role="synthetic_engineering_only")


def test_runner_npz_manifest_hashes_and_mirror_fixed_counts(tmp_path):
    config=fixture(tmp_path)
    result=evaluate_manifest(config,tmp_path/"run",["test"],)
    rows=json.loads((tmp_path/"run/per_image_mirror_only.json").read_text())
    assert rows[0]["population"]["all"]==rows[0]["metrics"]["regions"]["all"]["count"]
    assert rows[0]["population"]["all"]<96*128
    assert result["qualification"]["synthetic"] is True
    assert (tmp_path/"run/source_snapshot.zip").exists()


def test_formal_main_refuses_unqualified_rgb_z(tmp_path):
    config=fixture(tmp_path,False)
    config["experiment_role"]="formal_main_result"
    with pytest.raises(ValueError,match="not qualified"):
        evaluate_manifest(config,tmp_path/"run",["test"])
    receipt=json.loads((tmp_path/"run/receipt.json").read_text())
    assert receipt["stage"]=="failed"


def test_manifest_tamper_fails_before_prediction(tmp_path):
    config=fixture(tmp_path)
    path=tmp_path/"sample.npz"
    with path.open("ab") as f:f.write(b"different_source")
    with pytest.raises(ValueError,match="checksum"):
        evaluate_manifest(config,tmp_path/"run",["test"])


def test_test_split_locked_before_method_freeze(tmp_path):
    import pytest
    from tp.eval.run_eval import evaluate_manifest
    config = dict(method="raw_prompt", split="test", manifest="unused.json", tau=0.05, band=5,
                  experiment_role="development_diagnostic_not_final_test")
    with pytest.raises(ValueError, match="locked"):
        evaluate_manifest(config, tmp_path / "run", ["test"])


def test_time_budget_then_resume_gives_same_rows_as_one_shot(tmp_path):
    import json, shutil
    from tp.eval.run_eval import evaluate_manifest
    from tp.provenance import ROOT
    manifest = ROOT / "data/processed/synthetic_smoke_v1/manifest.json"
    if not manifest.is_file():
        import pytest; pytest.skip("synthetic smoke data not prepared")
    config = dict(method="raw_prompt", split="train", manifest=str(manifest), tau=0.05, band=5,
                  visualizations=0, experiment_role="engineering_smoke_not_final_test", seed=1)
    one = tmp_path / "one"; evaluate_manifest(dict(config), one, ["t"])
    two = tmp_path / "two"
    assert evaluate_manifest(dict(config), two, ["t"], time_budget_s=0.0) is None
    assert json.loads((two / "receipt.json").read_text())["stage"] == "partial_time_budget"
    evaluate_manifest(dict(config), two, ["t"], resume=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "evaluation_prediction_seconds"} for r in rows]
    assert strip(json.loads((one / "per_image.json").read_text())) == strip(json.loads((two / "per_image.json").read_text()))
    assert json.loads((two / "receipt.json").read_text())["resumed_sessions"]


def test_storage_policy_keeps_first_n_and_hashes_all(tmp_path):
    import json, hashlib
    import numpy as np
    from tp.eval.run_eval import evaluate_manifest
    from tp.provenance import ROOT
    manifest = ROOT / "data/processed/synthetic_smoke_v1/manifest.json"
    if not manifest.is_file():
        import pytest; pytest.skip("synthetic smoke data not prepared")
    cfg = dict(method="raw_prompt", split="train", manifest=str(manifest), tau=0.05, band=5, visualizations=0,
               experiment_role="engineering_smoke_not_final_test", seed=1, save_predictions=2)
    evaluate_manifest(cfg, tmp_path / "r", ["t"])
    rows = json.loads((tmp_path / "r" / "per_image.json").read_text())
    assert [r["prediction_saved"] for r in rows] == [True, True] + [False] * (len(rows) - 2)
    assert all(r["prediction_array_sha256"] for r in rows)
    saved = np.load(next((tmp_path / "r" / "predictions").glob("*.npy")))
    first = [r for r in rows if r["prediction_saved"]]
    assert hashlib.sha256(np.ascontiguousarray(saved, dtype=np.float32).tobytes()).hexdigest() in {r["prediction_array_sha256"] for r in first}


def test_train_scope_holdout_keeps_only_holdout_scenes(tmp_path):
    from tp.trust.data import scene_inner_holdout
    rows=[]
    for scene in ("sa","sb"):                      # synthetic dataset: sb is a holdout scene, sa is not
        s=make_sample(); s.scene=scene
        path=tmp_path/f"{scene}.npz";save_sample(s,path)
        rows.append(dict(path=path.name,sha256=sha256(path),dataset=s.dataset,scene=scene,frame=s.frame,split="train",sensor=s.sensor))
    assert [scene_inner_holdout(rows[0]["dataset"],r["scene"]) for r in rows]==[False,True]
    manifest=tmp_path/"manifest.json";write_json(manifest,dict(synthetic=True,records=rows))
    config=dict(seed=0,manifest=str(manifest),method="raw_prompt",split="train",tau=.05,band=1,alignment="none",
                visualizations=0,experiment_role="synthetic_engineering_only",train_scope="holdout")
    evaluate_manifest(config,tmp_path/"run",["test"])
    got=json.loads((tmp_path/"run/per_image.json").read_text())
    assert [r["scene"] for r in got]==["sb"]
    with pytest.raises(ValueError):
        evaluate_manifest(dict(config,train_scope="some"),tmp_path/"run2",["test"])
