"""PromptDA wrapper pieces that need no GPU/model; and cached-prediction evaluation."""
import json

import numpy as np
import pytest

from baselines.promptda import model_input_size, prepare_prompt


def test_model_input_size_matches_official_and_rounds_small_images():
    assert model_input_size(832, 1088)[:2] == (770, 1008)          # official INTER_AREA branch
    assert model_input_size(832, 1088)[2] == "official_area_downscale"
    h, w, mode = model_input_size(456, 608)
    assert (h, w) == (462, 602) and h % 14 == 0 and w % 14 == 0 and mode == "nearest_multiple_of_14_resize"
    assert model_input_size(448, 602)[2] == "none"


def test_prepare_prompt_raw_fills_holes_and_never_emits_zero():
    p = np.full((48, 64), 2.0, np.float32); v = np.ones_like(p, bool)
    v[:10] = False; p[:10] = 0
    out, info = prepare_prompt(p, v)
    assert out.shape == (192, 256) and np.all(out == 2.0) and info["status"] == "ok"


def test_oracle_variant_replaces_removed_pixels_with_prior():
    p = np.full((48, 64), 2.0, np.float32); v = np.ones_like(p, bool)
    p[20:30, 20:30] = 9.0                                              # wrong (far) prompt
    remove = np.zeros_like(v); remove[20:30, 20:30] = True
    prior = np.full_like(p, 2.1)
    out, info = prepare_prompt(p, v, variant="oracle_prior", remove=remove, fill_depth=prior)
    assert out.max() < 9 and np.isclose(out, 2.1).any() and info["removed"] == 100
    with pytest.raises(ValueError):
        prepare_prompt(p, v, variant="oracle_prior")


def test_empty_prompt_is_reported_not_filled():
    out, info = prepare_prompt(np.zeros((4, 4), np.float32), np.zeros((4, 4), bool))
    assert np.isnan(out).all() and info["status"] == "empty_prompt"


def test_cached_prediction_method_evaluates_any_verified_cache(tmp_path):
    from tp.eval.run_eval import evaluate_manifest
    from tp.data.schema import load_sample
    from tp.provenance import ROOT, sha256, write_json
    manifest = ROOT / "data/processed/synthetic_smoke_v1/manifest.json"
    if not manifest.is_file():
        pytest.skip("synthetic smoke data not prepared")
    recs = json.loads(manifest.read_text())["records"]
    cache = tmp_path / "cache"; (cache / "predictions").mkdir(parents=True)
    rows = []
    for r in recs:
        s = load_sample(manifest.parent / r["path"])
        path = cache / "predictions" / f"{s.sensor}__{s.scene}__{s.frame}.npz"
        np.savez_compressed(path, depth=s.gt.copy())                       # a perfect "model"
        import hashlib
        rows.append(dict(dataset=s.dataset, sensor=s.sensor, scene=s.scene, frame=s.frame, split=s.split,
                         rgb_sha256=hashlib.sha256(s.rgb.tobytes()).hexdigest(), sample_npz_sha256=r["sha256"],
                         prediction_path=str(path), prediction_sha256=sha256(path), seconds=0.1))
    write_json(cache / "images.json", rows)
    write_json(cache / "receipt.json", dict(stage="real_dev_verified", model_version=dict(name="fake_model"),
               images_sha256=sha256(cache / "images.json"), frame_count=len(rows),
               inputs=[{k: x[k] for k in ("dataset", "sensor", "scene", "frame", "split", "rgb_sha256", "sample_npz_sha256")} for x in rows]))
    cfg = dict(method="cached_prediction", method_label="fake_model_raw", prediction_cache=str(cache),
               expected_model="fake_model", split="train", manifest=str(manifest), tau=0.05, band=5,
               visualizations=0, experiment_role="engineering_smoke_not_final_test", seed=1)
    evaluate_manifest(cfg, tmp_path / "run", ["t"])
    out = json.loads((tmp_path / "run" / "per_image.json").read_text())
    assert {r["method"] for r in out} == {"fake_model_raw"}
    assert all(r["metrics"]["regions"]["all"]["absrel"] == 0 for r in out)
    with pytest.raises(ValueError, match="model mismatch"):
        evaluate_manifest(dict(cfg, expected_model="other"), tmp_path / "run2", ["t"])


def test_detector_prior_variant_is_deployable_and_removes_flagged_pixels():
    from baselines.promptda import DEPLOYABLE, VARIANTS, prepare_prompt
    assert "detector_prior" in VARIANTS and "detector_prior" in DEPLOYABLE and "oracle_prior" not in DEPLOYABLE
    p = np.full((96, 128), 2.0, np.float32); valid = np.ones_like(p, bool)
    remove = np.zeros_like(valid); remove[:, :64] = True
    out, info = prepare_prompt(p, valid, variant="detector_prior", remove=remove, fill_depth=np.full_like(p, 3.0))
    assert out.shape == (192, 256) and np.isclose(out[:, :100].mean(), 3.0) and np.isclose(out[:, 160:].mean(), 2.0)
