import json

import numpy as np
import pytest

from tp import formal
from tp.provenance import sha256


def _freeze(tmp_path, monkeypatch, content=b"frozen"):
    monkeypatch.setattr(formal, "ROOT", tmp_path)
    f = tmp_path / "artefact.bin"
    f.write_bytes(content)
    spec = tmp_path / "freeze.json"
    spec.write_text(json.dumps(dict(schema="trustprompt.freeze.v1", frozen_at_commit="abc",
                                    files={"artefact.bin": sha256(f)})), encoding="utf-8")
    return f, spec


def test_guard_passes_and_fails(tmp_path, monkeypatch):
    f, spec = _freeze(tmp_path, monkeypatch)
    rec = formal.guard(spec)
    assert rec["verified_files"] == 1 and rec["frozen_at_commit"] == "abc"
    f.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        formal.guard(spec)
    f.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        formal.guard(spec)


def test_guard_without_freeze_file(tmp_path):
    with pytest.raises(RuntimeError, match="not frozen"):
        formal.guard(tmp_path / "none.json")


def test_select_records_keeps_test_locked_by_default():
    from scripts.cache_completion import select_records
    recs = [dict(split="test", dataset="hammer", scene="scene12")]
    with pytest.raises(ValueError, match="locked"):
        select_records(recs, {"test"}, "all")
    assert select_records(recs, {"test"}, "all", allow_test=True) == recs


def test_run_eval_refuses_test_without_formal_role(tmp_path):
    from tp.eval.run_eval import evaluate_manifest
    with pytest.raises(ValueError, match="locked"):
        evaluate_manifest(dict(split="test", experiment_role="development_diagnostic_not_final_test"), tmp_path / "x", [])


def test_compare_can_force_frame_blocks():
    from tp.eval.family import compare
    rows = lambda off: [dict(dataset="hammer", sensor="d435", scene=f"scene{s}", frame=f"traj1_1_{i:06d}", split="test",
                             metrics=dict(regions=dict(C=dict(absrel=0.1 + off + 0.001 * i, count=10, status="ok"))))
                        for s in (12, 13, 14) for i in range(0, 80, 5)]
    a = compare(rows(0.0), rows(-0.05), dataset="hammer", n_resamples=200, seed=0)
    b = compare(rows(0.0), rows(-0.05), dataset="hammer", n_resamples=200, seed=0, unit="frame_block")
    assert a["unit"] == "scene" and b["unit"] == "frame_block"


def test_test_driver_plans_every_run():
    from scripts.run_test_eval import eval_jobs, steps
    s = steps(False)
    assert sum(1 for x in s if x[0].startswith("9 ")) == 5 * 2 * 3
    assert sum(1 for x in s if x[0].startswith("8 ")) == 3 * 2
    j = eval_jobs()
    assert len(j) == 2 * (3 + 15) and all("--split" in a and "test" in a for _, a in j)
