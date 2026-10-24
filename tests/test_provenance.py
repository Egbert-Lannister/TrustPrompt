import hashlib
import json

import pytest

from tp.provenance import deployment_origin


def make_deployment(root):
    (root / "source.py").write_bytes(b"pass\n")
    (root / "deployment_working_tree.diff").write_bytes(b"diff\n")
    manifest = dict(schema="trustprompt.deployment.v1", base_commit="a" * 40,
                    source_sha256={"source.py": hashlib.sha256(b"pass\n").hexdigest()},
                    diff_sha256=hashlib.sha256(b"diff\n").hexdigest())
    (root / "deployment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_absent_deployment_is_normal_checkout(tmp_path):
    assert deployment_origin(tmp_path) is None


def test_deployment_verifies_code_and_patch(tmp_path):
    expected = make_deployment(tmp_path)
    assert deployment_origin(tmp_path) == expected
    (tmp_path / "source.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="source mismatch"):
        deployment_origin(tmp_path)


def test_deployment_rejects_changed_patch(tmp_path):
    make_deployment(tmp_path)
    (tmp_path / "deployment_working_tree.diff").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="patch mismatch"):
        deployment_origin(tmp_path)


def test_deployment_rejects_traversal_before_read(tmp_path):
    manifest = make_deployment(tmp_path)
    manifest["source_sha256"] = {"../outside.py": "0" * 64}
    (tmp_path / "deployment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        deployment_origin(tmp_path)
