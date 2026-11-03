"""Gate for the one formal test evaluation of method v1 (docs/plan_test.md, docs/freeze_method_v1.md).

Every script that may read a test sample takes `--formal-test`; with it, `guard()` must pass first. The guard checks
that every file listed in configs/freeze_method_v1.json (frozen method artefacts, method library code, the test plan
and the test comparison family) still has the SHA256 recorded at the freeze. Without the flag the test split stays
locked exactly as before.
"""
from __future__ import annotations

import json

from tp.provenance import ROOT, sha256

FREEZE_FILE = ROOT / "configs" / "freeze_method_v1.json"
FORMAL_ROLES = ("formal_main_result", "formal_test_registered_z")


def guard(freeze_file=None) -> dict:
    """Raise unless the method v1 freeze still holds; return a record for the receipt."""
    path = freeze_file or FREEZE_FILE
    if not path.is_file():
        raise RuntimeError(f"{path} missing: method v1 is not frozen, the test split stays locked")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "trustprompt.freeze.v1":
        raise RuntimeError("unknown freeze schema")
    bad = []
    for rel, digest in spec["files"].items():
        p = ROOT / rel
        if not p.is_file():
            bad.append(f"{rel} (missing)")
        elif sha256(p) != digest:
            bad.append(f"{rel} (changed)")
    if bad:
        raise RuntimeError("method v1 freeze broken, test refused: " + ", ".join(bad))
    return dict(freeze_file=path.relative_to(ROOT).as_posix(), freeze_file_sha256=sha256(path),
                frozen_at_commit=spec.get("frozen_at_commit"), verified_files=len(spec["files"]))
