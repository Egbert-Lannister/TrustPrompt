"""Workspace-owned temporary files avoid restrictive Windows temp ACLs.

TP_TEST_TMP overrides the parent directory (used on hosts where the project
folder is mounted without delete permission, e.g. the Linux workspace).
MKL runs sequentially in tests, as in every script that loads torch: Anaconda's MKL brings its own OpenMP runtime,
which aborts the process next to torch's (OMP Error #15, docs/progress.md 2026-10-14).
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from pathlib import Path
import shutil
import uuid
import pytest


@pytest.fixture
def tmp_path():
    override = os.environ.get("TP_TEST_TMP")
    parent = Path(override) if override else Path(__file__).resolve().parents[1] / "results" / "_test_temp"
    path = parent / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        resolved = path.resolve()
        assert resolved.parent == parent.resolve() and resolved.is_relative_to(parent.resolve())
        shutil.rmtree(resolved)
