"""Workspace-owned temporary files avoid restrictive Windows temp ACLs."""
from pathlib import Path
import shutil
import uuid
import pytest


@pytest.fixture
def tmp_path():
    parent = Path(__file__).resolve().parents[1] / "results" / "_test_temp"
    path = parent / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        resolved = path.resolve()
        assert resolved.parent == parent.resolve() and resolved.is_relative_to(parent.resolve())
        shutil.rmtree(resolved)
