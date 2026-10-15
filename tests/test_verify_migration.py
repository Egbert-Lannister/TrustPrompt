"""verify_migration: size comparison and per-prediction SHA256 against images.json."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tp.provenance import sha256


def _cache(root):
    (root / "c1" / "predictions").mkdir(parents=True)
    rows = []
    for i in range(3):
        f = root / "c1" / "predictions" / f"p{i}.npz"
        np.savez_compressed(f, depth=np.full((4, 4), i, np.float32))
        rows.append(dict(prediction_path=f"E:\\old\\c1\\predictions\\p{i}.npz", prediction_sha256=sha256(f)))
    (root / "c1" / "images.json").write_text(json.dumps(rows)); (root / "c1" / "receipt.json").write_text(json.dumps(dict(stage="real_dev_verified")))


def _run(pairs, out):
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, "-m", "scripts.verify_migration", "--output", str(out)] + sum((["--pair", p] for p in pairs), [])
    return subprocess.run(args, cwd=root, capture_output=True, text=True)


def test_identical_copy_verifies_and_corruption_is_caught(tmp_path):
    import shutil
    src = tmp_path / "src"; _cache(src)
    dst = tmp_path / "dst"; shutil.copytree(src, dst)
    r = _run([f"{src}={dst}"], tmp_path / "v1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "v1" / "receipt.json").read_text())["stage"] == "migration_verified"
    f = dst / "c1" / "predictions" / "p1.npz"; b = bytearray(f.read_bytes()); b[-5] ^= 0xFF; f.write_bytes(bytes(b))
    (dst / "c1" / "predictions" / "p2.npz").unlink()
    r = _run([f"{src}={dst}"], tmp_path / "v2")
    rep = json.loads((tmp_path / "v2" / "report.json").read_text())[f"{src}={dst}"]
    assert r.returncode == 1 and rep["tree"]["missing_count"] == 1 and rep["caches"]["c1"]["hash_mismatch"] == 2
