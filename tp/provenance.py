"""Small, explicit experiment receipts. No environment-variable collection."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str | None:
    p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8")
    return p.stdout.strip() if p.returncode == 0 else None


def runtime() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "h5py", "Pillow", "matplotlib", "PyYAML", "pytest", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions)


def start_run(output: Path, config: dict, command: list[str], seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for folder in ("tp", "baselines", "scripts", "tests", "configs"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in {".py", ".yaml", ".json"} and "__pycache__" not in p.parts)
    files += [ROOT / "pyproject.toml", ROOT / "requirements-local.txt"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for rel in hashes:
            z.write(ROOT / rel, rel)
    receipt = dict(schema="trustprompt.run.v1", started_at_utc=datetime.now(timezone.utc).isoformat(),
                   seed=seed, command=command, shell_command=subprocess.list2cmdline(command),
                   git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain=v1"),
                   source_sha256=hashes, runtime=runtime(), model_version=None, config=config,
                   stage="started")
    write_json(output / "config.json", config)
    write_json(output / "receipt.json", receipt)
    (output / "working_tree.diff").write_text(git("diff", "HEAD", "--") or git("diff", "--") or "", encoding="utf-8")
    (output / "command.txt").write_text(receipt["shell_command"] + "\n", encoding="utf-8")
    return receipt
