"""Build the explicitly scoped public-code P1 deployment archive (no local data).

python -m scripts.build_p1_payload --output results/autodl_p1_20261024/p1_source.tar.gz
The archive includes a base-commit receipt, working diff, and exact deployed hashes.
Text is normalized to LF for Linux; both local and deployed hashes are recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = """
pyproject.toml requirements-local.txt
baselines/__init__.py baselines/completion.py
scripts/__init__.py scripts/setup_baselines.py scripts/hypersim_fetch.py scripts/train_omnidc.py
scripts/linux/p1_setup.sh scripts/linux/p1_run.sh
tp/__init__.py tp/provenance.py
tp/data/__init__.py tp/data/hypersim.py tp/data/remote_zip.py
tp/omnidc/__init__.py tp/omnidc/data.py tp/omnidc/gate.py tp/omnidc/gates.py tp/omnidc/losses.py tp/omnidc/model.py tp/omnidc/onesided.py
tp/sim/__init__.py tp/sim/corrupt.py tp/sim/hypersim_condition.py tp/sim/mirror_render.py tp/sim/sensor.py
tp/trust/__init__.py tp/trust/model.py
tests/__init__.py tests/conftest.py tests/test_omnidc_train.py tests/test_omnidc_gate.py tests/test_hypersim.py tests/test_hypersim_condition.py tests/test_onesided.py tests/test_remote_zip.py tests/test_provenance.py
configs/baselines/registry.json configs/hypersim/selection_v1.json configs/hypersim/training_v1.yaml configs/omnidc/p1.yaml
""".split()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        ap.error("Deployment archives must stay under this project's results directory")
    if output.exists():
        ap.error("Refusing to replace an existing deployment archive")
    payload, hashes, local_hashes = {}, {}, {}
    for rel in sorted(FILES):
        raw = (ROOT / rel).read_bytes()
        data = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
        payload[rel] = data
        hashes[rel] = hashlib.sha256(data).hexdigest()
        local_hashes[rel] = hashlib.sha256(raw).hexdigest()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args])
    diff = git("diff", "HEAD", "--", *FILES)
    payload["deployment_working_tree.diff"] = diff
    manifest = dict(schema="trustprompt.deployment.v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    base_commit=git("rev-parse", "HEAD").decode().strip(),
                    source_status=git("status", "--porcelain=v1", "--", *FILES).decode(),
                    source_sha256=hashes, local_source_sha256=local_hashes,
                    diff_sha256=hashlib.sha256(diff).hexdigest(),
                    diff_scope="Tracked file changes only; the archive and per-file hashes include new files too",
                    scope="P1 source/config/tests only; no private data, local weights, credentials, results or Git history",
                    newline_conversion="UTF-8 text normalized from CRLF to LF")
    payload["deployment_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for rel, data in payload.items():
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size,
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest(), source_files=len(hashes))))


if __name__ == "__main__":
    main()
