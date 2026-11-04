"""Mirror3D NYU annotations (official release, MIT) -> data/source_cache/mirror3d/{nyu.zip, nyu.csv}.

  python -m scripts.fetch_mirror3d_meta

The development set was prepared from these two files (receipt of data/processed/mirror3d_dev_v1, then read from
E:/GLARE_Data/first_surface/mirror3d/). The test samples must use the same annotation release, so each file is
accepted only if its SHA256 equals the development copy; a mismatch stops with an error and nothing is kept.
A file already present with the right hash is not downloaded again. Reads no test sample.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "source_cache" / "mirror3d"
FILES = {
    "nyu.zip": dict(url="https://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/nyu.zip",
                    sha256="c7fa506afe2490bbb50380e7abdbd1bcc61147c48d662a21a706935bb8613bc3"),
    "nyu.csv": dict(url="https://raw.githubusercontent.com/3dlg-hcvc/mirror3d/"
                        "1ce8e337e388fd61d5af016be883341add02e0a6/docs/metadata/nyu.csv",
                    sha256="d932946c160524e335b9f4be1e028f1a8bbff46d68f0950c47f975b6297bc4f6"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ok():
    return all((DEST / n).is_file() and sha256(DEST / n) == s["sha256"] for n, s in FILES.items())


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rec = dict(created=datetime.now().isoformat(timespec="seconds"), files={})
    for name, spec in FILES.items():
        p = DEST / name
        if p.is_file() and sha256(p) == spec["sha256"]:
            print(f"{name}: present, hash matches the development copy", flush=True)
            rec["files"][name] = dict(spec, downloaded=False)
            continue
        tmp = p.with_suffix(p.suffix + ".part")
        print(f"{name}: downloading {spec['url']}", flush=True)
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "TrustPrompt-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while b := r.read(1 << 20):
                f.write(b)
        got = sha256(tmp)
        if got != spec["sha256"]:
            tmp.unlink()
            raise SystemExit(f"{name}: SHA256 {got} differs from the development copy {spec['sha256']}; "
                             "the release changed, so the test is not prepared. Nothing was kept.")
        tmp.replace(p)
        print(f"{name}: {p.stat().st_size / 2 ** 20:.1f} MiB, hash matches the development copy", flush=True)
        rec["files"][name] = dict(spec, downloaded=True, bytes=p.stat().st_size)
    (DEST / "receipt.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("MIRROR3D ANNOTATIONS OK", flush=True)


if __name__ == "__main__":
    main()
