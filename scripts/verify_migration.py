"""Verify a copy of the E: data into the project (2026-10-14 migration; E: stays untouched as a backup).

  python -m scripts.verify_migration --pair SRC=DST [--pair ...] --output results/migration_e_to_d_v1

For every pair: every source file must exist at the destination with the same size. For every prediction
cache found under a destination (folder with images.json + predictions/), each prediction file's SHA256
must equal the value its images.json recorded when it was written. Checkpoints listed in KNOWN_CKPT must
match the SHA256 recorded by the runs that used them. Writes report.json and a receipt
(stage migration_verified only when nothing is missing or different).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import sys

from tp.provenance import ROOT, sha256, start_run, write_json

KNOWN_CKPT = {
    "prompt-depth-anything-vitl": "4b3a3e0a60ea68e719cad3b2e22da8c95ce78f9559ee04824c064a2a06b0f10f",
    "prompt-depth-anything-vits": "51b99fef5e3c467f3ccd12f2f4687220e7f1e88f6a07fe21f7a66adffca2af99",
    "prompt-depth-anything-vits-transparent": "4202b8167c4877f5415a98969f9077ad60447e6424164aa1336fab9e741c15e2",
}


def compare_tree(src, dst):
    missing, size_diff, n, total = [], [], 0, 0
    for dirpath, _, files in os.walk(src):
        for f in files:
            s = Path(dirpath) / f; rel = s.relative_to(src); d = dst / rel
            n += 1; sz = s.stat().st_size; total += sz
            if not d.is_file():
                missing.append(str(rel))
            elif d.stat().st_size != sz:
                size_diff.append(str(rel))
    return dict(files=n, bytes=total, missing=missing[:50], missing_count=len(missing),
                size_diff=size_diff[:50], size_diff_count=len(size_diff))


def check_caches(dst):
    out = {}
    for images in sorted(dst.rglob("images.json")):
        cache = images.parent
        if not (cache / "predictions").is_dir():
            continue
        rows = json.loads(images.read_text(encoding="utf-8"))
        def good(r):
            f = cache / "predictions" / PureWindowsPath(r["prediction_path"]).name
            return f.is_file() and sha256(f) == r["prediction_sha256"]
        bad = [r["prediction_path"] for r in rows if not good(r)]
        stage = json.loads((cache / "receipt.json").read_text(encoding="utf-8")).get("stage") if (cache / "receipt.json").is_file() else None
        out[str(cache.relative_to(dst))] = dict(stage=stage, predictions=len(rows), hash_mismatch=len(bad), examples=bad[:5])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, metavar="SRC=DST")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output); out = out if out.is_absolute() else ROOT / out
    receipt = start_run(out, dict(pairs=a.pair, purpose="verify E: -> D: copy"), [sys.executable, "-m", "scripts.verify_migration", *sys.argv[1:]], 0)
    report, ok = {}, True
    try:
        for p in a.pair:
            src, dst = (Path(x) for x in p.split("=", 1))
            if not src.exists():
                report[p] = dict(status="source_missing_skipped"); continue
            if src.is_file():                      # single file (e.g. Mirror3D metadata copied from GLARE's folder)
                same = dst.is_file() and dst.stat().st_size == src.stat().st_size and sha256(dst) == sha256(src)
                report[p] = dict(file=True, identical=same); ok &= same
                print(p, "identical" if same else "DIFFERENT", flush=True); continue
            t = compare_tree(src, dst)
            r = dict(tree=t)
            ok &= t["missing_count"] == 0 and t["size_diff_count"] == 0
            caches = check_caches(dst)
            if caches:
                r["caches"] = caches
                ok &= all(c["hash_mismatch"] == 0 for c in caches.values())
            ck = {}
            for name, want in KNOWN_CKPT.items():
                f = dst / name / "model.ckpt"
                if f.is_file():
                    ck[name] = sha256(f) == want
            if ck:
                r["checkpoints_match_recorded_sha256"] = ck; ok &= all(ck.values())
            report[p] = r
            print(p, "files", t["files"], "missing", t["missing_count"], "size_diff", t["size_diff_count"],
                  "caches", len(caches), "cache_hash_mismatch", sum(c["hash_mismatch"] for c in caches.values()), "ckpt", ck, flush=True)
        write_json(out / "report.json", report)
        receipt.update(stage="migration_verified" if ok else "migration_mismatch",
                       completed_at_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha256(out / "report.json"))
    except BaseException as error:
        receipt.update(stage="failed", error=repr(error)); raise
    finally:
        write_json(out / "receipt.json", receipt)
    print("MIGRATION VERIFIED" if ok else "MIGRATION MISMATCH: see report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
