"""Selective remote access to the HAMMER archive without downloading it whole.

The official release is one ZIP (_dataset_processed.zip). ZIP keeps a central
directory at the end, so with HTTP Range requests we can (1) list every member
with its size by fetching only the directory, and (2) extract only the members
we select (scenes / sensors / frame stride), verifying each member's CRC-32
(zipfile does this) and recording SHA256. Nothing else is downloaded.

  python -m scripts.hammer_remote index --output data/source_cache/hammer_index_v1.json
  python -m scripts.hammer_remote fetch --index data/source_cache/hammer_index_v1.json \
      --select configs/hammer_subset_v1.yaml --dest D:/data/hammer_subset_v1 --max-download-gib 40

`fetch` is resumable: members already on disk with matching size and SHA256
recorded in the receipt are skipped. License: MIT (dataset page, checked 2026-10-09).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

from scripts.fetch_nyu_subset import GIB, MIB, HTTPRangeReader

URL = "https://campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip"


def open_remote(url, max_download_bytes, retries=8, opener=None):
    return HTTPRangeReader(url, block_size=MIB, max_cached_blocks=64, max_read_bytes=64 * MIB,
                           timeout=60, max_total_download_bytes=max_download_bytes,
                           retry_attempts=retries, opener=opener)


def build_index(reader):
    with zipfile.ZipFile(reader) as z:
        members = [dict(name=i.filename, size=i.file_size, compressed=i.compress_size,
                        method=i.compress_type, crc=i.CRC, is_dir=i.is_dir())
                   for i in z.infolist()]
    files = [m for m in members if not m["is_dir"]]
    summary = {}
    for m in files:
        parts = PurePosixPath(m["name"]).parts
        key = "/".join(parts[:4]) if len(parts) > 4 else "/".join(parts[:-1])
        entry = summary.setdefault(key, dict(files=0, bytes=0, compressed=0))
        entry["files"] += 1; entry["bytes"] += m["size"]; entry["compressed"] += m["compressed"]
    return dict(schema="trustprompt.hammer_index.v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                identity=reader.archive_identity, member_count=len(members), file_count=len(files),
                total_bytes=sum(m["size"] for m in files), total_compressed=sum(m["compressed"] for m in files),
                index_downloaded_bytes=reader.downloaded_bytes, members=members,
                summary_by_prefix=dict(sorted(summary.items())))


def frame_number(name):
    found = re.findall(r"(\d+)", PurePosixPath(name).stem)
    return int(found[-1]) if found else None


def select_members(members, rule):
    """rule: include globs, exclude globs, optional frame_stride for files whose
    stem ends in a number (frame-indexed), applied after globs. Non-frame files
    (intrinsics, poses) matching include are always kept."""
    chosen = []
    stride = int(rule.get("frame_stride", 1))
    for m in members:
        if m["is_dir"]:
            continue
        name = m["name"]
        if not any(fnmatch.fnmatch(name, g) for g in rule["include"]):
            continue
        if any(fnmatch.fnmatch(name, g) for g in rule.get("exclude", [])):
            continue
        n = frame_number(name)
        is_frame = n is not None and PurePosixPath(name).suffix.lower() in {".png", ".jpg", ".exr", ".npy", ".tiff", ".tif"}
        if is_frame and n % stride:
            continue
        chosen.append(m)
    return chosen


def safe_target(dest, name):
    target = (dest / PurePosixPath(name)).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise ValueError(f"unsafe member path {name!r}")
    return target


def fetch(reader, index, chosen, dest, receipt_path):
    if reader.archive_identity != index["identity"]:
        raise ValueError("remote archive identity differs from the index; re-index first")
    receipt = json.loads(receipt_path.read_text(encoding="utf8")) if receipt_path.exists() else dict(
        schema="trustprompt.hammer_fetch.v1", identity=index["identity"], members={}, runs=[])
    if receipt["identity"] != index["identity"]:
        raise ValueError("existing receipt belongs to a different archive version")
    done = receipt["members"]
    with zipfile.ZipFile(reader) as z:
        for k, m in enumerate(chosen):
            target = safe_target(dest, m["name"])
            prior = done.get(m["name"])
            if prior and target.is_file() and target.stat().st_size == m["size"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            digest = hashlib.sha256()
            with z.open(m["name"]) as source, partial.open("wb") as sink:   # CRC checked at EOF
                while chunk := source.read(4 * MIB):
                    digest.update(chunk); sink.write(chunk)
            partial.replace(target)
            done[m["name"]] = dict(size=m["size"], crc=m["crc"], sha256=digest.hexdigest())
            if k % 50 == 0:
                receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf8")
                print(json.dumps(dict(progress=k + 1, of=len(chosen), downloaded_bytes=reader.downloaded_bytes)), flush=True)
    receipt["runs"].append(dict(finished_at_utc=datetime.now(timezone.utc).isoformat(), selected=len(chosen),
                                downloaded_bytes=reader.downloaded_bytes, retry_count=reader.retry_count))
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf8")
    return receipt


def main(argv=None, opener=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--url", default=URL); a.add_argument("--max-download-mib", type=int, default=256)
    b = sub.add_parser("fetch"); b.add_argument("--index", type=Path, required=True)
    b.add_argument("--select", type=Path, required=True); b.add_argument("--dest", type=Path, required=True)
    b.add_argument("--max-download-gib", type=float, default=40); b.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "index":
        if args.output.exists():
            raise FileExistsError(args.output)
        with open_remote(args.url, args.max_download_mib * MIB, opener=opener) as reader:
            index = build_index(reader)
        index["command"] = [sys.executable, *sys.argv]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(index, indent=1), encoding="utf8")
        print(json.dumps({k: index[k] for k in ("file_count", "total_bytes", "total_compressed", "index_downloaded_bytes")}))
        return index
    import yaml
    index = json.loads(args.index.read_text(encoding="utf8"))
    rule = yaml.safe_load(args.select.read_text(encoding="utf8"))
    chosen = select_members(index["members"], rule)
    plan = dict(selected_files=len(chosen), selected_bytes=sum(m["size"] for m in chosen),
                selected_compressed=sum(m["compressed"] for m in chosen))
    print(json.dumps(plan))
    if args.dry_run:
        return plan
    if plan["selected_compressed"] > args.max_download_gib * GIB:
        raise ValueError("selection exceeds --max-download-gib; narrow the selection")
    args.dest.mkdir(parents=True, exist_ok=True)
    with open_remote(index["identity"]["url"], int(args.max_download_gib * GIB) + 256 * MIB, opener=opener) as reader:
        return fetch(reader, index, chosen, args.dest, args.dest / "hammer_fetch_receipt.json")


if __name__ == "__main__":
    main()
