"""Selective remote access to the HAMMER archive without downloading it whole.

The official release is one ZIP (_dataset_processed.zip). ZIP keeps a central
directory at the end, so with HTTP Range requests we can (1) list every member
with its size by fetching only the directory, and (2) extract only the members
we select (scenes / sensors / frame stride), verifying each member's CRC-32
(zipfile does this) and recording SHA256. Nothing else is downloaded.

  python -m scripts.hammer_remote index --output data/source_cache/hammer_index_v1.json
  python -m scripts.hammer_remote pfetch --index data/source_cache/hammer_index_v1.json \
      --select configs/hammer_subset_v1.yaml --dest D:/data/hammer_subset_v1 --workers 6   # parallel (preferred)
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


def open_remote(url, max_download_bytes, retries=8, opener=None, block_size=MIB, cached_blocks=64):
    # fetch uses 8 MiB blocks: selected folders are contiguous in the ZIP, so big
    # blocks cut the request count; excluded data between runs costs little.
    return HTTPRangeReader(url, block_size=block_size, max_cached_blocks=cached_blocks, max_read_bytes=64 * MIB,
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
    with open_remote(index["identity"]["url"], int(args.max_download_gib * GIB) + 256 * MIB, opener=opener,
                     block_size=8 * MIB, cached_blocks=16) as reader:
        return fetch(reader, index, chosen, args.dest, args.dest / "hammer_fetch_receipt.json")




# ---------------------------------------------------------------------------
# Parallel fetch (aria2-style multi-connection, but only for selected members).
# Members are sorted by their local-header offset and packed into contiguous
# batches (<= batch_mib, gaps <= gap_kib merged). Each batch is one HTTP Range
# request; members are parsed from the batch bytes, inflated, CRC/size-checked
# and written atomically. Batches run on `workers` connections.
# ---------------------------------------------------------------------------
import struct
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")   # 30 bytes, signature 0x04034b50


def plan_batches(infos, batch_bytes, gap_bytes):
    """infos: objects with header_offset, compress_size, filename (sorted or not).
    Returns list of (start, end_inclusive, [infos]) covering each member's local
    header (upper-bounded by 30+name+64KiB extra) and data."""
    infos = sorted(infos, key=lambda i: i.header_offset)
    batches, current = [], None
    for info in infos:
        start = info.header_offset
        end = start + 30 + len(info.filename.encode("utf-8")) + 65535 + info.compress_size - 1
        if current and start - current[1] <= gap_bytes and end - current[0] + 1 <= batch_bytes:
            current[1] = max(current[1], end); current[2].append(info)
        else:
            if current:
                batches.append(tuple(current))
            current = [start, end, [info]]
    if current:
        batches.append(tuple(current))
    return batches


def extract_from_blob(blob, blob_start, info):
    off = info.header_offset - blob_start
    fields = LOCAL_HEADER.unpack_from(blob, off)
    if fields[0] != 0x04034B50:
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = fields[9], fields[10]
    data_off = off + 30 + name_len + extra_len
    raw = blob[data_off:data_off + info.compress_size]
    if len(raw) != info.compress_size:
        raise ValueError(f"truncated member data for {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        data = bytes(raw)
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(raw) + inflater.flush()
    else:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise ValueError(f"size/CRC mismatch for {info.filename}")
    return data


def _range_get(url, start, end, identity, opener, timeout, attempts):
    for attempt in range(attempts):
        try:
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            if identity.get("etag"):
                headers["If-Range"] = identity["etag"]
            response = opener(urllib.request.Request(url, headers=headers), timeout=timeout)
            try:
                if getattr(response, "status", None) != 206:
                    raise ValueError("server did not return 206 Partial Content")
                cr = response.headers.get("Content-Range", "")
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr.strip())
                if not m or (int(m.group(1)), int(m.group(2))) != (start, end) or int(m.group(3)) != identity["size"]:
                    raise ValueError(f"unexpected Content-Range {cr!r}")
                if identity.get("etag") and response.headers.get("ETag") not in (None, identity["etag"]):
                    raise ValueError("archive ETag changed")
                body = response.read()
            finally:
                response.close()
            if len(body) != end - start + 1:
                raise IOError("short Range body")
            return body
        except ValueError:
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 60))


def parallel_fetch(url, index, chosen_names, dest, receipt_path, workers=6, batch_mib=64, gap_kib=1024,
                   opener=None, timeout=120, attempts=8, max_bytes=None):
    opener = opener or urllib.request.urlopen
    receipt = json.loads(receipt_path.read_text(encoding="utf8")) if receipt_path.exists() else dict(
        schema="trustprompt.hammer_fetch.v1", identity=index["identity"], members={}, runs=[])
    if receipt["identity"] != index["identity"]:
        raise ValueError("existing receipt belongs to a different archive version")
    done = receipt["members"]
    # Central directory (gives header offsets); also re-checks the live archive identity.
    with open_remote(url, 256 * MIB, opener=opener) as reader:
        if reader.archive_identity != index["identity"]:
            raise ValueError("remote archive identity differs from the index; re-index first")
        with zipfile.ZipFile(reader) as z:
            by_name = {i.filename: i for i in z.infolist()}
    todo = []
    for name in chosen_names:
        info = by_name[name]
        target = safe_target(dest, name)
        if name in done and target.is_file() and target.stat().st_size == info.file_size:
            continue
        todo.append(info)
    batches = plan_batches(todo, batch_mib * MIB, gap_kib * 1024)
    planned = sum(e - s + 1 for s, e, _ in batches)
    if max_bytes is not None and planned > max_bytes:
        raise ValueError(f"planned transfer {planned} B exceeds cap {max_bytes} B")
    print(json.dumps(dict(stage="plan", members=len(todo), batches=len(batches), planned_bytes=planned)), flush=True)
    lock, stats = threading.Lock(), dict(bytes=0, members=0, batches=0)

    def work(batch):
        start, end, infos = batch
        end = min(end, index["identity"]["size"] - 1)
        blob = _range_get(url, start, end, index["identity"], opener, timeout, attempts)
        out = []
        for info in infos:
            data = extract_from_blob(blob, start, info)
            target = safe_target(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(data)
            partial.replace(target)
            out.append((info.filename, dict(size=info.file_size, crc=info.CRC,
                                            sha256=hashlib.sha256(data).hexdigest())))
        return len(blob), out

    started = time.time()
    error = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, b) for b in batches]
        for k, future in enumerate(as_completed(futures)):
            try:
                nbytes, out = future.result()
            except Exception as exc:          # keep finished batches; report and stop at the end
                error = error or exc
                continue
            with lock:
                done.update(out); stats["bytes"] += nbytes; stats["members"] += len(out); stats["batches"] += 1
                if stats["batches"] % 10 == 0 or stats["batches"] == len(batches):
                    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf8")
                    rate = stats["bytes"] / max(time.time() - started, 1e-6) / MIB
                    print(json.dumps(dict(batches=f"{stats['batches']}/{len(batches)}",
                                          gib=round(stats["bytes"] / GIB, 2), mib_per_s=round(rate, 1))), flush=True)
    receipt["runs"].append(dict(finished_at_utc=datetime.now(timezone.utc).isoformat(), mode="parallel",
                                workers=workers, batches_ok=stats["batches"], batches_planned=len(batches),
                                downloaded_bytes=stats["bytes"], error=repr(error) if error else None))
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf8")
    if error:
        raise RuntimeError(f"{len(batches) - stats['batches']} batches failed; re-run to resume") from error
    return receipt


def pfetch_main(argv=None, opener=None):
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True); ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-mib", type=int, default=64); ap.add_argument("--max-download-gib", type=float, default=20)
    args = ap.parse_args(argv)
    index = json.loads(args.index.read_text(encoding="utf8"))
    rule = yaml.safe_load(args.select.read_text(encoding="utf8"))
    chosen = [m["name"] for m in select_members(index["members"], rule)]
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "selection_rule.yaml").write_text(args.select.read_text(encoding="utf8"), encoding="utf8")
    return parallel_fetch(index["identity"]["url"], index, chosen, args.dest, args.dest / "hammer_fetch_receipt.json",
                          workers=args.workers, batch_mib=args.batch_mib, opener=opener,
                          max_bytes=int(args.max_download_gib * GIB))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pfetch":
        pfetch_main(sys.argv[2:])
    else:
        main()
