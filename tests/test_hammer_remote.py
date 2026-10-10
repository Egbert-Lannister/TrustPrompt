"""HAMMER selective remote extraction against an in-memory ZIP served by fake Range responses."""
import io
import json
import re
import zipfile

import pytest

from scripts.hammer_remote import main, select_members


class FakeResponse:
    def __init__(self, data, start, end, total):
        self.status = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(end - start + 1),
                        "ETag": '"v1"', "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"}
        self._body = io.BytesIO(data[start:end + 1])

    def read(self, n=-1):
        return self._body.read(n)

    def close(self):
        pass


def make_opener(blob, log):
    def opener(request, timeout=None):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        log.append(end - start + 1)
        return FakeResponse(blob, start, min(end, len(blob) - 1), len(blob))
    return opener


def archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for scene in ("scene2", "scene12"):
            z.writestr(f"_dataset_processed/{scene}/d435/intrinsics.txt", "fx 1")
            for i in range(6):
                z.writestr(f"_dataset_processed/{scene}/d435/depth/{i:06d}.png", bytes([i]) * 5000)
                z.writestr(f"_dataset_processed/{scene}/polarization/rgb/{i:06d}.png", bytes([i + 7]) * 3000)
    return buffer.getvalue()


def test_index_then_selective_resumable_fetch(tmp_path):
    blob, log = archive(), []
    opener = make_opener(blob, log)
    index_path = tmp_path / "index.json"
    index = main(["index", "--output", str(index_path), "--url", "https://example.org/h.zip"], opener=opener)
    assert index["file_count"] == 26
    rule = tmp_path / "rule.yaml"
    rule.write_text("include: ['*/scene2/d435/*']\nframe_stride: 3\n")
    dest = tmp_path / "out"
    main(["fetch", "--index", str(index_path), "--select", str(rule), "--dest", str(dest)], opener=opener)
    got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert got == ["_dataset_processed/scene2/d435/depth/000000.png", "_dataset_processed/scene2/d435/depth/000003.png",
                   "_dataset_processed/scene2/d435/intrinsics.txt", "hammer_fetch_receipt.json"]
    assert (dest / "_dataset_processed/scene2/d435/depth/000003.png").read_bytes() == bytes([3]) * 5000
    receipt = json.loads((dest / "hammer_fetch_receipt.json").read_text())
    assert len(receipt["members"]) == 3
    before = len(log)
    main(["fetch", "--index", str(index_path), "--select", str(rule), "--dest", str(dest)], opener=opener)
    assert len(json.loads((dest / "hammer_fetch_receipt.json").read_text())["runs"]) == 2
    assert len(log) - before <= 3          # only identity probe/directory reads, no member re-download


def test_select_rejects_by_exclude_and_keeps_non_frame_files():
    members = [dict(name=n, size=1, compressed=1, is_dir=False) for n in
               ["a/scene12/l515/depth/000001.png", "a/scene12/l515/pose.txt", "a/scene12/l515/depth/000002.png"]]
    chosen = select_members(members, {"include": ["*/l515/*"], "exclude": ["*000002*"], "frame_stride": 2})
    assert [m["name"] for m in chosen] == ["a/scene12/l515/pose.txt"]


def test_parallel_fetch_batches_extracts_verifies_and_resumes(tmp_path):
    from scripts.hammer_remote import pfetch_main
    blob, log = archive(), []
    opener = make_opener(blob, log)
    index_path = tmp_path / "index.json"
    main(["index", "--output", str(index_path), "--url", "https://example.org/h.zip"], opener=opener)
    rule = tmp_path / "rule.yaml"
    rule.write_text("include: ['*/d435/*', '*/polarization/rgb/*']\nexclude: ['*scene12*']\nframe_stride: 2\n")
    dest = tmp_path / "par"
    receipt = pfetch_main(["--index", str(index_path), "--select", str(rule), "--dest", str(dest),
                           "--workers", "3", "--batch-mib", "1"], opener=opener)
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.png"))
    assert files == [f"_dataset_processed/scene2/{d}/{i:06d}.png" for d in ("d435/depth", "polarization/rgb") for i in (0, 2, 4)]
    assert (dest / "_dataset_processed/scene2/polarization/rgb/000004.png").read_bytes() == bytes([11]) * 3000
    assert len(receipt["members"]) == 7 and receipt["runs"][-1]["error"] is None
    before = len(log)
    pfetch_main(["--index", str(index_path), "--select", str(rule), "--dest", str(dest)], opener=opener)
    member_requests = [n for n in log[before:] if n > 1024 * 1024]   # only block reads for the directory
    assert not member_requests or len(log) - before < 10


def test_extract_rejects_corrupted_member():
    import zlib, zipfile, io
    from scripts.hammer_remote import extract_from_blob
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.bin", b"x" * 4000)
    data = bytearray(buffer.getvalue())
    info = zipfile.ZipFile(io.BytesIO(bytes(data))).infolist()[0]
    good = extract_from_blob(bytes(data), 0, info)
    assert good == b"x" * 4000
    data[info.header_offset + 30 + len("a.bin") + 5] ^= 0xFF          # flip a compressed byte
    with pytest.raises(Exception):
        extract_from_blob(bytes(data), 0, info)
