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
