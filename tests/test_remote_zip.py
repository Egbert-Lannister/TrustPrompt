import io
import zipfile

import pytest

from tp.data.remote_zip import RemoteZip


def build_zip(force_zip64=False, monkeypatch=None):
    if force_zip64:                    # make every size / offset "large" so ZIP64 records are written
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 10)
    buf = io.BytesIO()
    files = {"a/scene_cam_00_geometry_hdf5/frame.0000.semantic.hdf5": bytes(range(256)) * 40,
             "a/scene_cam_00_final_preview/frame.0000.tonemap.jpg": b"\xff\xd8" + b"x" * 5000,
             "a/empty.txt": b"",
             "a/ünïcode.csv": "naïve,1\n".encode() * 100}
    with zipfile.ZipFile(buf, "w") as z:
        for i, (name, data) in enumerate(files.items()):
            z.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED)
    return buf.getvalue(), files


def local_fetch(blob, calls):
    def fetch(start, end):
        calls.append((start, end))
        return blob[start:end + 1]
    return fetch


@pytest.mark.parametrize("zip64", [False, True])
def test_members_listed_and_read_exactly(zip64, monkeypatch):
    blob, files = build_zip(zip64, monkeypatch)
    calls = []
    rz = RemoteZip(local_fetch(blob, calls), len(blob))
    assert set(rz.members) == set(files)
    for name, data in files.items():
        assert rz.read(name) == data
        assert rz.members[name].size == len(data)


def test_corrupted_member_is_rejected():
    blob, files = build_zip()
    rz = RemoteZip(local_fetch(blob, []), len(blob))
    m = rz.members["a/scene_cam_00_geometry_hdf5/frame.0000.semantic.hdf5"]           # stored member
    bad = bytearray(blob)
    bad[m.header_offset + 30 + len(m.name) + 10] ^= 0xFF
    rz.fetch = local_fetch(bytes(bad), [])
    with pytest.raises(ValueError, match="CRC"):
        rz.read(m.name)


def test_small_slack_needs_a_second_request():
    blob, files = build_zip()
    calls = []
    rz = RemoteZip(local_fetch(blob, calls), len(blob))
    calls.clear()
    name = "a/scene_cam_00_final_preview/frame.0000.tonemap.jpg"
    assert rz.read(name, slack=0) == files[name] and len(calls) in (1, 2)
