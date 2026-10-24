import io
import urllib.error
import zipfile

import pytest

from tp.data.remote_zip import RangeProtocolError, RemoteZip, http_range_fetcher


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


class Response:
    def __init__(self, status, headers, body=b"", forbid_body=False):
        self.status, self.headers, self.body = status, headers, body
        self.forbid_body, self.read_sizes = forbid_body, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size=-1):
        assert not self.forbid_body, "Rejected response body must not be read"
        assert size >= 0, "Unbounded HTTP read is forbidden"
        self.read_sizes.append(size)
        return self.body[:size]


def install_responses(monkeypatch, responses):
    calls = []
    responses = iter(responses)
    def urlopen(request, timeout):
        calls.append((request.get_method(), request.headers, timeout))
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr("tp.data.remote_zip.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("tp.data.remote_zip.time.sleep", lambda _: None)
    return calls


def head():
    return Response(200, {"Content-Length": "100"}, forbid_body=True)


def test_valid_range_is_bounded_and_clamped_to_archive(monkeypatch):
    response = Response(206, {"Content-Length": "3", "Content-Range": "bytes 97-99/100"}, b"abc")
    calls = install_responses(monkeypatch, [head(), response])
    fetch, size = http_range_fetcher("https://example.org/scene.zip", timeout=7)
    assert size == 100 and fetch(97, 1000) == b"abc"
    assert response.read_sizes == [4]
    assert calls[1] == ("GET", {"Accept-encoding": "identity", "Range": "bytes=97-99"}, 7)
    for start, end in [(-1, 2), (100, 101), (3, 2), (False, 2)]:
        with pytest.raises(ValueError, match="Range"):
            fetch(start, end)
    assert len(calls) == 2


@pytest.mark.parametrize("status,headers", [
    (200, {"Content-Length": "5000000000"}),
    (206, {"Content-Length": "3"}),
    (206, {"Content-Length": "3", "Content-Range": "bytes 10-12/*"}),
    (206, {"Content-Length": "3", "Content-Range": "bytes 11-13/100"}),
    (206, {"Content-Length": "3", "Content-Range": "bytes 10-12/101"}),
    (206, {"Content-Length": "4", "Content-Range": "bytes 10-12/100"}),
    (206, {"Content-Range": "bytes 10-12/100"}),
    (206, {"Content-Length": "3", "Content-Range": "bytes 10-12/100", "Content-Encoding": "gzip"}),
])
def test_bad_range_headers_never_read_or_retry_body(monkeypatch, status, headers):
    response = Response(status, headers, forbid_body=True)
    calls = install_responses(monkeypatch, [head(), response])
    fetch, _ = http_range_fetcher("https://example.org/scene.zip")
    with pytest.raises(RangeProtocolError):
        fetch(10, 12)
    assert len(calls) == 2 and response.read_sizes == []


@pytest.mark.parametrize("body", [b"ab", b"abcdextra"])
def test_truncated_or_oversized_range_is_rejected_after_bounded_read(monkeypatch, body):
    response = Response(206, {"Content-Length": "3", "Content-Range": "bytes 10-12/100"}, body)
    calls = install_responses(monkeypatch, [head(), response])
    fetch, _ = http_range_fetcher("https://example.org/scene.zip")
    with pytest.raises(RangeProtocolError, match="body"):
        fetch(10, 12)
    assert len(calls) == 2 and response.read_sizes == [4]


def test_temporary_network_timeout_retries_only_the_same_range(monkeypatch):
    response = Response(206, {"Content-Length": "3", "Content-Range": "bytes 10-12/100"}, b"abc")
    calls = install_responses(monkeypatch, [head(), TimeoutError("temporary"), response])
    fetch, _ = http_range_fetcher("https://example.org/scene.zip", retries=2)
    assert fetch(10, 12) == b"abc"
    assert calls[1] == calls[2] and len(calls) == 3


def test_head_failure_has_no_get_fallback_and_timeout_is_bounded(monkeypatch):
    error = urllib.error.HTTPError("https://example.org/scene.zip", 405, "HEAD unavailable", {}, None)
    calls = install_responses(monkeypatch, [error])
    with pytest.raises(urllib.error.HTTPError):
        http_range_fetcher("https://example.org/scene.zip")
    assert len(calls) == 1 and calls[0][0] == "HEAD"
    calls = install_responses(monkeypatch, [TimeoutError(), TimeoutError()])
    with pytest.raises(TimeoutError):
        http_range_fetcher("https://example.org/scene.zip", retries=2, timeout=3)
    assert len(calls) == 2 and all(c[0] == "HEAD" and c[2] == 3 for c in calls)
