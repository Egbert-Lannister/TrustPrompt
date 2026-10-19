"""Read single members of a large remote ZIP archive with HTTP range requests (Hypersim ships one ZIP per scene,
several GB each; we need a few files per frame).

A `RemoteZip` only needs `fetch(start, end) -> bytes` (inclusive byte range) and the archive size, so the
parsing is tested on local files (tests/test_remote_zip.py) and `http_range_fetcher` supplies the network side.
Supports ZIP64 (archives > 4 GB), stored and deflated members; every member is checked against its CRC-32.
"""
from __future__ import annotations

import struct
import time
import urllib.request
import zlib
from dataclasses import dataclass

EOCD = b"PK\x05\x06"
EOCD64_LOCATOR = b"PK\x06\x07"
EOCD64 = b"PK\x06\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


@dataclass(frozen=True)
class Member:
    name: str
    method: int            # 0 stored, 8 deflate
    crc32: int
    compressed_size: int
    size: int
    header_offset: int


def _zip64_extra(extra, size, csize, offset):
    """Replace 0xFFFFFFFF fields by the ZIP64 extended-information extra field (id 0x0001), in its fixed order."""
    i = 0
    while i + 4 <= len(extra):
        tag, n = struct.unpack_from("<HH", extra, i)
        if tag == 0x0001:
            vals = list(struct.unpack_from(f"<{n // 8}Q", extra, i + 4))
            if size == 0xFFFFFFFF:
                size = vals.pop(0)
            if csize == 0xFFFFFFFF:
                csize = vals.pop(0)
            if offset == 0xFFFFFFFF:
                offset = vals.pop(0)
            break
        i += 4 + n
    return size, csize, offset


class RemoteZip:
    def __init__(self, fetch, archive_size, members=None):
        """members: a previously read central directory ({name: Member}); read from the archive when None."""
        self.fetch, self.archive_size = fetch, int(archive_size)
        self.members = members if members is not None else self._central_directory()

    def _central_directory(self):
        tail_len = min(self.archive_size, 65536 + 22)
        tail = self.fetch(self.archive_size - tail_len, self.archive_size - 1)
        e = tail.rfind(EOCD)
        if e < 0:
            raise ValueError("no end-of-central-directory record")
        _, _, _, _, count, cd_size, cd_offset, _ = struct.unpack_from("<4sHHHHIIH", tail, e)
        loc = tail.rfind(EOCD64_LOCATOR, 0, e)
        if loc >= 0:                                            # ZIP64
            (eocd64_offset,) = struct.unpack_from("<Q", tail, loc + 8)
            rec = self.fetch(eocd64_offset, eocd64_offset + 55)
            if rec[:4] != EOCD64:
                raise ValueError("bad ZIP64 end-of-central-directory record")
            count, cd_size, cd_offset = struct.unpack_from("<QQQ", rec, 32)
        cd = self.fetch(cd_offset, cd_offset + cd_size - 1)
        members, i = {}, 0
        for _ in range(count):
            if cd[i:i + 4] != CENTRAL:
                raise ValueError("bad central directory entry")
            (method, _, _, crc, csize, size, n_name, n_extra, n_comment, _, _, _, offset) = struct.unpack_from(
                "<HHHIIIHHHHHII", cd, i + 10)
            name = cd[i + 46:i + 46 + n_name].decode("utf-8")
            extra = cd[i + 46 + n_name:i + 46 + n_name + n_extra]
            size, csize, offset = _zip64_extra(extra, size, csize, offset)
            members[name] = Member(name, method, crc, csize, size, offset)
            i += 46 + n_name + n_extra + n_comment
        return members

    def read(self, name, slack=1024):
        """Bytes of one member, decompressed and CRC-checked. One request when the local header's extra field fits
        in `slack` bytes, two otherwise."""
        m = self.members[name]
        head = 30 + len(m.name.encode("utf-8"))
        blob = self.fetch(m.header_offset, m.header_offset + head + slack + m.compressed_size - 1)
        if blob[:4] != LOCAL:
            raise ValueError(f"{name}: bad local file header")
        n_name, n_extra = struct.unpack_from("<HH", blob, 26)
        start = 30 + n_name + n_extra
        data = blob[start:start + m.compressed_size]
        if len(data) < m.compressed_size:
            data = self.fetch(m.header_offset + start, m.header_offset + start + m.compressed_size - 1)
        if m.method == 0:
            out = data
        elif m.method == 8:
            out = zlib.decompress(data, -15)
        else:
            raise ValueError(f"{name}: compression method {m.method} not supported")
        if len(out) != m.size or zlib.crc32(out) & 0xFFFFFFFF != m.crc32:
            raise ValueError(f"{name}: size / CRC-32 mismatch")
        return out


def http_range_fetcher(url, retries=5, timeout=60):
    """(fetch, size) for a URL whose server honours Range requests."""
    def request(headers, method="GET"):
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers, method=method), timeout=timeout) as r:
                    return r.status, r.headers, (r.read() if method == "GET" else b"")
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    _, headers, _ = request({}, "HEAD")
    size = int(headers["Content-Length"])

    def fetch(start, end):
        end = min(end, size - 1)
        status, _, body = request({"Range": f"bytes={start}-{end}"})
        if status != 206 or len(body) != end - start + 1:
            raise IOError(f"{url}: range {start}-{end} not honoured (status {status}, {len(body)} bytes)")
        return body
    return fetch, size
