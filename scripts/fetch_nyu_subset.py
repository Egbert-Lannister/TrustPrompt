"""Fetch only approved train frames from the official NYU MAT with bounded ranges.

HTTPRangeReader is selectively ported from GLARE glare/first_surface_archives.py.
Original source SHA256: 6178950737331845d60d8104b04a367471bd61fed06c4be5ba79f65ea5038f81
No imports from GLARE. All new files go under this project.
"""
from __future__ import annotations

from collections import OrderedDict
import errno
import http.client
import io
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import socket
import ssl
import time
from typing import Any, Callable, Collection, Mapping
import urllib.error
import urllib.request
import zipfile


MIB = 1024 ** 2
GIB = 1024 ** 3


class RangeProtocolError(OSError):
    """Remote server did not honor the bounded, stable byte-range contract."""


class RangeConnectionError(OSError):
    """Temporary connection retries were exhausted; acquisition is incomplete."""


def _temporary_connection_error(error: Exception) -> bool:
    # HTTP status, certificate, protocol and local validation failures must
    # never become retryable merely because they inherit from OSError.
    if isinstance(error, (RangeProtocolError, urllib.error.HTTPError, ssl.SSLCertVerificationError)):
        return False
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, Exception) and _temporary_connection_error(error.reason)
    if isinstance(error, (ssl.SSLEOFError, TimeoutError, ConnectionResetError,
                          ConnectionAbortedError, BrokenPipeError,
                          http.client.IncompleteRead, http.client.RemoteDisconnected)):
        return True
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    return isinstance(error, OSError) and error.errno in {
        errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
        errno.ETIMEDOUT, errno.EPIPE,
    }


class HTTPRangeReader(io.RawIOBase):
    """Seekable readonly file backed by bounded Range requests and an LRU cache.

    ``opener`` is an injectable urllib-compatible callable accepting a Request
    and timeout. The default uses urlopen. Single reads are capped independently
    of the cache so an accidental read() cannot allocate a multi-GB archive.
    """

    def __init__(
        self, url: str, *, opener: Callable[..., Any] | None = None,
        block_size: int = MIB, max_cached_blocks: int = 4,
        max_read_bytes: int = 64 * MIB, timeout: float = 30,
        max_total_download_bytes: int | None = None, expected_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise ValueError("an explicit HTTP(S) archive URL is required")
        for name, value in (("block_size", block_size), ("max_cached_blocks", max_cached_blocks),
                            ("max_read_bytes", max_read_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if block_size > 8 * MIB:
            raise ValueError("individual Range blocks are limited to 8 MiB")
        if max_total_download_bytes is not None and (
            isinstance(max_total_download_bytes, bool) or
            not isinstance(max_total_download_bytes, int) or max_total_download_bytes < 1
        ):
            raise ValueError("max_total_download_bytes must be a positive integer")
        self.url = url
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.max_read_bytes = max_read_bytes
        self.timeout = timeout
        self.max_total_download_bytes = max_total_download_bytes
        self._opener = opener or urllib.request.urlopen
        self._position = 0
        self._size: int | None = None
        self._etag: str | None = None
        self._last_modified: str | None = None
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.downloaded_bytes = 0
        self.request_count = 0
        self._fetch(0, 0)  # One byte establishes size without a full-body GET.
        if expected_size is not None and (
            isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size != self.size
        ):
            self.close()
            raise RangeProtocolError("archive size disagrees with caller-provided expected_size")

    @property
    def size(self) -> int:
        assert self._size is not None
        return self._size

    @property
    def cache_bytes(self) -> int:
        return sum(len(value) for value in self._cache.values())

    @property
    def archive_identity(self) -> dict[str, Any]:
        return {"url": self.url, "size": self.size, "etag": self._etag,
                "last_modified": self._last_modified}

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self.archive_identity, "downloaded_bytes": self.downloaded_bytes,
                "request_count": self.request_count, "cache_bytes": self.cache_bytes,
                "block_size": self.block_size, "max_cached_blocks": self.max_cached_blocks}

    def readable(self) -> bool:
        return not self.closed

    def seekable(self) -> bool:
        return not self.closed

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if not isinstance(offset, int):
            raise TypeError("seek offset must be an integer")
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid seek whence")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def _fetch(self, start: int, end: int) -> bytes:
        # Retry only this missing block, preserving prior cached blocks and
        # every attempt's byte/request accounting. TLS verification is never
        # changed. Protocol/budget failures from _fetch_once pass straight out.
        for attempt in range(3):
            try:
                return self._fetch_once(start, end)
            except Exception as error:
                if not _temporary_connection_error(error):
                    raise
                if attempt == 2:
                    raise RangeConnectionError(
                        "temporary Range connection failed after 3 attempts; acquisition incomplete"
                    ) from error
                time.sleep(attempt + 1)
        raise AssertionError("unreachable Range retry state")

    def _read_response(self, response: Any, size: int) -> bytes:
        try:
            chunk = response.read(size)
        except http.client.IncompleteRead as error:
            # urllib may expose bytes consumed before a premature socket EOF
            # only on the exception. They still count against the transfer cap.
            self.downloaded_bytes += len(error.partial)
            if len(error.partial) > size:
                raise RangeProtocolError("Range body exceeded its declared size") from error
            raise
        self.downloaded_bytes += len(chunk)
        return chunk

    def _fetch_once(self, start: int, end: int) -> bytes:
        expected = end - start + 1
        if self.max_total_download_bytes is not None and self.downloaded_bytes + expected > self.max_total_download_bytes:
            raise RangeProtocolError("configured cumulative transfer limit would be exceeded")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self._etag and not self._etag.startswith("W/"):
            headers["If-Range"] = self._etag
        elif self._last_modified:
            headers["If-Range"] = self._last_modified
        request = urllib.request.Request(self.url, headers=headers)
        self.request_count += 1
        response = self._opener(request, timeout=self.timeout)
        try:
            if getattr(response, "status", None) != 206:
                raise RangeProtocolError("server must return 206; full-body responses are forbidden")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in ("", "identity"):
                raise RangeProtocolError("encoded Range response is not byte-addressable")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "").strip())
            if match is None:
                raise RangeProtocolError("missing or invalid Content-Range")
            actual_start, actual_end, total = map(int, match.groups())
            if (actual_start, actual_end) != (start, end) or total <= actual_end:
                raise RangeProtocolError("Content-Range does not match the requested offset/extent")
            if self._size is not None and total != self._size:
                raise RangeProtocolError("archive size changed between Range requests")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError as exc:
                    raise RangeProtocolError("invalid Content-Length") from exc
                if length != expected:
                    raise RangeProtocolError("Content-Length disagrees with the requested Range")
            etag = response.headers.get("ETag")
            modified = response.headers.get("Last-Modified")
            if self._size is not None:
                if self._etag is not None and etag != self._etag:
                    raise RangeProtocolError("archive ETag changed or disappeared")
                if self._last_modified is not None and modified != self._last_modified:
                    raise RangeProtocolError("archive Last-Modified changed or disappeared")
            body = bytearray()
            while len(body) < expected:
                chunk = self._read_response(response, expected - len(body))
                if not chunk:
                    raise RangeProtocolError("truncated Range body")
                if len(chunk) > expected - len(body):
                    raise RangeProtocolError("Range body exceeded its declared size")
                body.extend(chunk)
            extra = self._read_response(response, 1)
            if extra:
                raise RangeProtocolError("Range body exceeded its declared size")
            if self._size is None:
                self._size, self._etag, self._last_modified = total, etag, modified
            return bytes(body)
        finally:
            response.close()

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if not isinstance(size, int):
            raise TypeError("read size must be an integer")
        if size < 0:
            size = max(0, self.size - self._position)
        size = min(size, max(0, self.size - self._position))
        if size > self.max_read_bytes:
            raise RangeProtocolError("single read exceeds max_read_bytes; stream approved members in chunks")
        remaining = size
        result = bytearray()
        while remaining:
            block_start = self._position // self.block_size * self.block_size
            if block_start not in self._cache:
                block_end = min(self.size - 1, block_start + self.block_size - 1)
                self._cache[block_start] = self._fetch(block_start, block_end)
                while len(self._cache) > self.max_cached_blocks:
                    self._cache.popitem(last=False)
            block = self._cache[block_start]
            self._cache.move_to_end(block_start)
            offset = self._position - block_start
            take = min(remaining, len(block) - offset)
            result.extend(block[offset:offset + take])
            self._position += take
            remaining -= take
        return bytes(result)

    def readinto(self, buffer: Any) -> int:
        content = self.read(len(buffer))
        buffer[:len(content)] = content
        return len(content)

    def close(self) -> None:
        if hasattr(self, "_cache"):
            self._cache.clear()
        super().close()


def main() -> None:
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import json
    import sys
    import h5py
    import numpy as np

    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=project / 'data/source_cache/nyu_smoke_v1.h5')
    parser.add_argument('--split-reference', type=Path, default=project / 'configs/splits/mirror3d_nyu_v1.json')
    parser.add_argument('--ids', nargs='+', type=int, default=[45, 184, 245, 440, 654])
    parser.add_argument('--max-download-mib', type=int, default=100)
    parser.add_argument('--legacy-cache', type=Path, default=Path('E:/GLARE_Data/first_surface/mirror3d/nyu_depth_v2_labeled.selected.h5'))
    args = parser.parse_args()
    if not 1 <= args.max_download_mib <= 100:
        raise ValueError('This smoke acquisition is capped at 100 MiB.')
    if len(args.ids) > 5 or len(set(args.ids)) != len(args.ids):
        raise ValueError('At most 5 distinct train frames are allowed.')
    ref = json.loads(args.split_reference.read_text(encoding='utf8'))
    mapping = {int(r['annotation_id']): r for r in ref['mapping']}
    if any(i not in mapping or mapping[i]['split'] != 'train' for i in args.ids):
        raise ValueError('Only approved training annotations may be read by this smoke fetcher.')
    if len({mapping[i]['scene'] for i in args.ids}) != len(args.ids):
        raise ValueError('Select different scenes for this smoke acquisition.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Will not overwrite existing cache: ' + str(args.output))
    url = 'https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
    manifest = {
        'schema': 'trustprompt.nyu_subset.v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv], 'selected_ids': args.ids,
        'source_index_rule': 'source_index=int(annotation_id); released-annotation correction',
        'source_url': url, 'depth_unit': 'metres', 'depth_convention': 'axial_z',
        'split_reference_sha256': hashlib.sha256(args.split_reference.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'payloads': [], 'status': 'INCOMPLETE', 'test_depth_accessed': False,
        'max_download_bytes': args.max_download_mib * MIB,
    }
    def save_manifest():
        args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf8')
    def strings(source, key):
        refs = source[key][:].ravel()
        return [''.join(chr(int(c)) for c in source[r][:].ravel()) for r in refs]
    reader = None
    legacy = None
    try:
        with HTTPRangeReader(url, block_size=MIB, max_cached_blocks=16, max_read_bytes=8*MIB,
                             max_total_download_bytes=args.max_download_mib*MIB,
                             expected_size=2972037809) as reader, h5py.File(reader, 'r') as source:
            manifest['source_identity'] = reader.archive_identity
            manifest['source_layouts'] = {k: {'shape': list(source[k].shape), 'dtype': str(source[k].dtype),
                                                  'chunks': source[k].chunks}
                                          for k in ['images', 'depths', 'rawDepths', 'labels', 'scenes', 'names']}
            scenes = strings(source, 'scenes')
            names = strings(source, 'names')
            mismatches = [r for i, r in mapping.items() if scenes[i] != r['scene']]
            if mismatches:
                raise ValueError('Online scenes disagree with historical correction: ' + str(mismatches))
            # N-1 is fetched only for the numbering regression, never if it belongs to a test scene.
            for i in args.ids:
                if ref['scene_splits'].get(scenes[i - 1]) == 'test':
                    raise ValueError('Regression predecessor belongs to a held-out scene.')
            manifest['metadata_validation'] = {'scene_count': len(scenes), 'class_count': len(names),
                'annotation_scene_mismatch_count': len(mismatches),
                'csv_scene_mismatch_count': sum(scenes[i] != r['csv_scene'] for i, r in mapping.items()),
                'mapping': list(mapping.values()), 'scene_splits': ref['scene_splits']}
            print(json.dumps({'stage':'metadata_verified','downloaded_bytes':reader.downloaded_bytes,
                              'layouts':manifest['source_layouts']}), flush=True)
            if args.legacy_cache.exists():
                legacy = h5py.File(args.legacy_cache, 'r')
                identity = json.loads(legacy.attrs.get('source_identity_json', '{}'))
                # Historical JSON over-escaped ETag quotes; normalize only this representation quirk.
                identity['etag'] = identity.get('etag', '').replace('\\"', '"')
                if identity != reader.archive_identity or legacy.attrs.get('raw_slices_unchanged') != 1:
                    raise ValueError('Legacy raw-slice identity differs from the live official source.')
                manifest['legacy_cache'] = {'path':str(args.legacy_cache), 'identity_verified':True,
                    'schema':str(legacy.attrs['schema']), 'reuse':'unchanged source slices only at exact source index; no predictions'}
            with h5py.File(args.output, 'x') as dest:
                dest.attrs.update(schema='trustprompt.nyu_subset.v1', source_url=url,
                                  depth_unit='metres', source_index_rule='annotation_N_equals_h5_index_N')
                dest.create_dataset('scenes', data=np.asarray(scenes, dtype=object), dtype=h5py.string_dtype('utf-8'))
                dest.create_dataset('names', data=np.asarray(names, dtype=object), dtype=h5py.string_dtype('utf-8'))
                frames = dest.create_group('frames')
                for i in args.ids:
                    group = frames.create_group(str(i))
                    group.attrs.update(source_index=i, scene=scenes[i], split='train')
                    for target, key, idx in [('images','images',i), ('depths','depths',i),
                                             ('rawDepths','rawDepths',i), ('labels','labels',i),
                                             ('depths_previous','depths',i-1)]:
                        use_legacy = legacy is not None and key in legacy and f'{key}_present' in legacy and bool(legacy[f'{key}_present'][idx])
                        array = (legacy if use_legacy else source)[key][idx]
                        group.create_dataset(target, data=array, compression='gzip')
                        manifest['payloads'].append({'annotation_id':i, 'field':target, 'source_dataset':key,
                            'source_index':idx, 'source_scene':scenes[idx], 'shape':list(array.shape),
                            'dtype':str(array.dtype), 'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                            'read_from':'legacy_unchanged_source_slice' if use_legacy else 'official_http_range'})
                        print(json.dumps({'stage':'slice_saved','id':i,'field':target,
                                          'reused':use_legacy,'downloaded_bytes':reader.downloaded_bytes}),flush=True)
                    dest.flush()
                dest.attrs['complete'] = 1
            manifest['status'] = 'COMPLETE_REAL_DATA_SUBSET'
            manifest['network'] = reader.metadata
            manifest['cache_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
            save_manifest()
    except BaseException as error:
        manifest['error'] = repr(error)
        if reader is not None:
            manifest['network'] = reader.metadata
        save_manifest()
        raise
    finally:
        if legacy is not None:
            legacy.close()
    print(json.dumps({'status': manifest['status'], 'cache': str(args.output),
                      'network':manifest['network']}), flush=True)


if __name__ == '__main__':
    main()

