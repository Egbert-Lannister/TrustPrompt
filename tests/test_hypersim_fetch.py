import io
import json
import zipfile

import h5py
import numpy as np
import pytest

import scripts.hypersim_fetch as hf


def semantic_bytes(mirror_frac, window_frac, shape=(20, 30)):
    sem = np.ones(shape, np.int16)
    n = sem.size
    sem.flat[:int(mirror_frac * n)] = hf.MIRROR
    sem.flat[int(mirror_frac * n):int(mirror_frac * n) + int(window_frac * n)] = hf.WINDOW
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h:
        h.create_dataset("dataset", data=sem, compression="gzip")
    return buf.getvalue()


@pytest.fixture()
def fake_release(tmp_path, monkeypatch):
    """Two scene archives; ai_A cam_00 frames 0-3 with (mirror, window) fractions below; ai_B only clean frames."""
    fracs = {("ai_A", 0): (0.2, 0), ("ai_A", 1): (0.005, 0.0), ("ai_A", 2): (0, 0.1), ("ai_A", 3): (0, 0)}
    fracs.update({("ai_B", i): (0, 0) for i in range(8)})
    archives = {}
    for scene in ("ai_A", "ai_B"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for (s, i), (m, w) in fracs.items():
                if s != scene:
                    continue
                g = f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{i:04d}"
                z.writestr(f"{g}.semantic.hdf5", semantic_bytes(m, w))
                z.writestr(f"{g}.depth_meters.hdf5", b"depth%d" % i * 100)
                z.writestr(f"{scene}/images/scene_cam_00_final_preview/frame.{i:04d}.tonemap.jpg", b"jpg%d" % i * 50)
        archives[f"https://x/{scene}.zip"] = buf.getvalue()
    calls = []

    def fetcher(url):
        blob = archives[url]
        def fetch(a, b):
            calls.append(url); return blob[a:b + 1]
        return fetch, len(blob)
    monkeypatch.setattr(hf, "DATA", tmp_path / "hs")
    monkeypatch.setattr(hf, "FRAMES", tmp_path / "hs" / "frames_v1.json")
    selection = dict(url_template="https://x/{scene}.zip", scenes={
        "ai_A": dict(split="train", frames={"cam_00": [0, 1, 2, 3]}),
        "ai_B": dict(split="val", frames={"cam_00": list(range(8))})})
    return selection, fetcher, calls


def test_stages_end_to_end_and_resume(fake_release):
    selection, fetcher, calls = fake_release
    objs = hf.scenes(selection, fetcher)
    p = hf.plan(selection, objs)
    assert p["A_semantic"]["files"] == 12 and p["B_rgb_depth_upper_bound"]["files"] == 24
    hf.download(objs, hf.wanted(selection, ["semantic"]), workers=3)
    out = hf.select(selection, objs)
    roles = {(r["scene"], r["frame"]): r["role"] for r in out["frames"]}
    assert roles[("ai_A", 0)] == "mirror" and roles[("ai_A", 2)] == "window" and ("ai_A", 1) not in roles
    assert out["counts"] == dict(mirror=1, window=1, clean=round(0.25 * 2))
    items = hf.wanted(selection, ["rgb", "depth"], [(r["scene"], r["camera"], r["frame"]) for r in out["frames"]])
    hf.download(objs, items, workers=2)
    assert (hf.DATA / "ai_A/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg").read_bytes() == b"jpg0" * 50
    n_calls = len(calls)
    objs2 = hf.scenes(selection, fetcher)                 # fresh process: index from disk, nothing re-fetched
    hf.download(objs2, items, workers=2)
    assert len(calls) == n_calls
    lines = (hf.DATA / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == len({json.loads(x)["name"] for x in lines}) == 12 + 2 * len(out["frames"])
