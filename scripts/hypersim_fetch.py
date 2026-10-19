"""Download the Hypersim files needed for the structured-error synthesis, member by member (Windows or Linux).

  python -m scripts.hypersim_fetch plan [--max-gb 40]   # read each scene ZIP's directory, report exact sizes;
                                                        # exit 3 if stage A + the stage B bound exceed max-gb, 4 if the disk is too small
  python -m scripts.hypersim_fetch semantic   # stage A: NYU40 semantic map of every public frame of those scenes
  python -m scripts.hypersim_fetch select     # frames that show a mirror (or a window); writes frames_v1.json
  python -m scripts.hypersim_fetch frames     # stage B: tone-mapped image + depth of the selected frames

Scenes: configs/hypersim/selection_v1.json (scripts/hypersim_scenes.py; 85 scenes whose meshes contain a mirror).
Data: the public Hypersim release (CC BY-SA 3.0), read with HTTP range requests from the official per-scene ZIPs
(tp/data/remote_zip.py; every member CRC-checked) into data/source_cache/hypersim_v1/<member path>. No Apple code
is run. Every stage is resumable: a file already present with the recorded size and SHA-256 is not fetched again.
Frame selection (fixed before any image is seen): mirror pixels >= 1 % of the image, or window pixels >= 5 %; plus
clean frames (no mirror, no window pixels) of the same scenes, a fixed-seed sample of 25 % of the selected count,
for keeping OMNI-DC's general behaviour during fine-tuning.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
import threading

import numpy as np

from tp.data.remote_zip import Member, RemoteZip, http_range_fetcher
from tp.provenance import ROOT, write_json

SELECTION = ROOT / "configs" / "hypersim" / "selection_v1.json"
DATA = ROOT / "data" / "source_cache" / "hypersim_v1"
FRAMES = DATA / "frames_v1.json"
MIRROR, WINDOW = 19, 9
MIN_MIRROR, MIN_WINDOW, CLEAN_SHARE, SEED = 0.01, 0.05, 0.25, 20261019
KINDS = {"semantic": "images/scene_{cam}_geometry_hdf5/frame.{i:04d}.semantic.hdf5",
         "depth": "images/scene_{cam}_geometry_hdf5/frame.{i:04d}.depth_meters.hdf5",
         "rgb": "images/scene_{cam}_final_preview/frame.{i:04d}.tonemap.jpg"}


class Scene:
    """One scene archive: its URL, cached directory index, and member lookup by (camera, frame, kind)."""

    def __init__(self, name, url, fetcher=http_range_fetcher):
        self.name, self.url, self._fetcher = name, url, fetcher
        self.index_path = DATA / "_index" / f"{name}.json"
        self._zip, self._lock = None, threading.Lock()

    def zip(self):
        with self._lock:
            if self._zip is None:
                fetch, size = self._fetcher(self.url)
                if self.index_path.is_file():
                    idx = json.loads(self.index_path.read_text(encoding="utf-8"))
                    if idx["archive_size"] != size:
                        raise ValueError(f"{self.name}: archive size changed since the index was read")
                    self._zip = RemoteZip(fetch, size, {m["name"]: Member(**m) for m in idx["members"]})
                else:
                    self._zip = RemoteZip(fetch, size)
                    self.index_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(self.index_path, dict(url=self.url, archive_size=size,
                                                     members=[vars(m) for m in self._zip.members.values()]))
            return self._zip

    def member(self, cam, frame, kind):
        """Members are stored as <scene>/images/...; looked up by the path below the scene folder."""
        members = self.zip().members
        if not hasattr(self, "_by_path"):
            self._by_path = {n.split("/", 1)[-1]: n for n in members}
        return members[self._by_path[KINDS[kind].format(cam=cam, i=frame)]]


def scenes(selection, fetcher=http_range_fetcher):
    return {name: Scene(name, selection["url_template"].format(scene=name), fetcher) for name in selection["scenes"]}


def wanted(selection, kinds, frames=None):
    """[(scene, camera, frame, kind)]: every public frame of the selection, or only `frames` [(scene, cam, id)]."""
    items = frames if frames is not None else [(s, cam, i) for s, d in selection["scenes"].items()
                                                 for cam, ids in d["frames"].items() for i in ids]
    return [(s, cam, i, k) for s, cam, i in items for k in kinds]


def local_path(member):
    return DATA / member.name


def download(scene_objs, items, workers=8):
    """Fetch the members of `items` that are not on disk yet; returns {member name: record}."""
    manifest = DATA / "manifest.jsonl"
    done = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[r["name"]] = r
    todo = []
    for s, cam, i, kind in items:
        m = scene_objs[s].member(cam, i, kind)
        rec = done.get(m.name)
        p = local_path(m)
        if rec and p.is_file() and p.stat().st_size == rec["size"] and hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]:
            continue
        todo.append((s, m))
    total = sum(m.compressed_size for _, m in todo)
    print(f"{len(items)} files wanted, {len(todo)} to fetch ({total / 1e9:.2f} GB on the wire)", flush=True)
    lock, got = threading.Lock(), 0

    def one(s, m):
        data = scene_objs[s].zip().read(m.name)
        p = local_path(m); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part"); tmp.write_bytes(data); tmp.replace(p)
        return dict(name=m.name, scene=s, size=len(data), crc32=m.crc32, sha256=hashlib.sha256(data).hexdigest())

    with ThreadPoolExecutor(workers) as ex, manifest.open("a", encoding="utf-8") as fh:
        futures = [ex.submit(one, s, m) for s, m in todo]
        for k, f in enumerate(as_completed(futures)):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush(); done[rec["name"]] = rec; got += rec["size"]
            if k % 200 == 0 or k == len(futures) - 1:
                print(f"  {k + 1}/{len(futures)} files, {got / 1e9:.2f} GB", flush=True)
    return done


def frame_fractions(path):
    import h5py
    with h5py.File(path, "r") as h:
        sem = np.asarray(h["dataset"])
    return float((sem == MIRROR).mean()), float((sem == WINDOW).mean())


def select(selection, scene_objs):
    rows = []
    for s, cam, i, _ in wanted(selection, ["semantic"]):
        p = local_path(scene_objs[s].member(cam, i, "semantic"))
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing: run the semantic stage first")
        mirror, window = frame_fractions(p)
        rows.append(dict(scene=s, camera=cam, frame=i, split=selection["scenes"][s]["split"], mirror=mirror, window=window))
    for r in rows:
        r["role"] = "mirror" if r["mirror"] >= MIN_MIRROR else "window" if r["window"] >= MIN_WINDOW else None
    chosen = [r for r in rows if r["role"]]
    clean = [r for r in rows if r["mirror"] == 0 and r["window"] == 0]
    rng = np.random.default_rng(SEED)
    for k in sorted(rng.choice(len(clean), min(len(clean), int(round(CLEAN_SHARE * len(chosen)))), replace=False)):
        clean[k]["role"] = "clean"; chosen.append(clean[k])
    out = dict(schema="trustprompt.hypersim_frames.v1", rule=dict(min_mirror=MIN_MIRROR, min_window=MIN_WINDOW,
               clean_share=CLEAN_SHARE, seed=SEED), counts={k: sum(1 for r in chosen if r["role"] == k)
               for k in ("mirror", "window", "clean")}, frames=chosen, all_fractions=rows)
    write_json(FRAMES, out)
    return out


def plan(selection, scene_objs):
    """Exact on-the-wire sizes per stage (stage B as an upper bound: every public frame)."""
    sizes = {}
    for kinds, label in ((["semantic"], "A_semantic"), (["rgb", "depth"], "B_rgb_depth_upper_bound")):
        items = wanted(selection, kinds)
        sizes[label] = dict(files=len(items), gb=sum(scene_objs[s].member(c, i, k).compressed_size
                                                     for s, c, i, k in items) / 1e9)
    per_frame = sizes["B_rgb_depth_upper_bound"]["gb"] / max(1, sizes["A_semantic"]["files"])
    return dict(sizes, B_gb_per_frame=per_frame)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["plan", "semantic", "select", "frames"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=None, help="plan: fail (exit 3) above this total")
    a = ap.parse_args()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    objs = scenes(selection)
    DATA.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    if a.stage == "plan":
        result = plan(selection, objs)
    elif a.stage == "semantic":
        download(objs, wanted(selection, ["semantic"]), a.workers); result = {}
    elif a.stage == "select":
        out = select(selection, objs); result = out["counts"]
    else:
        frames = json.loads(FRAMES.read_text(encoding="utf-8"))["frames"]
        download(objs, wanted(selection, ["rgb", "depth"], [(r["scene"], r["camera"], r["frame"]) for r in frames]), a.workers)
        result = {}
    log = DATA / "stages.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(stage=a.stage, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                                 command=[sys.executable, "-m", "scripts.hypersim_fetch", *sys.argv[1:]],
                                 selection_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(), result=result)) + "\n")
    print(json.dumps(result, indent=1))
    if a.stage == "plan" and a.max_gb is not None:
        total = result["A_semantic"]["gb"] + result["B_rgb_depth_upper_bound"]["gb"]
        if total > a.max_gb:
            print(f"total {total:.1f} GB exceeds the agreed {a.max_gb} GB: not downloading", flush=True)
            sys.exit(3)
        free = shutil.disk_usage(DATA).free / 1e9
        if free < total + 5:
            print(f"{DATA} has {free:.1f} GB free; the download needs up to {total:.1f} GB + 5 GB margin", flush=True)
            sys.exit(4)


if __name__ == "__main__":
    main()
