"""Training samples for the mainline fine-tuning (P1, docs/plan_omnidc.md sec. 5) from the downloaded Hypersim frames.

Every configuration sees the same frames, augmentation and OMNI-DC's own input noise; only the structured errors
differ:
  f0      clean sensor reading (tp.sim.sensor) + OMNI-DC's i.i.d. outliers (src/data/__init__.py add_noise: each
          reading replaced with probability U(0, 0.05) by a value from U[p10, p90] of the depths)
  f1, f2  the same, with condition C applied first (tp.sim.hypersim_condition; mirror and window frames)
With probability `p_sparse` the reading is thinned to 100-2000 random points (OMNI-DC's main training pattern), so
the fine-tuned model keeps working on sparse inputs (non-inferiority check N2).
Per sample: load the frame, resize to 480 x 640 (K scaled), synthesise, random scale in [1, 1.5] + random 480 x 640
crop + horizontal flip (OMNI-DC's augmentation), divide reading and GT by the median reading (OMNI-DC's
data_normalize_median). Sample n always uses the generator default_rng([seed, n]), so a run resumes exactly.
Labels (for the gate and the two-sided ConfInput loss): tp.sim.corrupt.type_labels on the final reading.
"""
from __future__ import annotations

import json

import numpy as np

from tp.data.hypersim import load_frame
from tp.sim.corrupt import type_labels
from tp.sim.hypersim_condition import SENSORS, condition
from tp.sim.sensor import apply_sensor

HEIGHT, WIDTH = 480, 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def training_frames(data_root, min_mirror, splits=("train",)):
    """Frames of frames_v1.json whose scene is in `splits` (Hypersim's official split); mirror frames need
    >= min_mirror mirror pixels (configs/hypersim/training_v1.yaml)."""
    rows = json.loads((data_root / "frames_v1.json").read_text(encoding="utf-8"))["frames"]
    return [r for r in rows if r["split"] in splits and (r["role"] != "mirror" or r["mirror"] >= min_mirror)]


def resize_frame(frame, h=HEIGHT, w=WIDTH):
    """Nearest-neighbour depth / semantic, bilinear RGB, K for the new grid (pixel centres kept)."""
    from PIL import Image
    H, W = frame.depth.shape
    sy, sx = h / H, w / W
    ys = np.minimum(((np.arange(h) + 0.5) / sy).astype(int), H - 1)
    xs = np.minimum(((np.arange(w) + 0.5) / sx).astype(int), W - 1)
    K = frame.K.copy()
    K[0] *= sx; K[1] *= sy
    K[0, 2] = (frame.K[0, 2] + 0.5) * sx - 0.5; K[1, 2] = (frame.K[1, 2] + 0.5) * sy - 0.5
    rgb = np.asarray(Image.fromarray(frame.rgb).resize((w, h), Image.BILINEAR))
    return type(frame)(frame.scene, frame.camera, frame.frame, rgb, frame.depth[ys][:, xs], frame.valid[ys][:, xs],
                       frame.semantic[ys][:, xs], K, frame.R)


def omnidc_noise(P, V, rng, max_prob=0.05):
    """OMNI-DC's add_noise on the valid readings (in place on copies)."""
    P, V = P.copy(), V.copy()
    vals = P[V]
    if vals.size == 0:
        return P, V
    p = rng.uniform(0, max_prob)
    hit = V & (rng.random(P.shape) < p)
    lo, hi = np.percentile(vals, 10), np.percentile(vals, 90)
    P[hit] = rng.uniform(lo, hi, int(hit.sum()))
    return P, V


def sparsify(P, V, rng, n=(100, 2000)):
    idx = np.flatnonzero(V)
    keep = rng.choice(idx, min(len(idx), int(rng.integers(n[0], n[1] + 1))), replace=False)
    V2 = np.zeros_like(V); V2.flat[keep] = True
    return np.where(V2, P, 0).astype(np.float32), V2


def augment(arrays, rng, h=HEIGHT, w=WIDTH, scale=(1.0, 1.5)):
    """Same random scale (nearest), crop and flip for every HxW(xC) array in `arrays`."""
    H, W = arrays[0].shape[:2]
    s = rng.uniform(*scale)
    sh, sw = int(round(H * s)), int(round(W * s))
    ys = np.minimum(((np.arange(sh) + 0.5) / s).astype(int), H - 1)
    xs = np.minimum(((np.arange(sw) + 0.5) / s).astype(int), W - 1)
    y0, x0 = int(rng.integers(0, sh - h + 1)), int(rng.integers(0, sw - w + 1))
    ys, xs = ys[y0:y0 + h], xs[x0:x0 + w]
    if rng.random() < 0.5:
        xs = xs[::-1]
    return [a[ys][:, xs] for a in arrays]


def make_sample(frame, variant, rng, p_sparse=0.3):
    """One training sample (numpy) of configuration `variant` (f0 / f1 / f2) from a resized frame."""
    gt = np.where(frame.valid, frame.depth, 0).astype(np.float32)
    if variant == "f0":
        P, V, _, _ = apply_sensor(gt, frame.valid, frame.rgb, str(rng.choice(SENSORS)), rng)
    elif variant in ("f1", "f2"):                        # sensor model + mirror / window errors where annotated
        out = condition(frame, rng)
        P, V = out["P_corrupt"], out["V_corrupt"]
    else:
        raise ValueError(f"unknown variant {variant}")
    P, V = omnidc_noise(P, V, rng)
    if rng.random() < p_sparse:
        P, V = sparsify(P, V, rng)
    rgb, P, V, gt, gv = augment([frame.rgb, P, V, gt, frame.valid], rng)
    if not V.any():
        raise ValueError("no reading left")
    med = float(np.median(P[V]))
    P, gt = np.where(V, P / med, 0).astype(np.float32), np.where(gv, gt / med, 0).astype(np.float32)
    labels = type_labels(P, V, gt, gv)
    rgbn = ((rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return dict(rgb=np.ascontiguousarray(rgbn), dep=P[None], gt=gt[None], labels=labels.astype(np.int64),
                valid=V, median_m=np.float32(med))


class HypersimTraining:
    """Map-style dataset: item n -> sample of frame perm[n % N] with generator default_rng([seed, n])."""

    def __init__(self, data_root, selection, frames, variant, seed, p_sparse=0.3):
        self.root, self.selection, self.frames = data_root, selection, frames
        self.variant, self.seed, self.p_sparse = variant, int(seed), p_sparse
        self.perm = np.random.default_rng(self.seed).permutation(len(frames))

    def __len__(self):
        return 2 ** 31 - 1                                   # iteration-based training; the sampler sets the range

    def __getitem__(self, n):
        import torch
        for attempt in range(10):                           # a crop without readings: draw the next frame
            f = self.frames[self.perm[(n + attempt) % len(self.frames)]]
            rng = np.random.default_rng([self.seed, n, attempt])
            frame = resize_frame(load_frame(self.root, f["scene"], f["camera"], f["frame"],
                                            self.selection["scenes"][f["scene"]]["camera"]["M_cam_from_uv"]))
            try:
                s = make_sample(frame, self.variant, rng, self.p_sparse)
            except ValueError:
                continue
            out = {k: torch.from_numpy(np.asarray(v)) for k, v in s.items()}
            out.update(K=torch.eye(3), pattern=torch.tensor(0), index=torch.tensor(n))
            return out
        raise RuntimeError(f"sample {n}: no usable crop in 10 frames")
