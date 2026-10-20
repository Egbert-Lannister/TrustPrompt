"""Hypersim frames as metric axial-Z depth with a pinhole K (input of the structured-error synthesis).

Hypersim stores `depth_meters` as the Euclidean distance to the camera centre and describes each scene's camera
by M_cam_from_uv (contrib/mikeroberts3000): ray_cam = M_cam_from_uv @ [u, v, 1] in OpenGL camera axes (x right,
y up, -z forward), with u, v in [-1, 1] at pixel centres (v = +1 on the top row). For scenes with tilt-shift
parameters the image plane is not perpendicular to the camera's z axis. We write pixel -> ray as one 3x3 matrix
in computer-vision axes (x right, y down, z forward), invert it and RQ-decompose it into K R: R rotates the camera
frame so that the image plane is perpendicular to the new optical axis, and K is upper triangular (with a small
skew for tilt-shift scenes; tp.sim.mirror_render handles skew). Depth is reported as Z in that rotated frame, so
backproject(Z, K) gives exact 3-D points and projecting them gives the pixel grid back.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

CV_FROM_GL = np.diag([1.0, -1.0, -1.0])
MIRROR, WINDOW = 19, 9


def uv_from_pixel(h, w):
    """Affine map from pixel indices [x, y, 1] (centres at integers) to Hypersim's [u, v, 1]."""
    return np.array([[2.0 / w, 0.0, 1.0 / w - 1.0], [0.0, -2.0 / h, 1.0 - 1.0 / h], [0.0, 0.0, 1.0]])


def pinhole(M_cam_from_uv, h, w):
    """(K, R, Kinv_cv): pixel -> ray (CV axes) is Kinv_cv; inv(Kinv_cv) = K R, K upper triangular, K[2, 2] = 1,
    positive focal lengths, R a rotation."""
    import scipy.linalg
    kinv = CV_FROM_GL @ np.asarray(M_cam_from_uv, np.float64) @ uv_from_pixel(h, w)
    K, R = scipy.linalg.rq(np.linalg.inv(kinv))
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0:
        raise ValueError("camera matrix is not a proper rotation times K")
    return K / K[2, 2], R, kinv


def axial_depth(distance, kinv, R):
    """Euclidean distance per pixel -> Z in the rotated pinhole frame (NaN stays NaN)."""
    h, w = distance.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    rays = np.stack([x, y, np.ones_like(x)], -1) @ kinv.T
    z_dir = (rays @ R.T)[..., 2] / np.linalg.norm(rays, axis=-1)
    return (distance * z_dir).astype(np.float32)


@dataclass
class Frame:
    scene: str
    camera: str
    frame: int
    rgb: np.ndarray            # uint8 HxWx3 (tone-mapped preview)
    depth: np.ndarray          # float32 HxW axial Z in metres, NaN where nothing was hit
    valid: np.ndarray          # bool HxW
    semantic: np.ndarray       # int16 HxW NYU40 labels (-1 unlabelled)
    K: np.ndarray              # 3x3, possibly with skew
    R: np.ndarray              # rotation from Hypersim's CV camera frame to the pinhole frame


def load_frame(root, scene, camera, frame, M_cam_from_uv):
    """Read one frame from the downloaded member files under `root` (scripts/hypersim_fetch.py layout)."""
    import h5py
    from PIL import Image
    base = root / scene / "images"
    with h5py.File(base / f"scene_{camera}_geometry_hdf5" / f"frame.{frame:04d}.depth_meters.hdf5", "r") as h:
        dist = np.asarray(h["dataset"], np.float32)
    with h5py.File(base / f"scene_{camera}_geometry_hdf5" / f"frame.{frame:04d}.semantic.hdf5", "r") as h:
        sem = np.asarray(h["dataset"]).astype(np.int16)
    rgb = np.asarray(Image.open(io.BytesIO((base / f"scene_{camera}_final_preview" /
                                            f"frame.{frame:04d}.tonemap.jpg").read_bytes())).convert("RGB"))
    h, w = dist.shape
    if rgb.shape[:2] != (h, w) or sem.shape != (h, w):
        raise ValueError(f"{scene}/{camera}/{frame}: image, depth and semantic grids differ")
    K, R, kinv = pinhole(M_cam_from_uv, h, w)
    z = axial_depth(dist, kinv, R)
    valid = np.isfinite(z) & (z > 0)
    return Frame(scene, camera, frame, rgb, np.where(valid, z, np.nan).astype(np.float32), valid, sem, K, R)
