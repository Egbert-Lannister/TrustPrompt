"""Plan revision R3: bound the gap between NYU registered depth and RGB-axis Z.

The NYU toolbox (project_depth_map.m) projects depth-camera points into the
RGB image but writes the *depth-camera* Z into the RGB grid. For a pixel with
ray r = K^-1 [u, v, 1] in the RGB camera and the toolbox extrinsics
X_rgb = R X_d + T, the stored value is Z_d = c . (X_rgb - T) with c the third
row of R^-1. With X_rgb = Z_rgb r this gives the exact inversion

    Z_rgb = (Z_d + c . T) / (c . r).

Projection rounding moves the true ray within +-0.5 px of the pixel centre, so
the bound takes the worst case over the four pixel corners.

Calibration is never typed in by hand: it is parsed from the official toolbox
files, whose SHA256 must match the audited hashes (docs/intrinsics_audit.md).
Unrecognised MATLAB statements raise instead of being guessed.
"""
from __future__ import annotations

import hashlib
import re
import zipfile

import numpy as np

AUDITED_SHA256 = {
    "camera_params.m": "03ad3755b605d1a4d2943c8cd56d722f1adec11c1ca7c761d5cf20dad5a1f660",
    "depth_world2rgb_world.m": "555d779e66cd76c18f3558a137af6fa866a5cfcdc889e02617eb9720f924bfdd",
    "project_depth_map.m": "78485d3a43482e3dd53279e8c86db6e966ec0bb398c524464376b0142bfc4286",
}
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def read_toolbox(zip_path, verify=True):
    """Return {basename: text} for the audited files inside the official ZIP."""
    texts = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            base = info.filename.rsplit("/", 1)[-1]
            if base in AUDITED_SHA256:
                raw = z.read(info)
                digest = hashlib.sha256(raw).hexdigest()
                if verify and digest != AUDITED_SHA256[base]:
                    raise ValueError(f"{base} SHA256 {digest} differs from audited hash")
                texts[base] = raw.decode("utf-8", errors="strict")
    missing = set(AUDITED_SHA256) - set(texts)
    if missing:
        raise ValueError(f"toolbox ZIP lacks {sorted(missing)}")
    return texts


def _strip_comments(text):
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def parse_camera_params(text):
    """Parse scalars and replay the rotation statement sequence of camera_params.m.

    Supported statements (anything else assigning R raises):
      R = [-][a, b, ...]      1x9 row vector (no ';') or 3x3 row-major (with ';')
      R = reshape(R, [3 3])   MATLAB column-major reshape
      R = inv(R')  /  R = R'
    """
    body = _strip_comments(text).replace("...", " ")
    scalars = {name: float(value) for name, value in
               re.findall(rf"\b([A-Za-z_]\w*)\s*=\s*({_NUM})\s*;", body)}
    # Protect ';' inside brackets so statements split on top-level ';' only.
    protected = re.sub(r"\[([^\]]*)\]", lambda m: "[" + m.group(1).replace(";", "|") + "]", body)
    rotation, steps = None, []
    for statement in protected.split(";"):
        s = " ".join(statement.split())
        if not re.match(r"^R\s*=", s):
            continue
        literal = re.match(r"^R\s*=\s*(-?)\s*\[(.*)\]$", s)
        if literal:
            values = [float(v) for v in re.findall(_NUM, literal.group(2))]
            if len(values) != 9:
                raise ValueError("rotation literal does not have 9 entries")
            sign = -1.0 if literal.group(1) else 1.0
            if "|" in literal.group(2):
                rotation = sign * np.array(values).reshape(3, 3)
                steps.append("literal_3x3" + ("_negated" if sign < 0 else ""))
            else:
                rotation = sign * np.array(values).reshape(1, 9)
                steps.append("literal_1x9" + ("_negated" if sign < 0 else ""))
        elif re.match(r"^R\s*=\s*reshape\(\s*R\s*,\s*\[\s*3\s*,?\s*3\s*\]\s*\)$", s):
            rotation = rotation.reshape(-1).reshape(3, 3, order="F"); steps.append("reshape_column_major")
        elif re.match(r"^R\s*=\s*inv\(\s*R\s*'\s*\)$", s):
            rotation = np.linalg.inv(rotation.T); steps.append("inv(R')")
        elif re.match(r"^R\s*=\s*R\s*'$", s):
            rotation = rotation.T; steps.append("R'")
        else:
            raise ValueError(f"unrecognised rotation statement: {s!r}")
    if rotation is None or rotation.shape != (3, 3):
        raise ValueError("no 3x3 rotation obtained from camera_params.m")
    return scalars, rotation, steps


parse_camera_params_text = parse_camera_params


def parse_extrinsic_application(text):
    """Return (rotation_form, translation_axis_map) from depth_world2rgb_world.m.

    Supported: ``points = R * points' (+ T * ones(...))`` with ``T = [a; b; c]``
    naming scalars, or per-row ``points(k,:) = points(k,:) + t_?`` lines.
    """
    body = " ".join(_strip_comments(text).split())
    if re.search(r"=\s*R\s*\*\s*\w+\s*'", body):
        form = "R_times_columns"
    else:
        raise ValueError("unrecognised rotation application in depth_world2rgb_world.m")
    mapping = {}
    vector = re.search(r"\bT\s*=\s*\[([^\]]*)\]", body)
    if vector and re.search(r"\+\s*T\s*\*", body):
        names = [n.strip() for n in vector.group(1).split(";")]
        if len(names) != 3 or not all(re.fullmatch(r"t_[xyz]", n) for n in names):
            raise ValueError(f"translation vector not recognised: {vector.group(1)!r}")
        mapping = dict(enumerate(names))
    else:
        for a, b, name in re.findall(
                r"\w+\(\s*(\d|:)\s*,\s*(\d|:)\s*\)\s*=\s*\w+\(\s*\1\s*,\s*\2\s*\)\s*\+\s*(t_[xyz])", body):
            mapping[(int(a) if a != ":" else int(b)) - 1] = name
    if sorted(mapping) != [0, 1, 2]:
        raise ValueError(f"translation assignment not recognised: {mapping}")
    return form, mapping


def extrinsics(texts):
    """X_rgb = R X_d + T exactly as the toolbox code computes it.

    rgb_world2rgb_plane divides X and Y by Z, so it is invariant to negating
    all three coordinates. If det(R) < 0 (the toolbox literal is negated), the
    physically consistent rigid transform with positive RGB depth is
    (-R, -T); both forms give identical projections. The flip is recorded.
    """
    scalars, rotation, steps = parse_camera_params_text(texts["camera_params.m"])
    form, mapping = parse_extrinsic_application(texts["depth_world2rgb_world.m"])
    translation = np.array([scalars[mapping[i]] for i in range(3)])
    flipped = bool(np.linalg.det(rotation) < 0)
    if flipped:
        rotation, translation = -rotation, -translation
    if np.linalg.det(rotation) <= 0:
        raise ValueError("rotation is not proper even after the projection-invariant sign flip")
    return dict(R=rotation, T=translation, rotation_steps=steps, rotation_form=form,
                translation_axis_map={str(k): v for k, v in mapping.items()},
                global_sign_flip_applied=flipped,
                sign_flip_justification="rgb_world2rgb_plane uses X/Z, Y/Z; negating (R,T) leaves projections unchanged",
                det_R=float(np.linalg.det(rotation)),
                orthogonality_error=float(np.abs(rotation @ rotation.T - np.eye(3)).max()))


def rgb_z(z_depth, K, R, T, offset=(0.0, 0.0)):
    """Exact RGB-axis Z for registered depth-camera Z on the RGB pixel grid."""
    z = np.asarray(z_depth, dtype=np.float64)
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    u += offset[0]; v += offset[1]
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    rays = np.stack([u, v, np.ones_like(u)], -1) @ Kinv.T
    c = np.linalg.inv(R)[2]
    return (z + c @ T) / (rays @ c)


def relative_gap(z_depth, valid, K, R, T):
    """Worst case over pixel-corner rays of |Z_rgb - Z_d| / Z_d on valid pixels.

    Raises if any recovered Z_rgb is nonpositive (sign convention error)."""
    z = np.asarray(z_depth, dtype=np.float64)
    worst = np.zeros(z.shape)
    for offset in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5), (0., 0.)):
        z_rgb = rgb_z(z, K, R, T, offset)
        if np.any(z_rgb[valid] <= 0):
            raise ValueError("nonpositive RGB-axis Z recovered; extrinsic sign convention is wrong")
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.abs(z_rgb - z) / z
        worst = np.maximum(worst, np.where(valid, gap, 0.0))
    return np.where(valid, worst, np.nan)
