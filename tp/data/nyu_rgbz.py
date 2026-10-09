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
    """Parse scalars and the rotation statement sequence from camera_params.m."""
    body = _strip_comments(text)
    scalars = {name: float(value) for name, value in
               re.findall(rf"\b([A-Za-z_]\w*)\s*=\s*({_NUM})\s*;", body)}
    rotation, steps = None, []
    for statement in re.split(r";", body):
        s = " ".join(statement.split())
        if not re.match(r"^R\s*=", s):
            continue
        literal = re.match(r"^R\s*=\s*(-?)\s*\[(.*)\]$", s)
        if literal:
            values = [float(v) for v in re.findall(_NUM, literal.group(2))]
            if len(values) != 9:
                raise ValueError("rotation literal does not have 9 entries")
            rotation = np.array(values).reshape(3, 3) * (-1.0 if literal.group(1) else 1.0)
            steps.append("literal" + ("_negated" if literal.group(1) else ""))
        elif re.match(r"^R\s*=\s*inv\(\s*R\s*'\s*\)$", s):
            rotation = np.linalg.inv(rotation.T); steps.append("inv(R')")
        elif re.match(r"^R\s*=\s*R\s*'$", s):
            rotation = rotation.T; steps.append("R'")
        else:
            raise ValueError(f"unrecognised rotation statement: {s!r}")
    if rotation is None:
        raise ValueError("no rotation found in camera_params.m")
    return scalars, rotation, steps


def parse_camera_params_text(text):
    """camera_params.m writes the rotation with ';' row separators inside [];
    rejoin bracket content before statement splitting."""
    body = _strip_comments(text)
    body = re.sub(r"\[([^\]]*)\]", lambda m: "[" + m.group(1).replace(";", " ") + "]", body)
    return parse_camera_params(body)


def parse_extrinsic_application(text):
    """Return (rotation_form, translation_axis_map) from depth_world2rgb_world.m.

    Supported rotation forms: points = R * points' (column vectors) and
    points = points * R'. Translation lines: points(k,:) = points(k,:) + t_?
    or points(:,k) = points(:,k) + t_?. Anything else raises.
    """
    body = " ".join(_strip_comments(text).split())
    if re.search(r"=\s*R\s*\*\s*\w+\s*'", body):
        form = "R_times_columns"
    elif re.search(r"=\s*\w+\s*\*\s*R\s*'", body):
        form = "rows_times_Rt"
    else:
        raise ValueError("unrecognised rotation application in depth_world2rgb_world.m")
    mapping = {}
    for a, b, name in re.findall(
            r"\w+\(\s*(\d|:)\s*,\s*(\d|:)\s*\)\s*=\s*\w+\(\s*\1\s*,\s*\2\s*\)\s*\+\s*(t_[xyz])", body):
        axis = int(a) if a != ":" else int(b)
        mapping[axis - 1] = name
    if sorted(mapping) != [0, 1, 2]:
        raise ValueError(f"translation assignment not recognised: {mapping}")
    return form, mapping


def extrinsics(texts):
    scalars, rotation, steps = parse_camera_params_text(texts["camera_params.m"])
    form, mapping = parse_extrinsic_application(texts["depth_world2rgb_world.m"])
    translation = np.array([scalars[mapping[i]] for i in range(3)])
    # Both supported forms mean X_rgb = R X_d + T for column vectors.
    return dict(R=rotation, T=translation, rotation_steps=steps, rotation_form=form,
                translation_axis_map={str(k): v for k, v in mapping.items()},
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
    """Worst case over pixel-corner rays of |Z_rgb - Z_d| / Z_d on valid pixels."""
    z = np.asarray(z_depth, dtype=np.float64)
    worst = np.zeros(z.shape)
    for offset in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5), (0., 0.)):
        gap = np.abs(rgb_z(z, K, R, T, offset) - z) / z
        worst = np.maximum(worst, np.where(valid, gap, 0.0))
    return np.where(valid, worst, np.nan)
