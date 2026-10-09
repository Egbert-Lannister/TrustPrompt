"""Synthetic checks for the NYU RGB-Z bound (plan revision R3). No real calibration here."""
import numpy as np
import pytest

from tp.data.nyu_rgbz import (extrinsics, parse_camera_params_text, parse_extrinsic_application,
                              relative_gap, rgb_z)

CAMERA = """
% synthetic file in MATLAB syntax; numbers are NOT NYU's
fx_rgb = 5.0e+02;
R = -[ 9.9e-01, 1.0e-02, 0.0e+00; ...
      -1.0e-02, 9.9e-01, 0.0e+00; ...
       0.0e+00, 0.0e+00, 1.0e+00 ];
R = inv(R');
t_x = 2.5e-02;
t_z = -3.0e-04;
t_y = 6.0e-04;
"""
APPLY = """
function points3d = depth_world2rgb_world(points3d)
  camera_params;
  points3d = R * points3d';
  points3d(1,:) = points3d(1,:) + t_x;
  points3d(2,:) = points3d(2,:) + t_z;   % axis map is parsed, not assumed
  points3d(3,:) = points3d(3,:) + t_y;
  points3d = points3d';
end
"""


def test_parser_follows_statement_sequence_and_axis_map():
    scalars, R, steps = parse_camera_params_text(CAMERA)
    literal = -np.array([[.99, .01, 0], [-.01, .99, 0], [0, 0, 1]])
    np.testing.assert_allclose(R, np.linalg.inv(literal.T))
    assert steps == ["literal_negated", "inv(R')"]
    form, mapping = parse_extrinsic_application(APPLY)
    assert form == "R_times_columns" and mapping == {0: "t_x", 1: "t_z", 2: "t_y"}
    e = extrinsics({"camera_params.m": CAMERA, "depth_world2rgb_world.m": APPLY})
    np.testing.assert_allclose(e["T"], [2.5e-2, -3e-4, 6e-4])


def test_unknown_statements_raise():
    with pytest.raises(ValueError):
        parse_camera_params_text(CAMERA + "\nR = R * 2;\n")
    with pytest.raises(ValueError):
        parse_extrinsic_application("points3d = foo(points3d);")


def test_inversion_recovers_rgb_z_exactly():
    a = 0.02
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T = np.array([0.025, 0.001, 0.004])
    K = np.array([[500., 0, 32], [0, 500, 24], [0, 0, 1]])
    h, w = 48, 64
    v, u = np.mgrid[0:h, 0:w].astype(float)
    z_rgb = 1.5 + 0.01 * u
    X_rgb = np.stack([(u - 32) / 500 * z_rgb, (v - 24) / 500 * z_rgb, z_rgb], -1)
    z_d = ((X_rgb - T) @ np.linalg.inv(R).T)[..., 2]
    np.testing.assert_allclose(rgb_z(z_d, K, R, T), z_rgb, rtol=1e-12)


def test_identity_extrinsics_have_zero_gap_and_invalid_is_nan():
    z = np.full((4, 5), 2.0)
    valid = np.ones((4, 5), bool); valid[0, 0] = False
    gap = relative_gap(z, valid, np.diag([500., 500, 1]), np.eye(3), np.zeros(3))
    assert np.isnan(gap[0, 0])
    np.testing.assert_allclose(gap[valid], 0, atol=1e-15)
