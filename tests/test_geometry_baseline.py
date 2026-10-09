from types import SimpleNamespace

import numpy as np
import pytest

from baselines.raw_prompt import predict
from tp.geom.camera import (
    backproject, fit_inverse_depth_plane, inverse_plane_depth, nearest_fill,
    plane_depth, project, resize_intrinsics, resize_nearest,
)


def camera():
    return np.array([[8., .2, 3.5], [0., 7., 2.5], [0., 0., 1.]])


def test_pinhole_roundtrip_axial_z_and_camera_axes():
    depth = np.linspace(1., 4., 48).reshape(6, 8)
    points = backproject(depth, camera())
    pixels, z = project(points, camera())
    y, x = np.indices(depth.shape)
    np.testing.assert_allclose(pixels, np.stack((x, y), axis=-1), atol=1e-13)
    np.testing.assert_array_equal(z, depth)
    assert points[0, 0, 0] < 0 and points[0, 0, 1] < 0
    assert points[-1, -1, 0] > 0 and points[-1, -1, 1] > 0


def test_invalid_depth_and_behind_camera_projection_are_nan():
    points = backproject(np.array([[0., np.inf], [-1., np.nan]]), camera())
    assert np.isnan(points).all()
    pixels, z = project(np.array([[1., 2., -3.], [1., 2., 0.]]), camera())
    assert np.isnan(pixels).all()
    np.testing.assert_array_equal(z, [-3., 0.])


def test_exact_tilted_inverse_plane_fit_and_explicit_plane():
    coefficients = np.array([.10, -.20, .6])
    depth = inverse_plane_depth(coefficients, camera(), (6, 8))
    mask = np.zeros((6, 8), dtype=bool)
    mask[::2, ::2] = True
    fitted = fit_inverse_depth_plane(depth, camera(), mask)
    np.testing.assert_allclose(fitted, coefficients, atol=1e-14)
    np.testing.assert_allclose(plane_depth(-coefficients, 1., camera(), (6, 8)), depth)
    xyz = backproject(depth, camera())
    np.testing.assert_allclose(xyz @ coefficients, 1., atol=1e-14)


def test_degenerate_plane_fit_raises_and_negative_depth_not_rendered():
    with pytest.raises(ValueError, match="non-collinear"):
        fit_inverse_depth_plane(np.ones((1, 4)), camera())
    with pytest.raises(ValueError, match="three"):
        fit_inverse_depth_plane(np.ones((1, 2)), camera())
    assert np.isnan(inverse_plane_depth([0., 0., -1.], camera(), (2, 2))).all()


def test_resize_intrinsics_matches_half_pixel_affine_coordinates():
    source_size, destination_size = (6, 8), (3, 12)
    original = camera()
    resized = resize_intrinsics(original, source_size, destination_size)
    point = np.array([[.4, -.3, 2.]])
    source_uv, _ = project(point, original)
    destination_uv, _ = project(point, resized)
    scale = np.array([1.5, .5])
    np.testing.assert_allclose(destination_uv, (source_uv + .5) * scale - .5)
    np.testing.assert_array_equal(resized[2], [0., 0., 1.])
    np.testing.assert_array_equal(original, camera())
    np.testing.assert_allclose(resize_intrinsics(resized, destination_size, source_size), original)


def test_nearest_resize_uses_same_centre_grid_for_depth_and_mask():
    depth = np.arange(12).reshape(3, 4)
    resized = resize_nearest(depth, (2, 2))
    np.testing.assert_array_equal(resized, [[1, 3], [9, 11]])
    mask = depth % 2 == 1
    np.testing.assert_array_equal(resize_nearest(mask, (2, 2)), resized % 2 == 1)
    assert resize_nearest(mask, (2, 2)).dtype == bool


def test_raw_prompt_fills_before_resize_without_gt_and_preserves_input():
    prompt = np.array([[1., 0., 0., 0., 5.]], dtype=np.float32)
    valid = prompt > 0
    before = prompt.copy()
    sample = SimpleNamespace(prompt=prompt, prompt_valid=valid,
                             rgb=np.zeros((1, 3, 3), dtype=np.uint8))
    prediction = predict(sample)  # Deliberately no gt attribute.
    expected = resize_nearest(nearest_fill(prompt, valid), (1, 3))
    np.testing.assert_array_equal(prediction, expected)
    np.testing.assert_array_equal(prompt, before)
    np.testing.assert_array_equal(valid, before > 0)
    assert prediction.dtype == np.float32


def test_raw_prompt_native_fill_and_resize_order_matters():
    prompt = np.array([[1., 0., 0., 4.]], dtype=np.float32)
    valid = prompt > 0
    sample = SimpleNamespace(prompt=prompt, prompt_valid=valid,
                             rgb=np.zeros((1, 2, 3), dtype=np.uint8))
    np.testing.assert_array_equal(predict(sample), [[1., 4.]])
    # Resizing first loses the left valid source and incorrectly fills with 4.
    resize_first = nearest_fill(resize_nearest(prompt, (1, 2)), resize_nearest(valid, (1, 2)))
    np.testing.assert_array_equal(resize_first, [[4., 4.]])


def test_nearest_fill_respects_valid_mask_and_all_holes_stay_invalid():
    np.testing.assert_array_equal(nearest_fill(np.array([[2., 99., 4.]]),
                                             np.array([[True, False, True]])), [[2., 2., 4.]])
    sample = SimpleNamespace(prompt=np.array([[0., np.nan], [np.inf, -1.]]),
                             prompt_valid=np.ones((2, 2), dtype=bool),
                             rgb=np.zeros((3, 5, 3), dtype=np.uint8))
    assert np.isnan(predict(sample)).all()
    assert predict(sample).shape == (3, 5)
