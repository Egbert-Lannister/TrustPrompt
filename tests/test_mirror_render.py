"""Mirror rendering against an analytic scene (plan section 3.4: symmetric point-cloud test)."""
import numpy as np

from tp.sim.mirror_render import backproject, fit_plane_from_depth, reflect, render_mirror, plane_depth_along_rays

K = np.array([[200.0, 0, 100], [0, 200.0, 75], [0, 0, 1]])
H, W = 150, 200


def test_reflection_is_an_involution_and_flips_signed_distance():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3)); n = np.array([1.0, 2.0, -0.5]); n /= np.linalg.norm(n)
    q = reflect(p, n, 0.7)
    np.testing.assert_allclose(reflect(q, n, 0.7), p, atol=1e-12)
    np.testing.assert_allclose(q @ n - 0.7, -(p @ n - 0.7), atol=1e-12)


def test_side_mirror_shows_the_front_wall_at_its_true_virtual_depth():
    # Front wall at Z = 4 m; a mirror on the side plane X = 1 m (normal towards the camera: -x).
    v, u = np.mgrid[0:H, 0:W].astype(float)
    x_dir = (u - K[0, 2]) / K[0, 0]
    gt = np.full((H, W), 4.0)
    mirror_z = 1.0 / np.where(x_dir > 0, x_dir, np.nan)          # where the ray meets X = 1
    omega = np.isfinite(mirror_z) & (mirror_z < 4.0)              # side-wall pixels in front of the wall
    gt[omega] = mirror_z[omega]
    n, d = np.array([1.0, 0, 0]), 1.0
    out, resolved = render_mirror(gt, np.ones_like(omega), K, omega, n, d)
    # The reflection of the front wall (X<1, Z=4) is X' = 2 - X, Z = 4: virtual depth exactly 4 where visible.
    assert resolved.sum() > 0.5 * omega.sum()
    np.testing.assert_allclose(out[resolved], 4.0, atol=1e-9)
    assert np.all(out[resolved] > mirror_z[resolved])             # always behind the mirror
    assert np.isnan(out[~omega]).all()


def test_plane_fit_recovers_mirror_plane_from_depth():
    v, u = np.mgrid[0:H, 0:W].astype(float)
    n_true = np.array([0.3, -0.2, -1.0]); n_true /= np.linalg.norm(n_true); d_true = -2.5
    z = plane_depth_along_rays(K, (H, W), n_true, d_true)
    mask = np.isfinite(z)
    n, d, r = fit_plane_from_depth(z, K, mask)
    P = backproject(np.where(mask, z, 1.0), K)[mask]
    np.testing.assert_allclose(P @ n - d, 0, atol=1e-9)
    assert r < 1e-9 and d > 0


def test_skewed_intrinsics_round_trip():
    import numpy as np
    from tp.sim.mirror_render import backproject, project
    K = np.array([[500.0, 9.0, 320.5], [0.0, 498.0, 240.2], [0.0, 0.0, 1.0]])    # Hypersim tilt-shift: skew ~ 2 %
    z = np.random.default_rng(0).uniform(0.5, 5.0, (6, 8))
    u, v = project(backproject(z, K), K)
    vv, uu = np.mgrid[0:6, 0:8]
    assert np.allclose(u, uu) and np.allclose(v, vv)


def test_backwall_depth_matches_explicit_ray_tracing():
    import numpy as np
    from tp.sim.mirror_render import backwall_depth
    K = np.array([[60.0, 0.0, 15.5], [0.0, 60.0, 11.5], [0.0, 0.0, 1.0]])
    n = np.array([0.3, -0.1, -1.0]); n /= np.linalg.norm(n); d = -2.0          # mirror ~2 m ahead, facing the camera
    b = 1.2
    z = backwall_depth(K, (24, 32), n, d, b)
    for (y, x) in ((3, 4), (12, 16), (20, 30)):
        r = np.array([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], 1.0])
        t = d / (r @ n); hit = t * r                                       # point on the mirror
        ro = r - 2 * (r @ n) * n                                           # reflected direction (same length as r)
        s = (-b - hit[2]) / ro[2]                                          # reach the wall Z = -b
        path = t * np.linalg.norm(r) + s * np.linalg.norm(ro)
        assert np.isclose(z[y, x], path / np.linalg.norm(r) * r[2], rtol=1e-6)
