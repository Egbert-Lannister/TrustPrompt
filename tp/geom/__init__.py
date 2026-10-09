"""Geometry with metric axial Z and explicit pixel-centre conventions."""

from .camera import (
    backproject,
    fit_inverse_depth_plane,
    inverse_plane_depth,
    nearest_fill,
    pixel_rays,
    plane_depth,
    project,
    resize_intrinsics,
    resize_nearest,
)

__all__ = [
    "backproject", "fit_inverse_depth_plane", "inverse_plane_depth", "nearest_fill",
    "pixel_rays", "plane_depth", "project", "resize_intrinsics", "resize_nearest",
]
