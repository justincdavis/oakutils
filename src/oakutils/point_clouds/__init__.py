# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Moudule for creating, filtering, and visualizing point clouds.

Module contents will only be populated if the 'open3d' package is installed.
This is to reduce the minimun required dependencies for the 'oakutils' package.
Since the 'open3d' package is not required for all operations and it is a
large package, it is not included in the 'oakutils' package dependencies
by default.

Submoudles
----------
callbacks
    Callbacks for point cloud creation and filtering.

Classes
-------
PointCloudVisualizer
    Class for visualizing point clouds.

Functions
---------
get_point_cloud_from_np_buffer
    Use to create a point cloud from a numpy array.
filter_point_cloud
    Use to filter a point cloud.
get_point_cloud_from_depth_image
    Use to create a point cloud from a depth image.
get_point_cloud_from_rgb_depth_image
    Use to create a point cloud from a RGB and depth image.
create_point_cloud
    Use to create a point cloud from a RGB and depth image as a callback.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

try:
    from . import callbacks
    from ._classes import PointCloudVisualizer
    from ._funcs import (
        filter_point_cloud,
        get_point_cloud_from_depth_image,
        get_point_cloud_from_np_buffer,
        get_point_cloud_from_rgb_depth_image,
    )
    from .callbacks import create_point_cloud

    __all__ = [
        "PointCloudVisualizer",
        "callbacks",
        "create_point_cloud",
        "filter_point_cloud",
        "get_point_cloud_from_depth_image",
        "get_point_cloud_from_np_buffer",
        "get_point_cloud_from_rgb_depth_image",
    ]

    _log.debug("Loaded point_clouds")
except ImportError:
    _log.info(
        "The 'open3d' package is not installed. Point cloud submodule will not be available.",
    )
