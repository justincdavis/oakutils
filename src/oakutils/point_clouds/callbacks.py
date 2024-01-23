# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Callbacks for point cloud creation and filtering.

Functions
---------
create_point_cloud
    Use to create a point cloud from a RGB and depth image.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._funcs import filter_point_cloud, get_point_cloud_from_rgb_depth_image

if TYPE_CHECKING:
    import numpy as np
    import open3d as o3d  # type: ignore[import]


def create_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_trunc: float = 25000.0,
    depth_scale: float = 1000.0,
    voxel_size: float = 0.1,
    nb_neighbors: int = 30,
    std_ratio: float = 0.1,
    *,
    filter_pc: bool | None = None,
    downsample_first: bool | None = None,
) -> o3d.geometry.PointCloud:
    """
    Use to create a point cloud from a RGB and depth image.

    Parameters
    ----------
    rgb : np.ndarray
        The RGB image to use.
    depth : np.ndarray
        The depth image to use.
    camera_intrinsics : o3d.camera.PinholeCameraIntrinsic
        The camera intrinsics to use.
    depth_trunc : float, optional
        Truncated depth values to this value. Defaults to 25000.0 to truncate depth
          values to 25 meters.
    depth_scale : float, optional
        Depth scaling factor. Defaults to 1000.0 to convert from millimeters to meters.
    filter_pc : bool, optional
        If True, filters the point cloud. Defaults to True.
    voxel_size : float, optional
        Voxel size to use for downsampling. Defaults to 0.1.
    nb_neighbors : int, optional
        Number of neighbors to use for outlier removal. Defaults to 30.
    std_ratio : float, optional
        Standard deviation ratio to use for outlier removal. Defaults to 0.1.
    downsample_first : bool, optional
        If True, downsamples the point cloud before outlier removal. Defaults to True.

    Returns
    -------
    o3d.geometry.PointCloud
        The point cloud created from the RGB and depth image.
    """
    # Create point cloud from RGB and depth image
    pcd = get_point_cloud_from_rgb_depth_image(
        rgb,
        depth,
        camera_intrinsics,
        depth_trunc=depth_trunc,
        depth_scale=depth_scale,
    )

    # Filter point cloud
    if filter_pc is None:
        filter_pc = True

    if filter_pc:
        pcd = filter_point_cloud(
            pcd,
            voxel_size=voxel_size,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            downsample_first=downsample_first,
        )

    return pcd
