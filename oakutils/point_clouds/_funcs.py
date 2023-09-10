from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import open3d as o3d

if TYPE_CHECKING:
    import numpy as np


def get_point_cloud_from_rgb_depth_image(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_trunc: float = 25000.0,
    depth_scale: float = 1000.0,
) -> o3d.geometry.PointCloud:
    """Creates an o3d point cloud from an RGB and a depth image.

    Parameters
    ----------
    rgb_image : np.ndarray
        The RGB image to use.
    depth_image : np.ndarray
        The depth image to use.
    camera_intrinsics : o3d.camera.PinholeCameraIntrinsic
        The camera intrinsics to use.
    depth_trunc : float, optional
        Truncated depth values to this value. Defaults to 25000.0 to truncate depth
        values to 25 meters.
    depth_scale : float, optional
        Depth scaling factor. Defaults to 1000.0 to convert from millimeters to meters.

    Returns
    -------
    o3d.geometry.PointCloud
        The point cloud created from the RGB and depth images.
    """
    if (
        rgb_image.shape[0] != depth_image.shape[0]
        or rgb_image.shape[1] != depth_image.shape[1]
    ):
        rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    rgb_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_trunc=depth_trunc, depth_scale=depth_scale
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics,
    )


def get_point_cloud_from_depth_image(
    depth_image: np.ndarray,
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 1000.0,
    depth_trunc: float = 25000.0,
    stride: int = 1,
    project_valid_depth_only: bool | None = None,
) -> o3d.geometry.PointCloud:
    """Creates an o3d point cloud from a depth image.

    Parameters
    ----------
    depth_image : np.ndarray
        The depth image to use.
    camera_intrinsics : o3d.camera.PinholeCameraIntrinsic
        The camera intrinsics to use.
    depth_scale : float, optional
        Depth scaling factor. Defaults to 1000.0 to convert from millimeters to meters.
    depth_trunc : float, optional
        Truncated depth values to this value. Defaults to 25000.0 to truncate depth
          values to 25 meters.
    stride : int, optional
        Sampling factor to support coarse point cloud extraction. Defaults to 1.
    project_valid_depth_only : bool, optional
        If True, only projects pixels with valid depth values. Defaults to True.

    Returns
    -------
    o3d.geometry.PointCloud
        The point cloud created from the depth image.
    """
    if project_valid_depth_only is None:
        project_valid_depth_only = True

    depth_o3d = o3d.geometry.Image(depth_image)

    return o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        camera_intrinsics,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        stride=stride,
        project_valid_depth_only=project_valid_depth_only,
    )


def filter_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float | None = 0.1,
    nb_neighbors: int | None = 30,
    std_ratio: float | None = 0.1,
    downsample_first: bool | None = None,
) -> o3d.geometry.PointCloud:
    """Filters the point cloud by performing voxel downsampling and outlier removal.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The point cloud to filter.
    voxel_size : float or None, optional
        The voxel size to use for downsampling. Defaults to 0.1.
    nb_neighbors : int or None, optional
        The number of neighbors to use for outlier removal. Defaults to 30.
    std_ratio : float or None, optional
        The standard deviation ratio to use for outlier removal. Defaults to 0.1.
    downsample_first : bool, optional
        If True, performs voxel downsampling first, then outlier removal.
        If False, performs outlier removal first, then voxel downsampling.
          Defaults to True.

    Returns
    -------
    o3d.geometry.PointCloud
        The filtered point cloud.
    """
    if downsample_first is None:
        downsample_first = True

    if downsample_first:
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if nb_neighbors is not None and std_ratio is not None:
            pcd = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)[0]
    else:
        if nb_neighbors is not None and std_ratio is not None:
            pcd = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)[0]
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pcd


def create_point_cloud_from_np(pcl_data: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert a numpy array to an open3d point cloud.
    The numpy array should be of shape (N, 3) or (N, 4) where N is the number of points.

    Parameters
    ----------
    pcl_data : np.ndarray
        The numpy array to convert.

    Returns
    -------
    o3d.geometry.PointCloud
        The open3d point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl_data)
    pcd.remove_non_finite_points()
    return pcd
