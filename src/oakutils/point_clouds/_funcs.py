from typing import Optional

import cv2
import numpy as np
import open3d


def get_point_cloud_from_rgb_depth_image(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: open3d.camera.PinholeCameraIntrinsic,
) -> open3d.geometry.PointCloud:
    """
    Creates an Open3D point cloud from an RGB and a depth image.
    The depth image is assumed to be in millimeters (the default for the Oak-D cameras).
    The RGB image is assumed to be in BGR format (the default for OpenCV).
    The camera intrinsics are provided in the open3d.camera.PinholeCameraIntrinsics format.

    :param rgb_image: The RGB image to use.
    :type rgb_image: np.ndarray
    :param depth_image: The depth image to use.
    :type depth_image: np.ndarray
    :param camera_intrinsics: The camera intrinsics to use.
    :type camera_intrinsics: open3d.camera.PinholeCameraIntrinsic
    :return: The point cloud created from the RGB and depth images.
    :rtype: open3d.geometry.PointCloud
    """

    if (
        rgb_image.shape[0] != depth_image.shape[0]
        or rgb_image.shape[1] != depth_image.shape[1]
    ):
        rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

    rgb_open3d = open3d.geometry.Image(rgb_image)
    depth_open3d = open3d.geometry.Image(depth_image)

    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb_open3d, depth_open3d)

    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics,
    )

    return pcd


def get_point_cloud_from_depth_image(
    depth_image: np.ndarray,
    camera_intrinsics: open3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 1000.0,
    depth_trunc: float = 25000.0,
    stride: int = 1,
    project_valid_depth_only: bool = True,
) -> open3d.geometry.PointCloud:
    """
    Creates an Open3D point cloud from a depth image.
    The depth image is assumed to be in millimeters (the default for the Oak-D cameras).
    The camera intrinsics are provided in the open3d.camera.PinholeCameraIntrinsics format.

    :param depth_image: The depth image to use.
    :type depth_image: np.ndarray
    :param camera_intrinsics: The camera intrinsics to use.
    :type camera_intrinsics: open3d.camera.PinholeCameraIntrinsic
    :param depth_scale: Depth scaling factor. Defaults to 1000.0 to convert from millimeters to meters.
    :type depth_scale: float
    :param depth_trunc: Truncated depth values to this value. Defaults to 25000.0 to truncate depth values to 25 meters.
    :type depth_trunc: float
    :param stride: Sampling factor to support coarse point cloud extraction. Defaults to 1.
    :type stride: int
    :param project_valid_depth_only: If True, only projects pixels with valid depth values. Defaults to True.
    :type project_valid_depth_only: bool
    :return: The point cloud created from the depth image.
    :rtype: open3d.geometry.PointCloud
    """
    depth_open3d = open3d.geometry.Image(depth_image)

    pcd = open3d.geometry.PointCloud.create_from_depth_image(
        depth_open3d,
        camera_intrinsics,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        stride=stride,
        project_valid_depth_only=project_valid_depth_only,
    )

    return pcd


def filter_point_cloud(
    pcd: open3d.geometry.PointCloud,
    voxel_size: Optional[float] = 0.1,
    nb_neighbors: Optional[int] = 30,
    std_ratio: Optional[float] = 0.1,
    downsample_first: bool = True,
) -> open3d.geometry.PointCloud:
    """
    Filters the point cloud by performing voxel downsampling and outlier removal.

    :param pcd: The point cloud to filter.
    :type pcd: open3d.geometry.PointCloud
    :param voxel_size: The voxel size to use for downsampling. Defaults to 0.1.
    :type voxel_size: float or None
    :param nb_neighbors: The number of neighbors to use for outlier removal. Defaults to 30.
    :type nb_neighbors: int or None
    :param std_ratio: The standard deviation ratio to use for outlier removal. Defaults to 0.1.
    :type std_ratio: float or None
    :param downsample_first: If True, performs voxel downsampling first, then outlier removal.
        If False, performs outlier removal first, then voxel downsampling. Defaults to True.
    :type downsample_first: bool
    :return: The filtered point cloud.
    :rtype: open3d.geometry.PointCloud
    """
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
