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
Tools for working with depth images.

Functions
---------
align_depth_to_rgb
    Use to align a depth image to an RGB image.
quantize_colormap_depth_frame
    Further quantize the depth image for nice visualization.
overlay_depth_frame
    Overlay the depth map on top of the RGB image.
"""
from __future__ import annotations

import cv2  # type: ignore[import]
import numpy as np


def align_depth_to_rgb(
    depth_image: np.ndarray,
    pixel_coords: np.ndarray,
    inverse_depth_intrinsic: np.ndarray,
    rgb_intrinsic: np.ndarray,
    depth_to_rgb_extrinsic: np.ndarray,
    rgb_width: int,
    rgb_height: int,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Use to align a depth image to an RGB image.

    Parameters
    ----------
    depth_image : np.ndarray
        The depth image to align.
    pixel_coords : np.ndarray
        The pixel coordinates of the depth image.
    inverse_depth_intrinsic : np.ndarray
        The inverse depth intrinsic matrix.
    rgb_intrinsic : np.ndarray
        The RGB intrinsic matrix.
    depth_to_rgb_extrinsic : np.ndarray
        The depth to RGB extrinsic matrix.
    rgb_width : int
        The width of the RGB image.
    rgb_height : int
        The height of the RGB image.
    depth_scale : float, optional
        Depth scaling factor. Defaults to 1000.0 to convert from millimeters to meters.

    Returns
    -------
    np.ndarray
        The aligned depth image.

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-pointcloud/rgbd-pointcloud/utils.py
    """
    # depth to 3d coordinates [x, y, z]
    cam_coords = (
        np.dot(inverse_depth_intrinsic, pixel_coords)
        * depth_image.flatten().astype(float)
        / depth_scale
    )

    # move the depth image 3d coordinates to the rgb camera  location
    cam_coords_homogeneous = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
    depth_points_homogeneous = np.dot(depth_to_rgb_extrinsic, cam_coords_homogeneous)

    # project the 3d depth points onto the rgb image plane
    rgb_frame_ref_cloud = depth_points_homogeneous[:3, :]
    rgb_frame_ref_cloud_normalized = rgb_frame_ref_cloud / rgb_frame_ref_cloud[2, :]
    rgb_image_pts = np.matmul(rgb_intrinsic, rgb_frame_ref_cloud_normalized)
    rgb_image_pts = rgb_image_pts.astype(np.int16)
    u_v_z = np.vstack((rgb_image_pts, rgb_frame_ref_cloud[2, :]))
    lft = np.logical_and(u_v_z[0] >= 0, u_v_z[0] < rgb_width)
    rgt = np.logical_and(u_v_z[1] >= 0, u_v_z[1] < rgb_height)
    idx_bool = np.logical_and(lft, rgt)
    u_v_z_sampled = u_v_z[:, np.where(idx_bool)]
    y_idx = u_v_z_sampled[1].astype(int)
    x_idx = u_v_z_sampled[0].astype(int)

    # place the valid aligned points into a new depth image
    aligned_depth_image: np.ndarray = np.full(
        (rgb_height, rgb_width),
        0,
        dtype=np.uint16,
    )
    aligned_depth_image[y_idx, x_idx] = u_v_z_sampled[3] * depth_scale
    return aligned_depth_image


def quantize_colormap_depth_frame(
    frame: np.ndarray,
    depth_scale_factor: float = 2.0,
    *,
    apply_colormap: bool | None = None,
) -> np.ndarray:
    """
    Further quantize the depth image for nice visualization.

    Parameters
    ----------
    frame : np.ndarray
        Depth map image
    depth_scale_factor : float
        Scale factor to apply to the depth map before quantization
    apply_colormap : bool
        Whether to apply a colormap to the depth map

    Returns
    -------
    np.ndarray
        Quantized depth map image

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-pointcloud/rgbd-pointcloud/utils.py
    """
    if apply_colormap is None:
        apply_colormap = True
    quantized_depth: np.ndarray = cv2.convertScaleAbs(
        frame.astype(float),
        alpha=255 / depth_scale_factor,
    )
    if apply_colormap:
        quantized_depth = cv2.applyColorMap(quantized_depth, cv2.COLORMAP_JET)
    return quantized_depth


def overlay_depth_frame(
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    rgb_alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay the depth map on top of the RGB image.

    Parameters
    ----------
    rgb_frame : np.ndarray
        RGB image
    depth_frame : np.ndarray
        Depth map image
    rgb_alpha : float
        Alpha value for blending the RGB and depth map images

    Returns
    -------
    np.ndarray
        Blended RGB and depth map image

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-pointcloud/rgbd-pointcloud/utils.py
    """
    depth_three_channel: np.ndarray = np.zeros_like(rgb_frame)
    depth_three_channel[:, :, 2] = depth_frame
    cond = depth_three_channel[:, :, 2] > 0
    depth_three_channel[cond, 2] = 255
    # Blend aligned depth + rgb image
    blended_image: np.ndarray = (1.0 - rgb_alpha) * depth_three_channel.astype(
        float,
    ) + rgb_alpha * rgb_frame.astype(float)
    blended_max: float = blended_image.max()
    return (255 * blended_image.astype(float) / blended_max).astype(np.uint8)
