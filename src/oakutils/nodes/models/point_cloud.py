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
Module for creating a point cloud model onboard.

Functions
---------
create_xyz_matrix
    Use to create a constant reprojection matrix for the given camera matrix and image size.
create_point_cloud
    Use to create a point_cloud model.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import depthai as dai
import numpy as np

from oakutils.nodes.xin import create_xin

from ._load import create_no_args_multi_link_model as _create_no_args_multi_link_model

if TYPE_CHECKING:
    from oakutils.calibration import CalibrationData


def create_xyz_matrix(width: int, height: int, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Use to create a constant reprojection matrix for the given camera matrix and image size.

    Note:
    This is for generating the input to the point cloud generation model.

    Parameters
    ----------
    width : int
        The width of the image
    height : int
        The height of the image
    camera_matrix : np.ndarray
        The camera matrix to use for the reprojection
        This should be a 3x3 matrix

    Returns
    -------
    np.ndarray
        The reprojection matrix
    """
    xs: np.ndarray = np.linspace(0, width - 1, width, dtype=np.float32)
    ys: np.ndarray = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid: np.ndarray = np.stack(np.meshgrid(xs, ys))  # WxHx2
    points_2d: np.ndarray = base_grid.transpose(1, 2, 0)  # 1xHxWx2

    # unpack coordinates
    u_coord: np.ndarray = points_2d[..., 0]
    v_coord: np.ndarray = points_2d[..., 1]

    # unpack intrinsics
    fx: np.ndarray = camera_matrix[0, 0]
    fy: np.ndarray = camera_matrix[1, 1]
    cx: np.ndarray = camera_matrix[0, 2]
    cy: np.ndarray = camera_matrix[1, 2]

    # projective
    x_coord: np.ndarray = (u_coord - cx) / fx
    y_coord: np.ndarray = (v_coord - cy) / fy

    xyz: np.ndarray = np.stack([x_coord, y_coord], axis=-1)
    xyz = np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
    return np.array([xyz], dtype=np.float16).view(np.int8)


def create_point_cloud(
    pipeline: dai.Pipeline,
    depth_link: dai.Node.Output,
    calibration: CalibrationData,
    input_stream_name: str = "xyz_to_pcl",
    shaves: int = 4,
) -> tuple[dai.node.NeuralNetwork, dai.node.XLinkIn, Callable[[dai.Device], None]]:
    """
    Use to create a point_cloud model with a specified kernel size.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the point_cloud to
    depth_link : dai.Node.Output
        The output link of the depth node
        Example: stereo.depth
        Explicity pass the object without calling (i.e. not stereo.depth())
    calibration : CalibrationData
        The calibration data for the camera
    input_stream_name : str, optional
        The name of the input stream, by default "xyz_to_pcl"
    shaves : int, optional
        The number of shaves to use, by default 4
        Must be between 1 and 6

    Returns
    -------
    dai.node.NeuralNetwork
        The point_cloud node
    dai.node.XLinkIn
        The input link to connect to the point_cloud node.
    Callable[[dai.Device], None]
        Function to pass the device, which will start the point cloud generation
    """
    model_type = "pointcloud"
    xin = create_xin(pipeline, input_stream_name)
    point_cloud_node = _create_no_args_multi_link_model(
        pipeline=pipeline,
        input_links=[xin.out, depth_link],
        model_name=model_type,
        input_names=["xyz", "depth"],
        input_sizes=[1, 1],
        input_blocking=[True, False],
        reuse_messages=[True, None],
        shaves=shaves,
    )

    xyz = create_xyz_matrix(
        calibration.left.size[0],
        calibration.left.size[1],
        calibration.left.K,
    )

    def _start_point_cloud(device: dai.Device, xyz: np.ndarray) -> None:
        buff = dai.Buffer()
        buff.setData(xyz)
        device.getInputQueue(input_stream_name).send(buff)

    return point_cloud_node, xin, partial(_start_point_cloud, xyz=xyz)
