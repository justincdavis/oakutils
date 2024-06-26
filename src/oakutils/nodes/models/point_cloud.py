# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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

from oakutils.nodes.neural_network import get_nn_data
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


def get_point_cloud_buffer(
    data: dai.NNData,
    frame_size: tuple[int, int] = (640, 400),
    scale: float = 1000.0,
    *,
    remove_zeros: bool | None = None,
) -> np.ndarray:
    """
    Use to convert the raw data output from a neural network execution and converts it to a point cloud.

    Parameters
    ----------
    data : dai.NNData
        Raw data output from a neural network execution.
    frame_size : tuple[int, int], optional
        The size of the buffer, by default (640, 400)
        Usually this will be the size of the depth frame.
        Which inherits its shape from the MonoCamera resolutions.
    scale: float, optional
        The scale to apply to the point cloud, by default 1000.0
        This will convert from mm to m.
    remove_zeros: bool, optional
        Whether to remove zero points, by default None
        If None, then True is used
        Recommended to set to True to remove zero points
        Can speedup reading and filtering of the point cloud
        by up to 10x

    Returns
    -------
    np.ndarray
        Point cloud buffer

    """
    if remove_zeros is None:
        remove_zeros = True

    pcl_data: np.ndarray = get_nn_data(
        data,
        reshape_to=(
            1,
            3,
            frame_size[1],
            frame_size[0],
        ),
        use_first_layer=True,
    )
    pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / scale

    if remove_zeros:
        # optimization over an np.all since it performs less checks
        # and realisticlly it does not matter if there is a few points
        # difference over hundreds of interations
        zero_val = 0.0
        pcl_data = pcl_data[pcl_data[:, 2] != zero_val]

    return pcl_data
