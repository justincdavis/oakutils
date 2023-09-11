from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._load import create_no_args_multi_link_model as _create_no_args_multi_link_model

if TYPE_CHECKING:
    import depthai as dai


def create_xyz_matrix(width: int, height: int, camera_matrix: np.ndarray) -> np.ndarray:
    """Creates a constant reprojection matrix for the given camera matrix and image size.
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
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
    points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

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

    xyz = np.stack([x_coord, y_coord], axis=-1)
    xyz = np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
    return np.array([xyz], dtype=np.float16).view(np.int8)


def create_point_cloud(
    pipeline: dai.Pipeline,
    xyz_link: dai.Node.Output,
    depth_link: dai.Node.Output,
) -> dai.node.NeuralNetwork:
    """Creates a point_cloud model with a specified kernel size.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the point_cloud to
    xyz_link : dai.Node.Output
        The output link of the xyz node
    depth_link : dai.Node.Output
        The output link of the depth node
        Example: stereo.depth
        Explicity pass the object without calling (i.e. not stereo.depth())

    Returns
    -------
    dai.node.NeuralNetwork
        The point_cloud node

    Raises
    ------
    ValueError
        If the kernel_size is invalid
    """
    model_type = "pointcloud"
    return _create_no_args_multi_link_model(
        pipeline=pipeline,
        input_link=[xyz_link, depth_link],
        model_name=model_type,
        input_names=["xyz", "depth"],
        reuse_messages=[True, None],
    )
