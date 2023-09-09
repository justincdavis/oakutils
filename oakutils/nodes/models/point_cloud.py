from __future__ import annotations

from typing import Tuple

import depthai as dai

from ._load import create_no_args_multi_link_model as _create_no_args_multi_link_model


def create_point_cloud(
    pipeline: dai.Pipeline,
    xyz_link: dai.Node.Output,
    depth_link: dai.Node.Output,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a point_cloud model with a specified kernel size

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
    dai.node.XLinkOut
        The output link of the point_cloud node
    str
        The name of the stream, determined by the model_type and attributes

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
