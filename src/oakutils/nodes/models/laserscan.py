# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Models for the laserscan node.

Functions
---------
create_laserscan
    Creates a laserscan model as a node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oakutils.nodes.neural_network import get_nn_data

from ._load import create_laserscan_model as _create_laserscan_model

if TYPE_CHECKING:
    import depthai as dai
    import numpy as np


def create_laserscan(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    width: int = 10,
    scans: int = 1,
    shaves: int = 1,
) -> dai.node.NeuralNetwork:
    """
    Use to create a laserscan model with a specified width.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the laserscan to
    input_link : dai.node.XLinkOut
        The input link to connect to the laserscan node.
        Example: stereo.depth.link
        Explicitly pass in the link as a non-called function.
    width : int, optional
        The width of the laserscan, by default 10
        Options are [5, 10, 20]
    scans : int, optional
        The number of scans to use, by default 1
        Options are [1, 3, 5]
        Scans are horizontal lines of depth data, each scan
        is sampled from a different row of the depth image, with
        even spacing. A center scan is always generated and is
        always the middle entry in the output scans.
    shaves : int, optional
        The number of shaves to use, by default 1
        Must be between 1 and 6

    Returns
    -------
    dai.node.NeuralNetwork
        The laserscan node

    Raises
    ------
    ValueError
        If the width is invalid

    """
    model_type = "laserscan"

    valid_widths: list[int] = [5, 10, 20]
    if width not in valid_widths:
        err_msg = "Invalid width, must be one of [5, 10, 20]"
        raise ValueError(err_msg)
    valid_scans: list[int] = [1, 3, 5]
    if scans not in valid_scans:
        err_msg = "Invalid scans, must be one of [1, 3, 5]"
        raise ValueError(err_msg)

    return _create_laserscan_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_type,
        width=width,
        scans=scans,
        shaves=shaves,
    )


def get_laserscan(data: dai.NNData) -> np.ndarray:
    """
    Use to get the laserscan data from the output tensor.

    Parameters
    ----------
    data : dai.NNData
        The data from the output tensor

    Returns
    -------
    np.ndarray
        The laserscan data

    """
    return get_nn_data(data, use_first_layer=True)
