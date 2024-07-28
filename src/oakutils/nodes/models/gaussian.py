# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for creating gaussian models.

Functions
---------
create_gaussian
    Creates a gaussian model with a specified kernel size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._load import create_single_kernel_model as _create_single_kernel_model

if TYPE_CHECKING:
    import depthai as dai


def create_gaussian(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    kernel_size: int = 3,
    shaves: int = 1,
    *,
    grayscale_out: bool | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to create a gaussian model with a specified kernel size.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the gaussian to
    input_link : dai.node.XLinkOut
        The input link to connect to the gaussian node.
        Example: cam_rgb.preview.link
        Explicitly pass in the link as a non-called function.
    kernel_size : int, optional
        The size of the gaussian kernel, by default 3
        Must be an odd integer
        Must be between 3 and 15 (since these are the compiled model sizes)
    shaves : int, optional
        The number of shaves to use, by default 1
        Must be between 1 and 6
    grayscale_out : bool, optional
        Whether or not to use graycale output, by default False

    Returns
    -------
    dai.node.NeuralNetwork
        The gaussian node

    """
    if grayscale_out is None:
        grayscale_out = False

    model_type = "laplacian"
    if grayscale_out:
        model_type += "gray"
    return _create_single_kernel_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_type,
        kernel_size=kernel_size,
        shaves=shaves,
    )
