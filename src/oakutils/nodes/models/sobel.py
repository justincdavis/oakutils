# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Models for the sobel node.

Functions
---------
create_sobel
    Creates a sobel model as a node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._load import create_no_args_model as _create_no_args_model
from ._load import create_single_kernel_model as _create_single_kernel_model

if TYPE_CHECKING:
    import depthai as dai


def create_sobel(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blur_kernel_size: int = 3,
    shaves: int = 1,
    *,
    use_blur: bool | None = None,
    grayscale_out: bool | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to create a sobel model with a specified kernel size.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the sobel to
    input_link : dai.node.XLinkOut
        The input link to connect to the sobel node.
        Example: cam_rgb.preview.link
        Explicitly pass in the link as a non-called function.
    blur_kernel_size : int, optional
        The size of the blur kernel, by default 3
        Only used when use_blur is True
        Must be an odd integer
        Must be between 3 and 15 (since these are the compiled model sizes)
    shaves : int, optional
        The number of shaves to use, by default 1
        Must be between 1 and 6
    use_blur : bool, optional
        Whether or not to use a blur before the sobel, by default False
    grayscale_out : bool, optional
        Whether or not to use graycale output, by default False

    Returns
    -------
    dai.node.NeuralNetwork
        The sobel node

    """
    if use_blur is None:
        use_blur = False
    if grayscale_out is None:
        grayscale_out = False

    model_type = "sobel"
    if use_blur:
        model_type += "blur"
    if grayscale_out:
        model_type += "gray"

    if use_blur:
        return _create_single_kernel_model(
            pipeline=pipeline,
            input_link=input_link,
            model_name=model_type,
            kernel_size=blur_kernel_size,
            shaves=shaves,
        )
    return _create_no_args_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_type,
        shaves=shaves,
    )
