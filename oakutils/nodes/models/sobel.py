from typing import Tuple

import depthai as dai

from ._load import create_no_args_model as _create_no_args_model
from ._load import create_single_kernel_model as _create_single_kernel_model


def create_sobel(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blur_kernel_size: int = 3,
    use_blur: bool = False,
    grayscale_out: bool = False,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a sobel model with a specified kernel size

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
    use_blur : bool, optional
        Whether or not to use a blur before the sobel, by default False
    grayscale_out : bool, optional
        Whether or not to use graycale output, by default False

    Returns
    -------
    dai.node.NeuralNetwork
        The sobel node
    dai.node.XLinkOut
        The output link of the sobel node
    str
        The name of the stream, determined by the model_type and attributes

    Raises
    ------
    ValueError
        If the kernel_size is invalid
    """
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
        )
    return _create_no_args_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_type,
    )
