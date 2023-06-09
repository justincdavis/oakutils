from typing import Tuple

import depthai as dai

from ._load import create_single_kernel_model as _create_single_kernel_model


def create_gaussian(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    kernel_size: int = 3,
    grayscale_out: bool = False,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a gaussian model with a specified kernel size

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
    grayscale_out : bool, optional
        Whether or not to use graycale output, by default False

    Returns
    -------
    dai.node.NeuralNetwork
        The gaussian node
    dai.node.XLinkOut
        The output link of the gaussian node
    str
        The name of the stream, determined by the model_type and attributes

    Raises
    ------
    ValueError
        If the kernel_size is invalid
    """
    model_type = "laplacian"
    if grayscale_out:
        model_type += "gray"
    return _create_single_kernel_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_type,
        kernel_size=kernel_size,
    )
