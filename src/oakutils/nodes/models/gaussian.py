from typing import Tuple

import depthai as dai

from ._parsing import parse_kernel_size, get_candidates
from ..neural_network import create_neural_network


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
    model_type = "gaussian"
    if grayscale_out:
        model_type = "gaussiangray"
    attributes = [str(kernel_size)]
    _ = parse_kernel_size(
        kernel_size
    )  # raises ValueError if invalid, so no need to check
    potential_blobs = get_candidates(
        model_type=model_type,
        attributes=attributes,
    )

    try:
        model = potential_blobs[0]
    except IndexError:
        raise ValueError(
            "Error acquiring model blob. Please check that all models are present in your installation through `dir(oakutils.blobs.models)`"
        )

    streamname = f"{model_type}_".join([a for a in attributes])
    nn, nn_out = create_neural_network(
        pipeline=pipeline,
        input_link=input_link,
        blob_path=model,
        stream_name=streamname,
    )

    return nn, nn_out, streamname
