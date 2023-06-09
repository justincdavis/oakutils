from typing import Tuple, List, Optional, Iterable, Union

import depthai as dai

from ..neural_network import create_neural_network
from ._parsing import get_candidates, parse_kernel_size


def create_model(
    pipeline: dai.Pipeline,
    input_link: Union[dai.Node.Output, Iterable[dai.Node.Output]],
    model_name: str,
    attributes: List[str],
    input_names: Optional[Iterable[str]] = None,
    reuse_messages: Optional[Iterable[Optional[bool]]] = None,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Gets the model blob based on the attributes and creates a neural network node

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the gaussian to
    input_link : dai.node.XLinkOut
        The input link to connect to the gaussian node.
        Example: cam_rgb.preview.link
        Explicitly pass in the link as a non-called function.
    model_name : str
        The name of the model to use
    attributes : List[str]
        The attributes of the model to use
    input_names : Optional[Iterable[str]], optional
        The names of the input layers, by default None
        If None, will use the default input names for the model
    reuse_messages : Optional[Iterable[Optional[bool]]], optional
        Whether or not the data on the stream will be reused, by default None

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
    potential_blobs = get_candidates(
        model_type=model_name,
        attributes=attributes,
    )

    try:
        name, attributes, path = potential_blobs[0]
    except IndexError:
        raise ValueError(
            "Error acquiring model blob. Please check that all models are present in your installation through `dir(oakutils.blobs.models)`"
        )

    streamname = f"{name}_".join([a for a in attributes])
    while streamname.endswith("_"):
        streamname = streamname[:-1]
    nn, nn_out = create_neural_network(
        pipeline=pipeline,
        input_link=input_link,
        blob_path=path,
        stream_name=streamname,
        input_names=input_names,
        reuse_messages=reuse_messages,
    )

    return nn, nn_out, streamname


def create_no_args_multi_link_model(
    pipeline: dai.Pipeline,
    input_links: List[dai.Node.Output],
    model_name: str,
    input_names: List[str],
    reuse_messages: List[Optional[bool]],
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a model with multiple input links

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the model to
    input_links : List[dai.node.XLinkOut]
        The input links to connect to the model node.
        Example: [cam_rgb.preview, cam_left.preview]
    model_name : str
        The name of the model to use
    input_names : List[str]
        The names of the input layers
    reuse_messages : List[Optional[bool]]
        Whether or not the data on the stream data will be reused

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    dai.node.XLinkOut
        The output link of the model node
    str
        The name of the stream, determined by the model_type and attributes
    """
    return create_model(
        pipeline=pipeline,
        input_link=input_links,
        model_name=model_name,
        attributes=[],
        input_names=input_names,
        reuse_messages=reuse_messages,
    )

def create_no_args_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a model with no arguments

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the model to
    input_link : dai.node.XLinkOut
        The input link to connect to the model node.
        Example: cam_rgb.preview
    model_name : str
        The name of the model to use

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    dai.node.XLinkOut
        The output link of the model node
    str
        The name of the stream, determined by the model_type and attributes
    """
    return create_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_name,
        attributes=[],
    )


def create_single_kernel_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    kernel_size: int,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a model with a single kernel size

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the model to
    input_link : dai.node.XLinkOut
        The input link to connect to the model node.
        Example: cam_rgb.preview
    model_name : str
        The name of the model to use
    kernel_size : int
        The size of the kernel to use

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    dai.node.XLinkOut
        The output link of the model node
    str
        The name of the stream, determined by the model_type and attributes
    """
    _ = parse_kernel_size(
        kernel_size
    )  # raises ValueError if invalid, so no need to check
    attributes = [str(kernel_size)]
    return create_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_name,
        attributes=attributes,
    )


def create_double_kernel_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    kernel_size1: int,
    kernel_size2: int,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut, str]:
    """
    Creates a model with a two kernel sizes

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the model to
    input_link : dai.node.XLinkOut
        The input link to connect to the model node.
        Example: cam_rgb.preview
    model_name : str
        The name of the model to use
    kernel_size1 : int
        The size of the kernel to use
    kernel_size2 : int
        The size of the kernel to use

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    dai.node.XLinkOut
        The output link of the model node
    str
        The name of the stream, determined by the model_type and attributes
    """
    _ = parse_kernel_size(
        kernel_size1
    )  # raises ValueError if invalid, so no need to check
    _ = parse_kernel_size(
        kernel_size2
    )  # raises ValueError if invalid, so no need to check
    attributes = [str(kernel_size1), str(kernel_size2)]
    return create_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_name,
        attributes=attributes,
    )
