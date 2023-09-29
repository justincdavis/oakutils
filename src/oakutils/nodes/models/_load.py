from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from oakutils.nodes.neural_network import create_neural_network

from ._parsing import get_candidates, parse_kernel_size

if TYPE_CHECKING:
    import depthai as dai

_log = logging.getLogger(__name__)


def create_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output | Iterable[dai.Node.Output],
    model_name: str,
    attributes: list[str],
    shaves: int,
    input_names: Iterable[str] | None = None,
    reuse_messages: Iterable[bool | None] | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to get the model blob based on the attributes and creates a neural network node.

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
    shaves : int
        The number of shaves to use
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
        shaves=shaves,
    )

    try:
        _, attributes, path = potential_blobs[0]
    except IndexError as err:
        base_str = "Error acquiring model blob."
        base_str += (
            "Please check that all models are present in your installation through: "
        )
        base_str += "`dir(oakutils.blobs.models)`"
        raise ValueError(f"{base_str}\n Possible blobs: {potential_blobs}") from err

    return create_neural_network(
        pipeline=pipeline,
        input_link=input_link,
        blob_path=path,
        input_names=input_names,
        reuse_messages=reuse_messages,
    )


def create_no_args_multi_link_model(
    pipeline: dai.Pipeline,
    input_links: list[dai.Node.Output],
    model_name: str,
    shaves: int,
    input_names: list[str],
    reuse_messages: list[bool | None],
) -> dai.node.NeuralNetwork:
    """
    Use to create a model with multiple input links.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the model to
    input_links : List[dai.node.XLinkOut]
        The input links to connect to the model node.
        Example: [cam_rgb.preview, cam_left.preview]
    model_name : str
        The name of the model to use
    shaves : int
        The number of shaves to use
    input_names : List[str]
        The names of the input layers
    reuse_messages : List[Optional[bool]]
        Whether or not the data on the stream data will be reused

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    """
    _log.warning(
        f"Multi-link models do not have passthrough! Creating model {model_name} with multiple input links..."
    )
    return create_model(
        pipeline=pipeline,
        input_link=input_links,
        model_name=model_name,
        attributes=[],
        shaves=shaves,
        input_names=input_names,
        reuse_messages=reuse_messages,
    )


def create_no_args_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    shaves: int,
) -> dai.node.NeuralNetwork:
    """
    Use to create a model with no arguments.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the model to
    input_link : dai.node.XLinkOut
        The input link to connect to the model node.
        Example: cam_rgb.preview
    model_name : str
        The name of the model to use
    shaves : int
        The number of shaves to use

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    """
    return create_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_name,
        attributes=[],
        shaves=shaves,
    )


def create_single_kernel_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    kernel_size: int,
    shaves: int,
) -> dai.node.NeuralNetwork:
    """
    Use to create a model with a single kernel size.

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
    shaves : int
        The number of shaves to use

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
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
        shaves=shaves,
    )


def create_double_kernel_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    kernel_size1: int,
    kernel_size2: int,
    shaves: int,
) -> dai.node.NeuralNetwork:
    """
    Use to create a model with a two kernel sizes.

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
    shaves : int
        The number of shaves to use

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
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
        shaves=shaves,
    )
