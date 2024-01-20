# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

from oakutils.nodes.neural_network import create_neural_network

from ._parsing import get_candidates, parse_kernel_size

if TYPE_CHECKING:
    import depthai as dai

_log = logging.getLogger(__name__)


def create_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output | list[dai.Node.Output],
    model_name: str,
    attributes: list[str],
    shaves: int,
    input_names: str | list[str] | None = None,
    input_sizes: int | list[int] | None = None,
    *,
    input_blocking: bool | list[bool] | None = None,
    reuse_messages: bool | list[bool | None] | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to get the model blob based on the attributes and creates a neural network node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the gaussian to
    input_link : Union[dai.node.XLinkOut, List[dai.node.XLinkOut]]
        The input link to connect to the gaussian node.
        Example: cam_rgb.preview.link
        Explicitly pass in the link as a non-called function.
    model_name : str
        The name of the model to use
    attributes : List[str]
        The attributes of the model to use
    shaves : int
        The number of shaves to use
    input_names : Optional[List[str]], optional
        The names of the input layers, by default None
        If None, will use the default input names for the model
    input_sizes : Optional[List[int]], optional
        The sizes of the queue for each input stream, by default None
        If None, will use the default input sizes for the model
    input_blocking : Optional[List[bool]], optional
        Whether or not the input stream will be blocking, by default None
        If None, will use the default input blocking for the model
    reuse_messages : Optional[List[Optional[bool]]], optional
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
        err_msg = f"{base_str}\n Possible blobs: {potential_blobs}"
        raise ValueError(err_msg) from err

    new_path: pathlib.Path = pathlib.Path(path)

    return create_neural_network(
        pipeline=pipeline,
        input_link=input_link,
        blob_path=new_path,
        input_names=input_names,
        input_sizes=input_sizes,
        input_blocking=input_blocking,
        reuse_messages=reuse_messages,
    )


def create_no_args_multi_link_model(
    pipeline: dai.Pipeline,
    input_links: list[dai.Node.Output],
    model_name: str,
    shaves: int,
    input_names: list[str] | None = None,
    input_sizes: list[int] | None = None,
    *,
    input_blocking: list[bool] | None = None,
    reuse_messages: list[bool | None] | None = None,
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
    input_names : List[str], optional
        The names of the input layers
    input_sizes : List[int], optional
        The sizes of the queue for each input stream
    input_blocking : List[bool], optional
        Whether or not the input stream will be blocking
    reuse_messages : List[Optional[bool]], optional
        Whether or not the data on the stream data will be reused

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    """
    _log.warning(
        f"Multi-link models do not have passthrough! Creating model {model_name} with multiple input links...",
    )
    return create_model(
        pipeline=pipeline,
        input_link=input_links,
        model_name=model_name,
        attributes=[],
        shaves=shaves,
        input_names=input_names,
        input_sizes=input_sizes,
        input_blocking=input_blocking,
        reuse_messages=reuse_messages,
    )


def create_no_args_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    shaves: int,
    input_names: str | None = None,
    input_sizes: int | None = None,
    *,
    input_blocking: bool | None = None,
    reuse_messages: bool | None = None,
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
    input_names : str, optional
        The names of the input layers
    input_sizes : int, optional
        The sizes of the queue for each input stream
    input_blocking : bool, optional
        Whether or not the input stream will be blocking
    reuse_messages : bool, optional
        Whether or not the data on the stream data will be reused

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
        input_names=input_names,
        input_sizes=input_sizes,
        input_blocking=input_blocking,
        reuse_messages=reuse_messages,
    )


def create_single_kernel_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    kernel_size: int,
    shaves: int,
    input_names: str | None = None,
    input_sizes: int | None = None,
    *,
    input_blocking: bool | None = None,
    reuse_messages: bool | None = None,
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
    input_names : str, optional
        The names of the input layers
    input_sizes : int, optional
        The sizes of the queue for each input stream
    input_blocking : bool, optional
        Whether or not the input stream will be blocking
    reuse_messages : bool, optional
        Whether or not the data on the stream data will be reused


    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    """
    _ = parse_kernel_size(
        kernel_size,
    )  # raises ValueError if invalid, so no need to check
    attributes = [str(kernel_size)]
    return create_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_name,
        attributes=attributes,
        shaves=shaves,
        input_names=input_names,
        input_sizes=input_sizes,
        input_blocking=input_blocking,
        reuse_messages=reuse_messages,
    )


def create_double_kernel_model(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    model_name: str,
    kernel_size1: int,
    kernel_size2: int,
    shaves: int,
    input_names: str | None = None,
    input_sizes: int | None = None,
    *,
    input_blocking: bool | None = None,
    reuse_messages: bool | None = None,
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
    input_names : str, optional
        The names of the input layers
    input_sizes : int, optional
        The sizes of the queue for each input stream
    input_blocking : bool, optional
        Whether or not the input stream will be blocking
    reuse_messages : bool, optional
        Whether or not the data on the stream data will be reused

    Returns
    -------
    dai.node.NeuralNetwork
        The model node
    """
    _ = parse_kernel_size(
        kernel_size1,
    )  # raises ValueError if invalid, so no need to check
    _ = parse_kernel_size(
        kernel_size2,
    )  # raises ValueError if invalid, so no need to check
    attributes = [str(kernel_size1), str(kernel_size2)]
    return create_model(
        pipeline=pipeline,
        input_link=input_link,
        model_name=model_name,
        attributes=attributes,
        shaves=shaves,
        input_names=input_names,
        input_sizes=input_sizes,
        input_blocking=input_blocking,
        reuse_messages=reuse_messages,
    )
