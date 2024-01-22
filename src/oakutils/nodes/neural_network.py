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
"""
Module for creating neural network nodes.

Functions
---------
create_neural_network
    Creates a neural network node.
get_nn_frame
    Takes the raw data output from a neural network execution and converts it to a frame
    usable by cv2.
get_nn_bgr_frame
    Takes the raw data output from a neural network execution and converts it to a BGR frame
    usable by cv2.
get_nn_gray_frame
    Takes the raw data output from a neural network execution and converts it to a grayscale frame
    usable by cv2.
get_nn_point_cloud_buffer
    Takes the raw data output from a neural network execution and converts it to a point cloud buffer.
"""
from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Callable

import cv2  # type: ignore[import]
import depthai as dai
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_log = logging.getLogger(__name__)


def create_neural_network(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output | list[dai.Node.Output],
    blob_path: Path,
    input_names: str | list[str] | None = None,
    input_sizes: int | list[int] | None = None,
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
    *,
    reuse_messages: bool | list[bool | None] | None = None,
    input_blocking: bool | list[bool] | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to create a neural network node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the neural network to
    input_link : Union[dai.Node.Output, list[dai.Node.Output]]
        The input link to connect to the image manip node or,
        if there are multiple input links, an iterable of input links.
        Example: cam_rgb.preview or (cam_rgb.preview, stereo.depth)
    blob_path : str
        The path to the blob file to use for the neural network.
        Will be converted to a pathlib.Path.
    input_names : Optional[Union[str, list[str]]], optional
        The names of the input links, by default None
        Must be the same length as input_link if a list
    reuse_messages : Optional[Union[bool, list[bool]]], optional
        Whether to reuse messages, by default None
        Must be the same length as input_link if a list
        Values may be None if the input link does not need a value set
    input_sizes : Optional[Union[int, list[int]]], optional
        The size of the input queue, by default None
        Must be the same length as input_link if a list
    input_blocking : Optional[Union[bool, list[bool]]], optional
        Whether the input queue is blocking, by default None
        Must be the same length as input_link if a list
    num_inference_threads : int, optional
        The number of inference threads, by default 2
    num_nce_per_inference_thread : Optional[int], optional
        The number of NCEs per inference thread, by default None
         NCE: Neural Compute Engine
    num_pool_frames : Optional[int], optional
        The number of pool frames, by default None

    Returns
    -------
    dai.node.NeuralNetwork
        The neural network node

    Raises
    ------
    ValueError
        If input_link is an iterable and input_names is None
        If input_link and input_names are iterables and are not the same length
        If input_link and reuse_messages are iterables and are not the same length
    TypeError
        If input_link is an iterable and input_names is not an iterable
        If input_link is an iterable and reuse_messages is not an iterable
    """
    if isinstance(input_link, list):
        if input_names is None:
            err_msg = "input_names must be provided if input_link is an iterable"
            raise ValueError(
                err_msg,
            )
        if not isinstance(input_names, list):
            err_msg = "input_names must be an iterable if input_link is an iterable"
            raise TypeError(
                err_msg,
            )
        if len(input_link) != len(input_names):
            err_msg = "input_link and input_names must be the same length if both are iterables"
            raise ValueError(
                err_msg,
            )
        if reuse_messages is not None:
            if not isinstance(reuse_messages, list):
                err_msg = (
                    "reuse_messages must be an iterable if input_link is an iterable"
                )
                raise TypeError(
                    err_msg,
                )
            if len(input_link) != len(reuse_messages):
                err_msg = "input_link and reuse_messages must be the same length if both are iterables"
                raise ValueError(
                    err_msg,
                )

    # create the node and handle the always present parameters
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(num_inference_threads)

    # handle the optional parameters
    if num_nce_per_inference_thread is not None:
        nn.setNumNCEPerInferenceThread(num_nce_per_inference_thread)
    if num_pool_frames is not None:
        nn.setNumPoolFrames(num_pool_frames)

    # connect the input link to the neural network node
    if not isinstance(input_link, list):
        # handle a single input to the network
        input_link.link(nn.input)
    else:
        if input_names is None or reuse_messages is None:
            err_msg = "input_names and reuse_messages must be provided if input_link is an iterable"
            raise ValueError(
                err_msg,
            )
        if isinstance(input_names, str) or isinstance(reuse_messages, bool):
            err_msg = "input_names and reuse_messages must be iterables if input_link is an iterable"
            raise TypeError(
                err_msg,
            )
        if input_blocking is not None and isinstance(input_blocking, bool):
            input_blocking = [input_blocking] * len(input_link)
        if input_sizes is not None and isinstance(input_sizes, int):
            input_sizes = [input_sizes] * len(input_link)

        input_data = zip(input_link, input_names, reuse_messages)
        for idx, (link, name, reuse_message) in enumerate(input_data):
            _log.debug(
                f"Linking {link.name} to {name}, assigning reuse: {reuse_message}",
            )
            link.link(nn.inputs[name])
            if reuse_message is not None:
                nn.inputs[name].setReusePreviousMessage(reuse_message)
            if input_blocking is not None:
                with contextlib.suppress(IndexError):
                    nn.inputs[name].setBlocking(input_blocking[idx])
            if input_sizes is not None:
                with contextlib.suppress(IndexError):
                    nn.inputs[name].setQueueSize(input_sizes[idx])

    return nn


def _normalize(
    frame: np.ndarray,
    factor: float | Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Use to normalize a frame.

    Parameters
    ----------
    frame : np.ndarray
        The frame to normalize.
    factor : Optional[float, Callable[[np.ndarray], np.ndarray]], optional
        The normalization factor.

    Returns
    -------
    np.ndarray
        The normalized frame.
    """
    if factor is None:
        return frame
    if isinstance(factor, float):
        return frame * factor
    return factor(frame)  # pyright: ignore [reportCallIssue]


def _resize(frame: np.ndarray, factor: float | None = None) -> np.ndarray:
    """
    Use to resize a frame.

    Parameters
    ----------
    frame : np.ndarray
        The frame to resize.
    factor : Optional[float], optional
        The resize factor.

    Returns
    -------
    np.ndarray
        The resized frame.
    """
    if factor is None:
        return frame
    resized_frame: np.ndarray = cv2.resize(
        frame,
        (0, 0),
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_LINEAR,
    )
    return resized_frame


def get_nn_frame(
    data: np.ndarray | dai.NNData,
    channels: int,
    frame_size: tuple[int, int] = (640, 480),
    resize_factor: float | None = None,
    normalization: float | Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    swap_rb: bool | None = None,
) -> np.ndarray:
    """
    Use to convert the raw data output from a neural network execution and return a frame.

    Parameters
    ----------
    data : Union[np.ndarray, dai.NNData]
        Raw data output from a neural network execution.
    channels : int
        The number of channels in the frame.
    frame_size : tuple[int, int], optional
        The size of the frame, by default (640, 480)
        This is the size of the frame before any resizing is applied.
        If frame_size is incorrect, an error will occur.
    resize_factor : Optional[float], optional
        The resize factor to apply to the frame, by default None
    normalization : Optional[float, Callable[[np.ndarray], np.ndarray]], optional
        The normalization to apply to the frame, by default None
        If a float then the frame is multiplied by the float.
        If a callable then the frame is passed to the callable and
        set to the return value.
        If resize_factor is less than 1.0, then normalization is applied
        after resizing.
    swap_rb : Optional[bool], optional
        Whether to swap the red and blue channels, by default None
        If None, then False is used

    Returns
    -------
    np.ndarray
        Frame usable by cv2.
    """
    if swap_rb is None:
        swap_rb = False

    if isinstance(data, dai.NNData):
        data = data.getData()
    frame: np.ndarray = (
        data.view(np.float16)
        .reshape((channels, frame_size[1], frame_size[0]))
        .transpose(1, 2, 0)
    )
    frame += 0.5
    frame *= 255.0

    if swap_rb:
        frame = frame[:, :, ::-1]

    if resize_factor is not None and normalization is not None:
        resize_to_be_smaller = 1.0
        if resize_factor <= resize_to_be_smaller:
            frame = _normalize(_resize(frame, resize_factor), normalization)
        else:
            frame = _resize(_normalize(frame, normalization), resize_factor)
    else:
        if resize_factor is not None:
            frame = _resize(frame, resize_factor)
        if normalization is not None:
            frame = _normalize(frame, normalization)

    return np.ascontiguousarray(frame, dtype=np.uint8)


def get_nn_bgr_frame(
    data: np.ndarray | dai.NNData,
    frame_size: tuple[int, int] = (640, 480),
    resize_factor: float | None = None,
    normalization: float | Callable | None = None,
) -> np.ndarray:
    """
    Use to convert the raw data output from a neural network execution and return a BGR frame.

    Parameters
    ----------
    data : Union[np.ndarray, dai.NNData]
        Raw data output from a neural network execution.
    frame_size : tuple[int, int], optional
        The size of the frame, by default (640, 480)
    resize_factor : Optional[float], optional
        The resize factor to apply to the frame, by default None
    normalization : Optional[float, Callable], optional
        The normalization to apply to the frame, by default None
        If a float then the frame is multiplied by the float.
        If a callable then the frame is passed to the callable and
        set to the return value.
        If resize_factor is less than 1.0, then normalization is applied
        after resizing.

    Returns
    -------
    np.ndarray
        BGR frame usable by cv2.
    """
    return get_nn_frame(
        data=data,
        channels=3,
        frame_size=frame_size,
        resize_factor=resize_factor,
        normalization=normalization,
    )


def get_nn_gray_frame(
    data: np.ndarray | dai.NNData,
    frame_size: tuple[int, int] = (640, 480),
    resize_factor: float | None = None,
    normalization: float | Callable | None = None,
) -> np.ndarray:
    """
    Use to convert the raw data output from a neural network execution and return a grayscale frame.

    Parameters
    ----------
    data : Union[np.ndarray, dai.NNData]
        Raw data output from a neural network execution.
    frame_size : tuple[int, int], optional
        The size of the frame, by default (640, 480)
    resize_factor : Optional[float], optional
        The resize factor to apply to the frame, by default None
    normalization : Optional[float, Callable], optional
        The normalization to apply to the frame, by default None
        If a float then the frame is multiplied by the float.
        If a callable then the frame is passed to the callable and
        set to the return value.
        If resize_factor is less than 1.0, then normalization is applied
        after resizing.

    Returns
    -------
    np.ndarray
        Grayscale frame usable by cv2.
    """
    return get_nn_frame(
        data=data,
        channels=1,
        frame_size=frame_size,
        resize_factor=resize_factor,
        normalization=normalization,
    )


def get_nn_point_cloud_buffer(
    data: dai.NNData,
    frame_size: tuple[int, int] = (640, 400),
    scale: float = 1000.0,
    *,
    remove_zeros: bool | None = None,
) -> np.ndarray:
    """
    Use to convert the raw data output from a neural network execution and converts it to a point cloud.

    Parameters
    ----------
    data : dai.NNData
        Raw data output from a neural network execution.
    frame_size : tuple[int, int], optional
        The size of the buffer, by default (640, 400)
        Usually this will be the size of the depth frame.
        Which inherits its shape from the MonoCamera resolutions.
    scale: float, optional
        The scale to apply to the point cloud, by default 1000.0
        This will convert from mm to m.
    remove_zeros: bool, optional
        Whether to remove zero points, by default None
        If None, then True is used
        Recommended to set to True to remove zero points
        Can speedup reading and filtering of the point cloud
        by up to 10x

    Returns
    -------
    np.ndarray
        Point cloud buffer
    """
    if remove_zeros is None:
        remove_zeros = True

    pcl_data = np.array(data.getFirstLayerFp16()).reshape(
        1,
        3,
        frame_size[1],
        frame_size[0],
    )
    pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / scale

    if remove_zeros:
        # optimization over an np.all since it performs less checks
        # and realisticlly it does not matter if there is a few points
        # difference over hundreds of interations
        zero_val = 0.0
        pcl_data = pcl_data[pcl_data[:, 2] != zero_val]

    return pcl_data
