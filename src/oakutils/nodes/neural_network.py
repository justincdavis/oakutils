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

import logging
from pathlib import Path
from typing import Callable, Iterable

import cv2
import depthai as dai
import numpy as np

_log = logging.getLogger(__name__)


def create_neural_network(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output | Iterable[dai.Node.Output],
    blob_path: str,
    input_names: str | Iterable[str] | None = None,
    reuse_messages: bool | Iterable[bool | None] | None = None,
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to create a neural network node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the neural network to
    input_link : Union[dai.Node.Output, Iterable[dai.Node.Output]]
        The input link to connect to the image manip node or,
        if there are multiple input links, an iterable of input links.
        Example: cam_rgb.preview or (cam_rgb.preview, stereo.depth)
    blob_path : str
        The path to the blob file to use for the neural network.
        Will be converted to a pathlib.Path.
    input_names : Optional[Union[str, Iterable[str]]], optional
        The names of the input links, by default None
        Must be the same length as input_link if Iterable
    reuse_messages : Optional[Union[bool, Iterable[bool]]], optional
        Whether to reuse messages, by default None
        Must be the same length as input_link if Iterable
        Values may be None if the input link does not need a value set
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
    """
    if hasattr(input_link, "__iter__"):
        if input_names is None:
            raise ValueError(
                "input_names must be provided if input_link is an iterable"
            )
        if not hasattr(input_names, "__iter__"):
            raise ValueError(
                "input_names must be an iterable if input_link is an iterable"
            )
        if len(input_link) != len(input_names):
            raise ValueError(
                "input_link and input_names must be the same length if both are iterables"
            )
        if reuse_messages is not None:
            if not hasattr(reuse_messages, "__iter__"):
                raise ValueError(
                    "reuse_messages must be an iterable if input_link is an iterable"
                )
            if len(input_link) != len(reuse_messages):
                raise ValueError(
                    "input_link and reuse_messages must be the same length if both are iterables"
                )

    bpath: Path = Path(blob_path)

    # create the node and handle the always present parameters
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(bpath)
    nn.setNumInferenceThreads(num_inference_threads)

    # handle the optional parameters
    if num_nce_per_inference_thread is not None:
        nn.setNumNCEPerInferenceThread(num_nce_per_inference_thread)
    if num_pool_frames is not None:
        nn.setNumPoolFrames(num_pool_frames)

    # connect the input link to the neural network node
    if not hasattr(input_link, "__iter__"):
        # handle a single input to the network
        input_link.link(nn.input)
    else:
        input_data = zip(input_link, input_names, reuse_messages)
        for link, name, reuse_message in input_data:
            _log.debug(
                f"Linking {link.name} to {name}, assigning reuse: {reuse_message}"
            )
            link.link(nn.inputs[name])
            if reuse_message is not None:
                nn.inputs[name].setReusePreviousMessage(reuse_message)

    return nn


def _normalize(frame: np.ndarray, factor: float | Callable | None = None) -> np.ndarray:
    """
    Use to normalize a frame.

    Parameters
    ----------
    frame : np.ndarray
        The frame to normalize.
    factor : Optional[float, Callable], optional
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
    return factor(frame)


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
    return cv2.resize(
        frame,
        (0, 0),
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_LINEAR,
    )


def get_nn_frame(
    data: np.ndarray | dai.NNData,
    channels: int,
    frame_size: tuple[int, int] = (640, 480),
    resize_factor: float | None = None,
    normalization: float | Callable | None = None,
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
    normalization : Optional[float, Callable], optional
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
    frame = (
        data.view(np.float16)
        .reshape((channels, frame_size[1], frame_size[0]))
        .transpose(1, 2, 0)
    )
    frame += 0.5
    frame *= 255.0

    if swap_rb:
        frame = frame[:, :, ::-1]

    if resize_factor is not None and normalization is not None:
        if resize_factor <= 1.0:
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
        1, 3, frame_size[1], frame_size[0]
    )
    pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / scale

    if remove_zeros:
        # optimization over an np.all since it performs less checks
        # and realisticlly it does not matter if there is a few points
        # difference over hundreds of interations
        pcl_data = pcl_data[pcl_data[:, 2] != 0.0]

    return pcl_data
