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
get_nn_point_cloud
    Takes the raw data output from a neural network execution and converts it to a point cloud.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sized

import cv2
import depthai as dai
import numpy as np

_log = logging.getLogger(__name__)


def create_neural_network(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output | Sized[dai.Node.Output],
    blob_path: str,
    input_names: str | Sized[str] | None = None,
    input_size: int = 5,
    input_blocking: bool | None = None,
    input_reuse_message: bool | None = None,
    input_wait_for_message: bool | None = None,
    inputs_sizes: Sized[int] | None = None,
    inputs_blocking: Sized[bool] | None = None,
    inputs_reuse_messages: Sized[bool | None] | None = None,
    inputs_wait_for_messages: Sized[bool] | None = None,
    num_inference_threads: int = 0,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
) -> dai.node.NeuralNetwork:
    """
    Use to create a neural network node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the neural network to
    input_link : Union[dai.Node.Output, Sized[dai.Node.Output]]
        The input link to connect to the image manip node or,
        if there are multiple input links, an iterable of input links.
        Example: cam_rgb.preview or (cam_rgb.preview, stereo.depth)
    blob_path : str
        The path to the blob file to use for the neural network.
        Will be converted to a pathlib.Path.
    input_names : Optional[Union[str, Sized[str]]], optional
        The names of the input links, by default None
        Must be the same length as input_link if Sized
    input_size : int
        The size of the queue for the input link, by default will
        use a queue size of 5.
    input_blocking : bool, optional
        Whether the input link is blocking, by default None
        If None will use True
    input_reuse_message : bool, optional
        Whether to reuse messages, by default None
        If None will use False
    input_wait_for_messages : bool, optional
        Whether to wait for messages, by default None
        If None will use True
    inputs_sizes : Optional[Union[int, Sized[int]]], optional
        The size of the queue for the input links, by default will
        use a queue size of 1.
        Must be the same length as input_link if Sized
    inputs_blocking : Optional[Union[bool, Sized[bool]]], optional
        Whether the input link is blocking, by default will set to False.
        Must be the same length as input_link if Sized
    inputs_reuse_messages : Optional[Union[bool, Sized[bool]]], optional
        Whether to reuse messages, by default None
        Must be the same length as input_link if Sized
        Values may be None if the input link does not need a value set
    inputs_wait_for_messages : Optional[Union[bool, Sized[bool]]], optional
        Whether to wait for messages, by default None
        Must be the same length as input_link if Sized
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

    Raises
    ------
    ValueError
        If input_link is an iterable and input_names is None
        If input_link and input_names are iterables and are not the same length
        If input_link and reuse_messages are iterables and are not the same length
        If input_link and input_sizes are iterables and are not the same length
        If input_link and input_blocking are iterables and are not the same length
    """
    # generate the pathlib path
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

    # print(f"inputConfig Queue Size: {nn.input.getQueueSize()}")
    # print(f"inputConfig Reuse Previous Message: {nn.input.getReusePreviousMessage()}")
    # print(f"inputConfig Blocking: {nn.input.getBlocking()}")
    # print(f"inputConfig Wait for Message: {nn.input.getWaitForMessage()}")
    # try:
    #     for nn_input in input_names:
    #         print(f"inputConfig Name: {nn_input}")
    #         print(f"inputConfig Queue Size: {nn.inputs[nn_input].getQueueSize()}")
    #         print(f"inputConfig Reuse Previous Message: {nn.inputs[nn_input].getReusePreviousMessage()}")
    #         print(f"inputConfig Blocking: {nn.inputs[nn_input].getBlocking()}")
    #         print(f"inputConfig Wait for Message: {nn.inputs[nn_input].getWaitForMessage()}")
    # except TypeError:
    #     pass

    # connect the input link to the neural network node
    if hasattr(input_link, "__len__"):
        # handle the inputs parameters
        param_name_list = [
            "inputs_reuse_messages",
            "inputs_sizes",
            "inputs_blocking",
            "inputs_wait_for_messages",
        ]
        param_list = [
            inputs_reuse_messages,
            inputs_sizes,
            inputs_blocking,
            inputs_wait_for_messages,
        ]
        param_default_values = [
            False,
            1,
            False,
            True,
        ]   
        # handle the arguments & input_names
        if input_names is None:
            raise ValueError("input_names must be provided if input_link has size")
        if not hasattr(input_names, "__len__"):
            raise ValueError("input_names must be an iterable if input_link has size")
        if len(input_link) != len(input_names):
            raise ValueError("input_link and input_names must have the same length")
        # handle the other parameters
        for param_name, param, param_default in zip(
            param_name_list, 
            param_list, 
            param_default_values,
        ):
            if param is None:
                param = [param_default for _ in range(len(input_link))]    
            if not hasattr(param, "__len__"):
                raise ValueError(f"{param_name} must have __len__ if input_link has size")
            if len(param) != len(input_link):
                raise ValueError(f"{param_name} and input_link must have the same length")
        
        # assign the links
        input_data = zip(
            input_link, input_names, inputs_sizes, inputs_blocking, inputs_reuse_messages, inputs_wait_for_messages
        )
        for link, name, size, blocking, reuse_message, wait_message in input_data:
            link.link(nn.inputs[name])
            if size is not None:
                nn.inputs[name].setQueueSize(size)
            if blocking is not None:
                nn.inputs[name].setBlocking(blocking)
            if reuse_message is not None:
                nn.inputs[name].setReusePreviousMessage(reuse_message)
            if wait_message is not None:
                nn.inputs[name].setWaitForMessage(wait_message)
    else:
        # handle the input parameters
        if input_blocking is None:
            input_blocking = True
        if input_reuse_message is None:
            input_reuse_message = False
        if input_wait_for_message is None:
            input_wait_for_message = True
        # assign basic link
        input_link.link(nn.input)
        nn.input.setQueueSize(input_size)
        if input_blocking is not None:
            nn.input.setBlocking(input_blocking)
        if input_reuse_message is not None:
            nn.input.setReusePreviousMessage(input_reuse_message)
        if input_wait_for_message is not None:
            nn.input.setWaitForMessage(input_wait_for_message)

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
    normalization: float | Callable | None = 255.0,
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
        The normalization to apply to the frame, by default 255.0
        If a float then the frame is multiplied by the float.
        If a callable then the frame is passed to the callable and
        set to the return value.
        If resize_factor is less than 1.0, then normalization is applied
        after resizing.
    swap_rb : Optional[bool], optional
        Whether to swap the red and blue channels, by default None

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

    if swap_rb:
        frame = frame[:, :, ::-1]

    if resize_factor is not None and resize_factor <= 1.0:
        frame = _normalize(_resize(frame, resize_factor), normalization)
    else:
        frame = _resize(_normalize(frame, normalization), resize_factor)

    return frame.astype(np.uint8)


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


def get_nn_point_cloud(
    data: dai.NNData,
    frame_size: tuple[int, int] = (640, 400),
    scale: float = 1000.0,
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

    Returns
    -------
    np.ndarray
        Point cloud
    """
    pcl_data = np.array(data.getFirstLayerFp16()).reshape(
        1, 3, frame_size[1], frame_size[0]
    )
    return pcl_data.reshape(3, -1).T.astype(np.float64) / scale
