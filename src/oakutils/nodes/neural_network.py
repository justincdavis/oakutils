from __future__ import annotations

from pathlib import Path
from typing import Iterable

import depthai as dai
import numpy as np


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
    """Creates a neural network node.

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
        if input_names is not None:
            input_link.link(nn.inputs[input_names])
        else:
            input_link.link(nn.input)
    else:
        input_data = zip(input_link, input_names, reuse_messages)
        for link, name, reuse_message in input_data:
            link.link(nn.inputs[name])
            if reuse_message is not None:
                nn.inputs[name].setReusePreviousMessage(reuse_message)

    return nn


def get_nn_bgr_frame(
    data: np.ndarray | dai.NNData, frame_size: tuple[int, int] = (640, 480)
) -> np.ndarray:
    """Takes the raw data output from a neural network execution and converts it to a BGR frame
    usable by cv2.

    Parameters
    ----------
    data : Union[np.ndarray, dai.NNData]
        Raw data output from a neural network execution.

    Returns
    -------
    np.ndarray
        BGR frame usable by cv2.
    """
    if isinstance(data, dai.NNData):
        data = data.getData()
    frame = (
        data.view(np.float16)
        .reshape((3, frame_size[1], frame_size[0]))
        .transpose(1, 2, 0)
    )
    return (frame * 255 + 127.5).astype(np.uint8)


def get_nn_gray_frame(
    data: np.ndarray | dai.NNData, frame_size: tuple[int, int] = (640, 480)
) -> np.ndarray:
    """Takes the raw data output from a neural network execution and converts it to a grayscale frame
    usable by cv2.

    Parameters
    ----------
    data : Union[np.ndarray, dai.NNData]
        Raw data output from a neural network execution.

    Returns
    -------
    np.ndarray
        Grayscale frame usable by cv2.
    """
    if isinstance(data, dai.NNData):
        data = data.getData()
    frame = (
        data.view(np.float16)
        .reshape((1, frame_size[1], frame_size[0]))
        .transpose(1, 2, 0)
    )
    return (frame * 255 + 127.5).astype(np.uint8)


def get_nn_point_cloud(
    data: dai.NNData, frame_size: tuple[int, int] = (640, 400)
) -> np.ndarray:
    """Takes the raw data output from a neural network execution and converts it to a point cloud

    Parameters
    ----------
    data : dai.NNData
        Raw data output from a neural network execution.
    frame_size : tuple[int, int], optional
        The size of the frame, by default (640, 480)

    Returns
    -------
    np.ndarray
        Point cloud
    """
    pcl_data = np.array(data.getFirstLayerFp16()).reshape(
        1, 3, frame_size[1], frame_size[0]
    )
    return pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
