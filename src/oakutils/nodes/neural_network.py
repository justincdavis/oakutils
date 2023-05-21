from typing import Tuple, Optional, Union

import numpy as np
import depthai as dai


def create_neural_network(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: str,
    stream_name: str = "nn",
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: Optional[int] = None,
    num_pool_frames: Optional[int] = None,
) -> Tuple[dai.node.NeuralNetwork, dai.node.XLinkOut]:
    """
    Creates a neural network node.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the neural network to
    input_link : dai.node.XLinkOut
        The input link to connect to the image manip node.
        Example: cam_rgb.preview.link
        Explicitly pass in the link as a non-called function.
    blob_path : str
        The path to the blob file to use for the neural network.
    stream_name : str, optional
        The name of the stream, by default "nn"
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
    dai.node.XLinkOut
        The output link of the neural network node, stream name is default "nn"
    """
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
    input_link.link(nn.input)

    # create the output link
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName(stream_name)
    nn.out.link(xout_nn.input)

    return nn, xout_nn


def get_nn_bgr_frame(
    data: Union[np.ndarray, dai.NNData], frame_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Takes the raw data output from a neural network execution and converts it to a BGR frame
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
    frame = (frame * 255 + 127.5).astype(np.uint8)
    return frame


def get_nn_gray_frame(
    data: Union[np.ndarray, dai.NNData], frame_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Takes the raw data output from a neural network execution and converts it to a grayscale frame
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
    frame = (frame * 255 + 127.5).astype(np.uint8)
    return frame
