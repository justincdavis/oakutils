# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for creating and using depthai Mobilenet Detection Network nodes.

Functions
---------
create_mobilenet_detection_network
    Use to create a mobilenet detection network node
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import depthai as dai
    from typing_extensions import TypeAlias

    from ._model_data import MobilenetData

MobileNetDectionNetwork: TypeAlias = (
    "dai.Node.MobileNetDetectionNetwork | dai.Node.MobileNetSpatialDetectionNetwork"  # type: ignore[name-defined]
)


def _create_mobilenet_detection_network_parameters(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: Path,
    confidence_threshold: float,
    bounding_box_scale_factor: float = 0.5,
    depth_input_link: dai.Node.Output | None = None,
    lower_depth_threshold: int = 100,
    upper_depth_threshold: int = 20000,
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
    *,
    spatial: bool | None = None,
    input_blocking: bool | None = None,
) -> MobileNetDectionNetwork:
    if spatial is None:
        spatial = False

    mobilenet_detection_network: MobileNetDectionNetwork = (
        pipeline.createMobileNetDetectionNetwork()
    )
    if spatial:
        mobilenet_detection_network = pipeline.createMobileNetSpatialDetectionNetwork()
        mobilenet_detection_network.setDepthLowerThreshold(lower_depth_threshold)
        mobilenet_detection_network.setDepthUpperThreshold(upper_depth_threshold)
        mobilenet_detection_network.setBoundingBoxScaleFactor(bounding_box_scale_factor)
        if depth_input_link is None:
            err_msg = "You must set depth_input_link if spatial is True!"
            raise ValueError(err_msg)
        depth_input_link.link(mobilenet_detection_network.inputDepth)

    mobilenet_detection_network.setBlobPath(blob_path)
    mobilenet_detection_network.setConfidenceThreshold(confidence_threshold)
    mobilenet_detection_network.setNumInferenceThreads(num_inference_threads)
    if num_nce_per_inference_thread is not None:
        mobilenet_detection_network.setNumNCEPerInferenceThread(
            num_nce_per_inference_thread,
        )
    if num_pool_frames is not None:
        mobilenet_detection_network.setNumPoolFrames(num_pool_frames)
    if input_blocking is None:
        input_blocking = False
    mobilenet_detection_network.input.setBlocking(input_blocking)

    # link the input
    input_link.link(mobilenet_detection_network.input)

    return mobilenet_detection_network


def _create_mobilenet_detection_network_data(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: Path,
    mobilenet_data: MobilenetData,
) -> MobileNetDectionNetwork:
    return _create_mobilenet_detection_network_parameters(
        pipeline,
        input_link,
        blob_path,
        confidence_threshold=mobilenet_data.confidence_threshold,
        bounding_box_scale_factor=mobilenet_data.bounding_box_scale_factor,
        depth_input_link=mobilenet_data.depth_input_link,
        lower_depth_threshold=mobilenet_data.lower_depth_threshold,
        upper_depth_threshold=mobilenet_data.upper_depth_threshold,
        num_inference_threads=mobilenet_data.num_inference_threads,
        num_nce_per_inference_thread=mobilenet_data.num_nce_per_inference_thread,
        num_pool_frames=mobilenet_data.num_pool_frames,
        spatial=mobilenet_data.spatial,
        input_blocking=mobilenet_data.input_blocking,
    )


def create_mobilenet_detection_network(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: Path,
    confidence_threshold: float | None = None,
    bounding_box_scale_factor: float = 0.5,
    depth_input_link: dai.Node.Output | None = None,
    lower_depth_threshold: int = 100,
    upper_depth_threshold: int = 20000,
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
    *,
    mobilenet_data: MobilenetData | None = None,
    spatial: bool | None = None,
    input_blocking: bool | None = None,
) -> MobileNetDectionNetwork:
    """
    Use to create a Mobilenet Detection Network node.

    Should pass either a single MobilenetData object to the function
    OR pass all the parameters to the function. If using the parameters,
    then the MobilenetData object must be set to None.
    The yolo_data: MobilenetData is required as a keyword only parameter.
    If altering parameters at runtime, then parameters are recommended.
    Otherwise, the MobilenetData object is recommended.
    This function can be used to create either a Mobilenet Detection Network node
    or a Mobilenet Spatial Detection Network node simply by changing the spatial
    parameter to True.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the Mobilenet Detection Network node to
    input_link : dai.Node.Output
        The input link to connect to the Mobilenet Detection Network node
        Example: cam_rgb.preview
    blob_path : str
        The path to the blob file
    mobilenet_data : MobilenetData, optional
        The Mobilenet Detection Network data, by default None
    confidence_threshold : float
        The confidence threshold
    bounding_box_scale_factor : float, optional
        The bounding box scale factor, by default 0.5
    spatial : bool, optional
        Whether or not to use spatial coordinates, by default None
        If None, then False is used
    depth_input_link : dai.Node.Output, optional
        The depth input link to connect to the Mobilenet Spatial Detection Network node
        Example: stereo.depth
        Must be set if spatial is True
    lower_depth_threshold : float, optional
        The lower depth threshold for detections. By default 100 mm
        Only used if spatial is True.
    upper_depth_threshold : float, optional
        The upper depth threshold for detections. By default 20000 mm
        Only used if spatial is True
    num_inference_threads : int, optional
        The number of inference threads, by default 2
    num_nce_per_inference_thread : int, optional
        The number of NCEs per inference thread, by default None
    num_pool_frames : int, optional
        The number of pool frames, by default None
    input_blocking : bool, optional
        Whether or not to use input blocking, by default None
        If None, then False is used

    Returns
    -------
    dai.Node.MobilenetDetectionNetwork | dai.Node.MobilenetSpatialDetectionNetwork
        The Mobilenet Detection Network node

    Raises
    ------
    ValueError
        If mobilenet_data is None and confidence_threshold is None
        If spatial is True and depth_input_link is None

    """
    # handle if mobilenet_data is passed
    if mobilenet_data is not None:
        return _create_mobilenet_detection_network_data(
            pipeline,
            input_link,
            blob_path,
            mobilenet_data,
        )
    # check parameter validity
    if confidence_threshold is None:
        err_msg = "If mobilenet_data is None, then confidence_threshold must be set."
        raise ValueError(err_msg)
    return _create_mobilenet_detection_network_parameters(
        pipeline,
        input_link,
        blob_path,
        confidence_threshold=confidence_threshold,
        bounding_box_scale_factor=bounding_box_scale_factor,
        depth_input_link=depth_input_link,
        lower_depth_threshold=lower_depth_threshold,
        upper_depth_threshold=upper_depth_threshold,
        num_inference_threads=num_inference_threads,
        num_nce_per_inference_thread=num_nce_per_inference_thread,
        num_pool_frames=num_pool_frames,
        spatial=spatial,
        input_blocking=input_blocking,
    )
