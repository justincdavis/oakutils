# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for creating and using depthai Yolo Detection Network nodes.

Functions
---------
create_yolo_detection_network
    Use to create a yolo detection network node
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import depthai as dai
    from typing_extensions import TypeAlias

    from ._model_data import YolomodelData

YoloDectionNetwork: TypeAlias = (
    "dai.Node.YoloDetectionNetwork | dai.Node.YoloSpatialDetectionNetwork"  # type: ignore[name-defined]
)


def _create_yolo_detection_network_parameters(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    num_classes: int,
    coordinate_size: int,
    anchors: list[float],
    anchor_masks: dict[str, list[int]],
    depth_input_link: dai.Node.Output | None = None,
    lower_depth_threshold: int = 100,
    upper_depth_threshold: int = 20000,
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
    *,
    spatial: bool | None = None,
    input_blocking: bool | None = None,
) -> YoloDectionNetwork:
    if spatial is None:
        spatial = False

    yolo_detection_network: YoloDectionNetwork = pipeline.createYoloDetectionNetwork()
    if spatial:
        yolo_detection_network = pipeline.createYoloSpatialDetectionNetwork()
        yolo_detection_network.setDepthLowerThreshold(lower_depth_threshold)
        yolo_detection_network.setDepthUpperThreshold(upper_depth_threshold)
        if depth_input_link is None:
            err_msg = "You must set depth_input_link if spatial is True!"
            raise ValueError(err_msg)
        depth_input_link.link(yolo_detection_network.inputDepth)

    yolo_detection_network.setBlobPath(blob_path)
    yolo_detection_network.setConfidenceThreshold(confidence_threshold)
    yolo_detection_network.setNumClasses(num_classes)
    yolo_detection_network.setCoordinateSize(coordinate_size)
    yolo_detection_network.setAnchors(anchors)
    yolo_detection_network.setAnchorMasks(anchor_masks)
    yolo_detection_network.setIouThreshold(iou_threshold)
    yolo_detection_network.setNumInferenceThreads(num_inference_threads)
    if num_nce_per_inference_thread is not None:
        yolo_detection_network.setNumNCEPerInferenceThread(num_nce_per_inference_thread)
    if num_pool_frames is not None:
        yolo_detection_network.setNumPoolFrames(num_pool_frames)
    if input_blocking is None:
        input_blocking = False
    yolo_detection_network.input.setBlocking(input_blocking)

    # link the input
    input_link.link(yolo_detection_network.input)

    return yolo_detection_network


def _create_yolo_detection_network_data(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: Path,
    yolo_data: YolomodelData,
) -> YoloDectionNetwork:
    return _create_yolo_detection_network_parameters(
        pipeline,
        input_link,
        blob_path,
        confidence_threshold=yolo_data.confidence_threshold,
        iou_threshold=yolo_data.iou_threshold,
        num_classes=yolo_data.num_classes,
        coordinate_size=yolo_data.coordinate_size,
        anchors=yolo_data.anchors,
        anchor_masks=yolo_data.anchor_masks,
        depth_input_link=yolo_data.depth_input_link,
        lower_depth_threshold=yolo_data.lower_depth_threshold,
        upper_depth_threshold=yolo_data.upper_depth_threshold,
        num_inference_threads=yolo_data.num_inference_threads,
        num_nce_per_inference_thread=yolo_data.num_nce_per_inference_thread,
        num_pool_frames=yolo_data.num_pool_frames,
        spatial=yolo_data.spatial,
        input_blocking=yolo_data.input_blocking,
    )


def create_yolo_detection_network(
    pipeline: dai.Pipeline,
    input_link: dai.Node.Output,
    blob_path: Path,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
    num_classes: int | None = None,
    coordinate_size: int | None = None,
    anchors: list[float] | None = None,
    anchor_masks: dict[str, list[int]] | None = None,
    depth_input_link: dai.Node.Output | None = None,
    lower_depth_threshold: int = 100,
    upper_depth_threshold: int = 20000,
    num_inference_threads: int = 2,
    num_nce_per_inference_thread: int | None = None,
    num_pool_frames: int | None = None,
    *,
    yolo_data: YolomodelData | None = None,
    spatial: bool | None = None,
    input_blocking: bool | None = None,
) -> YoloDectionNetwork:
    """
    Use to create a Yolo Detection Network node.

    Should pass either a single YoloModelData object to the function
    OR pass all the parameters to the function. If using the parameters,
    then the YoloModelData object must be set to None.
    The yolo_data: YoloModelData is required as a keyword only parameter.
    If altering parameters at runtime, then parameters are recommended.
    Otherwise, the YoloModelData object is recommended.
    This function can be used to create either a Yolo Detection Network node
    or a Yolo Spatial Detection Network node simply by changing the spatial
    parameter in the YoloModelData or parameters to True. Corresponding,
    depth_input_link must be set if spatial is True.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the Yolo Detection Network node to
    input_link : dai.Node.Output
        The input link to connect to the Yolo Detection Network node
        Example: cam_rgb.preview
    blob_path : str
        The path to the blob file
    yolo_data : YolomodelData, optional
        The Yolo model data, by default None
        If None, then the other parameters must be set
    confidence_threshold : float, optional
        The confidence threshold, by default None
    iou_threshold : float
        The IOU threshold, by default None
    num_classes : int
        The number of classes the model detects, by default None
    coordinate_size : int
        The coordinate size of each detection, by default None
    anchors : list[int]
        The anchors for the yolo model, by default None
    anchor_masks : dict[str, list[int]]
        The anchor masks for the yolo model, by default None
    spatial : bool, optional
        Whether or not to use spatial coordinates, by default None
        If None, then False is used
    depth_input_link : dai.Node.Output, optional
        The depth input link to connect to the Yolo Spatial Detection Network node
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
    dai.Node.YoloDetectionNetwork | dai.Node.YoloSpatialDetectionNetwork
        The Yolo Detection Network node

    Raises
    ------
    ValueError
        If yolo_data is None and any of the following parameters are None:
        confidence_threshold, iou_threshold, num_classes, coordinate_size,
        anchors, anchor_masks
        If spatial is True and depth_input_link is None

    """
    # handle whether we are using YoloModelData or direct parameters
    if yolo_data is not None:
        return _create_yolo_detection_network_data(
            pipeline,
            input_link,
            blob_path,
            yolo_data,
        )
    # check if the parameters are all set
    if confidence_threshold is None:
        err_msg = "If yolo_data is None, then confidence_threshold must be set."
        raise ValueError(err_msg)
    if iou_threshold is None:
        err_msg = "If yolo_data is None, then iou_threshold must be set."
        raise ValueError(err_msg)
    if num_classes is None:
        err_msg = "If yolo_data is None, then num_classes must be set."
        raise ValueError(err_msg)
    if coordinate_size is None:
        err_msg = "If yolo_data is None, then coordinate_size must be set."
        raise ValueError(err_msg)
    if anchors is None:
        err_msg = "If yolo_data is None, then anchors must be set."
        raise ValueError(err_msg)
    if anchor_masks is None:
        err_msg = "If yolo_data is None, then anchor_masks must be set."
        raise ValueError(err_msg)
    return _create_yolo_detection_network_parameters(
        pipeline,
        input_link,
        blob_path,
        confidence_threshold,
        iou_threshold,
        num_classes,
        coordinate_size,
        anchors,
        anchor_masks,
        depth_input_link,
        lower_depth_threshold,
        upper_depth_threshold,
        num_inference_threads,
        num_nce_per_inference_thread,
        num_pool_frames,
        spatial=spatial,
        input_blocking=input_blocking,
    )
