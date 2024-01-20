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

YoloDectionNetwork: TypeAlias = "dai.Node.YoloDetectionNetwork | dai.Node.YoloSpatialDetectionNetwork"  # type: ignore[name-defined]


def create_yolo_detection_network(
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
    """
    Use to create a Yolo Detection Network node.

    This function can be used to create either a Yolo Detection Network node
    or a Yolo Spatial Detection Network node simply by changing the spatial
    parameter to True.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the Yolo Detection Network node to
    input_link : dai.Node.Output
        The input link to connect to the Yolo Detection Network node
        Example: cam_rgb.preview
    blob_path : str
        The path to the blob file
    confidence_threshold : float
        The confidence threshold
    iou_threshold : float
        The IOU threshold
    num_classes : int
        The number of classes the model detects
    coordinate_size : int
        The coordinate size of each detection
    anchors : list[int]
        The anchors for the yolo model
    anchor_masks : dict[str, list[int]]
        The anchor masks for the yolo model
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
        If spatial is True and depth_input_link is None
    """
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
