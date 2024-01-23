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

MobileNetDectionNetwork: TypeAlias = "dai.Node.MobileNetDetectionNetwork | dai.Node.MobileNetSpatialDetectionNetwork"  # type: ignore[name-defined]


def create_mobilenet_detection_network(
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
    """
    Use to create a Mobilenet Detection Network node.

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
        If spatial is True and depth_input_link is None
    """
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
