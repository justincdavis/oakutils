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
Module for creating stereo depth nodes.

Functions
---------
create_stereo_depth
    Creates a stereo depth given only a pipeline object.
create_stereo_depth_from_mono_cameras
    Creates a stereo depth node from a pipeline and two mono cameras.
"""
from __future__ import annotations

import depthai as dai

from .mono_camera import create_left_right_cameras


def create_stereo_depth(
    pipeline: dai.Pipeline,
    resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P,
    fps: int = 60,
    brightness: int = 1,
    saturation: int = 1,
    contrast: int = 1,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,
    isp_3a_fps: int | None = 15,
    input_queue_size: int = 3,
    preset: dai.node.StereoDepth.PresetMode = dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
    align_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.LEFT,
    confidence_threshold: int = 255,
    rectify_edge_color: int = 0,
    median_filter: dai.MedianFilter = dai.MedianFilter.KERNEL_7x7,
    subpixel_fractional_bits: int = 3,
    min_brightness: int = 0,
    max_brightness: int = 255,
    decimation_factor: int = 1,
    decimation_mode: dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode = dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEAN,
    spatial_alpha: float = 0.5,
    spatial_delta: int = 0,
    spatial_radius: int = 2,
    spatial_iterations: int = 1,
    speckle_range: int = 20,
    temporal_alpha: float = 0.5,
    temporal_delta: int = 0,
    temporal_mode: dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3,
    threshold_min_range: int = 200,
    threshold_max_range: int = 25000,
    bilateral_sigma: int = 1,
    stereo_input_queue_size: int = 3,
    *,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
    lr_check: bool | None = None,
    extended_disparity: bool | None = None,
    subpixel: bool | None = None,
    enable_spatial_filter: bool | None = None,
    enable_speckle_filter: bool | None = None,
    enable_temporal_filter: bool | None = None,
    stereo_input_reuse: bool | None = None,
    stereo_input_blocking: bool | None = None,
    stereo_input_wait_for_message: bool | None = None,
) -> tuple[dai.node.StereoDepth, dai.node.MonoCamera, dai.node.MonoCamera]:
    """
    Use to create a stereo depth given only a pipeline object.

    Note:
    Creates mono cameras for the left and right cameras using the create_left_right_cameras function.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the stereo depth node to
    resolution : dai.MonoCameraProperties.SensorResolution, optional
        The resolution of the mono camera, by default dai.MonoCameraProperties.SensorResolution.THE_400_P
    fps: int, optional
        The fps of the mono camera, by default 60
    brightness: int, optional
        The brightness of the mono camera, by default 1
    saturation: int, optional
        The saturation of the mono camera, by default 1
    contrast: int, optional
        The contrast of the mono camera, by default 1
    sharpness: int, optional
        The sharpness of the mono camera, by default 1
    luma_denoise: int, optional
        The luma denoise of the mono camera, by default 1
    chroma_denoise: int, optional
        The chroma denoise of the mono camera, by default 1
    isp_3a_fps: int, optional
        The 3a fps of the mono camera, by default 15
    input_queue_size : int, optional
        The queue size of the input, by default 3
    input_reuse : Optional[bool], optional
        Whether to reuse the previous message, by default None
        If None, will be set to False
    input_blocking : Optional[bool], optional
        Whether to block the input, by default None
        If None, will be set to False
    input_wait_for_message : Optional[bool], optional
        Whether to wait for a message, by default None
        If None, will be set to False
    left : dai.node.MonoCamera
        The left mono camera node
    right : dai.node.MonoCamera
        The right mono camera node
    preset : dai.node.StereoDepth.PresetMode, optional
        The preset mode of the stereo depth node, by default dai.node.StereoDepth.PresetMode.HIGH_DENSITY
    align_socket : dai.CameraBoardSocket, optional
        The camera board socket of the stereo depth node, by default dai.CameraBoardSocket.LEFT
    confidence_threshold : int, optional
        The confidence threshold of the stereo depth node, by default 200
    rectify_edge_color : int, optional
        The rectify edge color of the stereo depth node, by default 0
    median_filter : dai.StereoDepthProperties.MedianFilter, optional
        The median filter of the stereo depth node, by default dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    lr_check : bool, optional
        The left right check of the stereo depth node, by default True
    extended_disparity : bool, optional
        The extended disparity of the stereo depth node, by default False
    subpixel : bool, optional
        The subpixel of the stereo depth node, by default False
    subpixel_fractional_bits : int, optional
        The subpixel fractional bits of the stereo depth node, by default 3
    min_brightness : int, optional
        The min brightness of the stereo depth node, by default 0
    max_brightness : int, optional
        The max brightness of the stereo depth node, by default 255
    decimation_factor : int, optional
        The decimation factor of the stereo depth node, by default 1
    decimation_mode : dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode, optional
        The decimation mode of the stereo depth node, by default dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEAN
    enable_spatial_filter : bool, optional
        The enable spatial filter of the stereo depth node, by default False
    spatial_alpha : float, optional
        The spatial alpha of the stereo depth node, by default 0.5
    spatial_delta : int, optional
        The spatial delta of the stereo depth node, by default 0
    spatial_radius : int, optional
        The spatial radius of the stereo depth node, by default 2
    spatial_iterations : int, optional
        The spatial iterations of the stereo depth node, by default 1
    enable_speckle_filter : bool, optional
        The enable speckle filter of the stereo depth node, by default False
    speckle_range : int, optional
        The speckle range of the stereo depth node, by default 20
    enable_temporal_filter : bool, optional
        The enable temporal filter of the stereo depth node, by default False
    temporal_alpha : float, optional
        The temporal alpha of the stereo depth node, by default 0.5
    temporal_delta : int, optional
        The temporal delta of the stereo depth node, by default 0
    temporal_mode : dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode, optional
        The temporal mode of the stereo depth node, by default dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3
    threshold_min_range : int, optional
        The threshold min range of the stereo depth node, by default 200
    threshold_max_range : int, optional
        The threshold max range of the stereo depth node, by default 25000
    bilateral_sigma : int, optional
        The bilateral sigma of the stereo depth node, by default 1
    stereo_input_queue_size : int, optional
        The queue size of the input, by default 3
    stereo_input_reuse : Optional[bool], optional
        Whether to reuse the previous message, by default None
        If None, will be set to False
    stereo_input_blocking : Optional[bool], optional
        Whether to block the input, by default None
        If None, will be set to False
    stereo_input_wait_for_message : Optional[bool], optional
        Whether to wait for a message, by default None
        If None, will be set to False


    Returns
    -------
    dai.node.StereoDepth
        The stereo depth node
    dai.node.MonoCamera
        The left mono camera node
    dai.node.MonoCamera
        The right mono camera node
    """
    if lr_check is None:
        lr_check = True
    if extended_disparity is None:
        extended_disparity = False
    if subpixel is None:
        subpixel = False
    if enable_spatial_filter is None:
        enable_spatial_filter = False
    if enable_speckle_filter is None:
        enable_speckle_filter = False
    if enable_temporal_filter is None:
        enable_temporal_filter = False

    left_cam, right_cam = create_left_right_cameras(
        pipeline=pipeline,
        resolution=resolution,
        fps=fps,
        brightness=brightness,
        saturation=saturation,
        contrast=contrast,
        sharpness=sharpness,
        luma_denoise=luma_denoise,
        chroma_denoise=chroma_denoise,
        isp_3a_fps=isp_3a_fps,
        input_queue_size=input_queue_size,
        input_reuse=input_reuse,
        input_blocking=input_blocking,
        input_wait_for_message=input_wait_for_message,
    )
    stereo = create_stereo_depth_from_mono_cameras(
        pipeline=pipeline,
        left=left_cam,
        right=right_cam,
        preset=preset,
        align_socket=align_socket,
        confidence_threshold=confidence_threshold,
        rectify_edge_color=rectify_edge_color,
        median_filter=median_filter,
        lr_check=lr_check,
        extended_disparity=extended_disparity,
        subpixel=subpixel,
        subpixel_fractional_bits=subpixel_fractional_bits,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        decimation_factor=decimation_factor,
        decimation_mode=decimation_mode,
        enable_spatial_filter=enable_spatial_filter,
        spatial_alpha=spatial_alpha,
        spatial_delta=spatial_delta,
        spatial_radius=spatial_radius,
        spatial_iterations=spatial_iterations,
        enable_speckle_filter=enable_speckle_filter,
        speckle_range=speckle_range,
        enable_temporal_filter=enable_temporal_filter,
        temporal_alpha=temporal_alpha,
        temporal_delta=temporal_delta,
        temporal_mode=temporal_mode,
        threshold_min_range=threshold_min_range,
        threshold_max_range=threshold_max_range,
        bilateral_sigma=bilateral_sigma,
        input_queue_size=stereo_input_queue_size,
        input_reuse=stereo_input_reuse,
        input_blocking=stereo_input_blocking,
        input_wait_for_message=stereo_input_wait_for_message,
    )
    return (
        stereo,
        left_cam,
        right_cam,
    )


def create_stereo_depth_from_mono_cameras(
    pipeline: dai.Pipeline,
    left: dai.node.MonoCamera,
    right: dai.node.MonoCamera,
    preset: dai.node.StereoDepth.PresetMode = dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
    align_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.LEFT,
    confidence_threshold: int = 255,
    rectify_edge_color: int = 0,
    median_filter: dai.MedianFilter = dai.MedianFilter.KERNEL_7x7,
    subpixel_fractional_bits: int = 3,
    min_brightness: int = 0,
    max_brightness: int = 255,
    decimation_factor: int = 1,
    decimation_mode: dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode = dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEAN,
    spatial_alpha: float = 0.5,
    spatial_delta: int = 0,
    spatial_radius: int = 2,
    spatial_iterations: int = 1,
    speckle_range: int = 20,
    temporal_alpha: float = 0.5,
    temporal_delta: int = 0,
    temporal_mode: dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3,
    threshold_min_range: int = 200,
    threshold_max_range: int = 25000,
    bilateral_sigma: int = 1,
    input_queue_size: int = 3,
    *,
    lr_check: bool | None = None,
    extended_disparity: bool | None = None,
    subpixel: bool | None = None,
    enable_spatial_filter: bool | None = None,
    enable_speckle_filter: bool | None = None,
    enable_temporal_filter: bool | None = None,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
) -> dai.node.StereoDepth:
    """
    Use to create a stereo depth node from a pipeline and two mono cameras.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the stereo depth node to
    left : dai.node.MonoCamera
        The left mono camera node
    right : dai.node.MonoCamera
        The right mono camera node
    preset : dai.node.StereoDepth.PresetMode, optional
        The preset mode of the stereo depth node, by default dai.node.StereoDepth.PresetMode.HIGH_DENSITY
    align_socket : dai.CameraBoardSocket, optional
        The camera board socket of the stereo depth node, by default dai.CameraBoardSocket.LEFT
    confidence_threshold : int, optional
        The confidence threshold of the stereo depth node, by default 200
    rectify_edge_color : int, optional
        The rectify edge color of the stereo depth node, by default 0
    median_filter : dai.StereoDepthProperties.MedianFilter, optional
        The median filter of the stereo depth node, by default dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    lr_check : bool, optional
        The left right check of the stereo depth node, by default True
    extended_disparity : bool, optional
        The extended disparity of the stereo depth node, by default False
    subpixel : bool, optional
        The subpixel of the stereo depth node, by default False
    subpixel_fractional_bits : int, optional
        The subpixel fractional bits of the stereo depth node, by default 3
    min_brightness : int, optional
        The min brightness of the stereo depth node, by default 0
    max_brightness : int, optional
        The max brightness of the stereo depth node, by default 255
    decimation_factor : int, optional
        The decimation factor of the stereo depth node, by default 1
        Valid values are 1, 2, 3, 4
    decimation_mode : dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode, optional
        The decimation mode of the stereo depth node, by default dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEAN
    enable_spatial_filter : bool, optional
        The enable spatial filter of the stereo depth node, by default False
    spatial_alpha : float, optional
        The spatial alpha of the stereo depth node, by default 0.5
        Valid values are 0.0 - 1.0
    spatial_delta : int, optional
        The spatial delta of the stereo depth node, by default 0
    spatial_radius : int, optional
        The spatial radius of the stereo depth node, by default 2
    spatial_iterations : int, optional
        The spatial iterations of the stereo depth node, by default 1
    enable_speckle_filter : bool, optional
        The enable speckle filter of the stereo depth node, by default False
    speckle_range : int, optional
        The speckle range of the stereo depth node, by default 20
    enable_temporal_filter : bool, optional
        The enable temporal filter of the stereo depth node, by default False
    temporal_alpha : float, optional
        The temporal alpha of the stereo depth node, by default 0.5
        Valid values are 0.0 - 1.0
    temporal_delta : int, optional
        The temporal delta of the stereo depth node, by default 0
    temporal_mode : dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode, optional
        The temporal mode of the stereo depth node, by default dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3
    threshold_min_range : int, optional
        The threshold min range of the stereo depth node, by default 200
    threshold_max_range : int, optional
        The threshold max range of the stereo depth node, by default 25000
    bilateral_sigma : int, optional
        The bilateral sigma of the stereo depth node, by default 1
    input_queue_size : int, optional
        The queue size of the input, by default 3
    input_reuse : Optional[bool], optional
        Whether to reuse the previous message, by default None
        If None, will be set to False
    input_blocking : Optional[bool], optional
        Whether to block the input, by default None
        If None, will be set to False
    input_wait_for_message : Optional[bool], optional
        Whether to wait for a message, by default None
        If None, will be set to False

    Returns
    -------
    dai.node.StereoDepth
        The stereo depth node

    Raises
    ------
    ValueError
        If spatial_alpha is not between 0.0 and 1.0
    ValueError
        If temporal_alpha is not between 0.0 and 1.0
    ValueError
        If decimation_factor is not 1,2,3,4
    """
    # parse the inputs
    if lr_check is None:
        lr_check = True
    if extended_disparity is None:
        extended_disparity = False
    if subpixel is None:
        subpixel = False
    if enable_spatial_filter is None:
        enable_spatial_filter = False
    if enable_speckle_filter is None:
        enable_speckle_filter = False
    if enable_temporal_filter is None:
        enable_temporal_filter = False
    # all alpha parameters should be between 0.0 and 1.0
    min_spatial_alpha, max_spacial_alpha = 0.0, 1.0
    if not min_spatial_alpha <= spatial_alpha <= max_spacial_alpha:
        err_msg = "spatial_alpha should be between 0.0 and 1.0"
        raise ValueError(err_msg)
    min_temporal_alpha, max_temporal_alpha = 0.0, 1.0
    if not min_temporal_alpha <= temporal_alpha <= max_temporal_alpha:
        err_msg = "temporal_alpha should be between 0.0 and 1.0"
        raise ValueError(err_msg)
    # decimation should be 1,2,3,4
    if decimation_factor not in [1, 2, 3, 4]:
        err_msg = "decimation_factor should be 1,2,3,4"
        raise ValueError(err_msg)

    stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(preset)

    stereo.setDepthAlign(align_socket)

    stereo.setLeftRightCheck(lr_check)
    stereo.setExtendedDisparity(extended_disparity)
    stereo.setSubpixel(subpixel)
    if subpixel:
        stereo.setSubpixelFractionalBits(subpixel_fractional_bits)

    stereo.setRectifyEdgeFillColor(rectify_edge_color)
    stereo.initialConfig.setConfidenceThreshold(confidence_threshold)
    # The internal DepthAI types are weird here, so we have to do this
    stereo.initialConfig.setMedianFilter(median_filter)  # type: ignore[arg-type]

    config: dai.RawStereoDepthConfig = stereo.initialConfig.get()

    # brightness filter
    config.postProcessing.brightnessFilter.minBrightness = min_brightness
    config.postProcessing.brightnessFilter.maxBrightness = max_brightness

    # decimation filter
    config.postProcessing.decimationFilter.decimationFactor = decimation_factor
    config.postProcessing.decimationFilter.decimationMode = decimation_mode

    # spatial filter
    config.postProcessing.spatialFilter.enable = enable_spatial_filter
    config.postProcessing.spatialFilter.alpha = spatial_alpha
    config.postProcessing.spatialFilter.delta = spatial_delta
    config.postProcessing.spatialFilter.holeFillingRadius = spatial_radius
    config.postProcessing.spatialFilter.numIterations = spatial_iterations

    # speckle filter
    config.postProcessing.speckleFilter.enable = enable_speckle_filter
    config.postProcessing.speckleFilter.speckleRange = speckle_range

    # temporal filter
    config.postProcessing.temporalFilter.enable = enable_temporal_filter
    config.postProcessing.temporalFilter.alpha = temporal_alpha
    config.postProcessing.temporalFilter.delta = temporal_delta
    config.postProcessing.temporalFilter.persistencyMode = temporal_mode

    # threshold filter
    config.postProcessing.thresholdFilter.minRange = threshold_min_range
    config.postProcessing.thresholdFilter.maxRange = threshold_max_range

    # misc
    config.postProcessing.bilateralSigmaValue = bilateral_sigma

    # write back the config
    stereo.initialConfig.set(config)

    if input_reuse is None:
        input_reuse = False
    if input_blocking is None:
        input_blocking = False
    if input_wait_for_message is None:
        input_wait_for_message = False

    stereo.inputConfig.setQueueSize(input_queue_size)
    stereo.inputConfig.setReusePreviousMessage(input_reuse)
    stereo.inputConfig.setBlocking(input_blocking)
    stereo.inputConfig.setWaitForMessage(input_wait_for_message)

    # link nodes
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # return data
    return stereo
