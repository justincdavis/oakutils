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
Module for creating mono camera nodes.

Functions
---------
create_mono_camera
    Creates a pipeline for the mono camera.
create_left_right_cameras
    Wrapper function for creating the left and right mono cameras.
"""
from __future__ import annotations

import depthai as dai


def create_mono_camera(
    pipeline: dai.Pipeline,
    socket: dai.CameraBoardSocket,
    resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P,
    fps: int = 60,
    brightness: int = 0,
    saturation: int = 0,
    contrast: int = 0,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,
    isp_3a_fps: int | None = 15,
    input_queue_size: int = 3,
    *,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
) -> dai.node.MonoCamera:
    """
    Use to create a pipeline for the mono camera.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the mono camera to
    socket : dai.CameraBoardSocket
        The socket the camera is plugged into
    resolution : dai.MonoCameraProperties.SensorResolution, optional
        The resolution of the mono camera, by default dai.MonoCameraProperties.SensorResolution.THE_400_P
    fps: int, optional
        The fps of the mono camera, by default 60
    brightness: int, optional
        The brightness of the mono camera, by default 0
        Valid values are -10 ... 10
    saturation: int, optional
        The saturation of the mono camera, by default 0
        Valid values are -10 ... 10
    contrast: int, optional
        The contrast of the mono camera, by default 0
        Valid values are -10 ... 10
    sharpness: int, optional
        The sharpness of the mono camera, by default 1
        Valid values are 0 ... 4
    luma_denoise: int, optional
        The luma denoise of the mono camera, by default 1
        Valid values are 0 ... 4
    chroma_denoise: int, optional
        The chroma denoise of the mono camera, by default 1
        Valid values are 0 ... 4
    isp_3a_fps: Optional[int], optional
        The fps of how often the 3a algorithms will run, by default 15
        Reducing this can help with performance onboard the device.
        A common value to reduce CPU usage on device is 15.
        Reference: https://docs.luxonis.com/projects/api/en/latest/tutorials/debugging/#resource-debugging
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
    dai.node.MonoCamera
        The mono camera node

    Raises
    ------
    ValueError
        If the socket is not LEFT or RIGHT
    ValueError
        If the fps is not between 0 and 120
    ValueError
        If the brightness is not between -10 and 10
    ValueError
        If the saturation is not between -10 and 10
    ValueError
        If the contrast is not between -10 and 10
    ValueError
        If the sharpness is not between 0 and 4
    ValueError
        If the luma_denoise is not between 0 and 4
    ValueError
        If the chroma_denoise is not between 0 and 4
    """
    if socket not in (dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT):
        err_msg = "socket must be LEFT or RIGHT"
        raise ValueError(err_msg)

    min_fps, max_fps = 0, 120
    if fps < min_fps or fps > max_fps:
        err_msg = "fps must be between 0 and 120"
        raise ValueError(err_msg)
    min_brightness, max_brightness = -10, 10
    if brightness < min_brightness or brightness > max_brightness:
        err_msg = "brightness must be between -10 and 10"
        raise ValueError(err_msg)
    min_saturation, max_saturation = -10, 10
    if saturation < min_saturation or saturation > max_saturation:
        err_msg = "saturation must be between -10 and 10"
        raise ValueError(err_msg)
    min_contrast, max_contrast = -10, 10
    if contrast < min_contrast or contrast > max_contrast:
        err_msg = "contrast must be between -10 and 10"
        raise ValueError(err_msg)
    min_sharpness, max_sharpness = 0, 4
    if sharpness < min_sharpness or sharpness > max_sharpness:
        err_msg = "sharpness must be between 0 and 4"
        raise ValueError(err_msg)
    min_luma_denoise, max_luma_denoise = 0, 4
    if luma_denoise < min_luma_denoise or luma_denoise > max_luma_denoise:
        err_msg = "luma_denoise must be between 0 and 4"
        raise ValueError(err_msg)
    min_chroma_denoise, max_chroma_denoise = 0, 4
    if chroma_denoise < min_chroma_denoise or chroma_denoise > max_chroma_denoise:
        err_msg = "chroma_denoise must be between 0 and 4"
        raise ValueError(err_msg)

    # static properties
    cam: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(socket)
    cam.initialControl.setBrightness(brightness)
    cam.initialControl.setSaturation(saturation)
    cam.initialControl.setContrast(contrast)
    cam.initialControl.setSharpness(sharpness)
    cam.initialControl.setLumaDenoise(luma_denoise)
    cam.initialControl.setChromaDenoise(chroma_denoise)

    # user defined properties
    cam.setResolution(resolution)
    cam.setFps(fps)

    if isp_3a_fps is not None:
        cam.setIsp3aFps(isp_3a_fps)

    if input_reuse is None:
        input_reuse = False
    if input_blocking is None:
        input_blocking = False
    if input_wait_for_message is None:
        input_wait_for_message = False

    cam.inputControl.setQueueSize(input_queue_size)
    cam.inputControl.setReusePreviousMessage(input_reuse)
    cam.inputControl.setBlocking(input_blocking)
    cam.inputControl.setWaitForMessage(input_wait_for_message)

    return cam


def create_left_right_cameras(
    pipeline: dai.Pipeline,
    resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P,
    fps: int = 60,
    brightness: int = 0,
    saturation: int = 0,
    contrast: int = 0,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,
    isp_3a_fps: int | None = 15,
    input_queue_size: int = 3,
    *,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
) -> tuple[dai.node.MonoCamera, dai.node.MonoCamera]:
    """
    Use to create the left and right mono cameras.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the two mono cameras to
    resolution : dai.MonoCameraProperties.SensorResolution, optional
        The resolution of the mono camera, by default dai.MonoCameraProperties.SensorResolution.THE_400_P
    fps: int, optional
        The fps of the mono camera, by default 60
    brightness: int, optional
        The brightness of the mono camera, by default 0
        Valid values are -10 ... 10
    saturation: int, optional
        The saturation of the mono camera, by default 0
        Valid values are -10 ... 10
    contrast: int, optional
        The contrast of the mono camera, by default 0
        Valid values are -10 ... 10
    sharpness: int, optional
        The sharpness of the mono camera, by default 1
        Valid values are 0 ... 4
    luma_denoise: int, optional
        The luma denoise of the mono camera, by default 1
        Valid values are 0 ... 4
    chroma_denoise: int, optional
        The chroma denoise of the mono camera, by default 1
        Valid values are 0 ... 4
    isp_3a_fps: Optional[int], optional
        The fps of how often the 3a algorithms will run, by default None
        Reducing this can help with performance onboard the device.
        A common value to reduce CPU usage on device is 15.
        Reference: https://docs.luxonis.com/projects/api/en/latest/tutorials/debugging/#resource-debugging
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
    dai.node.MonoCamera
        The left mono camera node
    dai.node.XLinkOut
        The output node for the left mono camera
    dai.node.MonoCamera
        The right mono camera node
    dai.node.XLinkOut
        The output node for the right mono camera
    """
    left_cam = create_mono_camera(
        pipeline=pipeline,
        socket=dai.CameraBoardSocket.LEFT,
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
    right_cam = create_mono_camera(
        pipeline=pipeline,
        socket=dai.CameraBoardSocket.RIGHT,
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

    return left_cam, right_cam
