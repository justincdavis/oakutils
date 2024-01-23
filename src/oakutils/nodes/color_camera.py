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
Module for creating a color camera node in the pipeline.

Functions
---------
create_color_camera
    Creates a pipeline for the color camera.
"""
from __future__ import annotations

import depthai as dai

from oakutils.tools import get_tuple_from_color_sensor_resolution


def create_color_camera(
    pipeline: dai.Pipeline,
    resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    preview_size: tuple[int, int] = (640, 480),
    fps: int = 30,
    brightness: int = 0,
    saturation: int = 0,
    contrast: int = 0,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,
    isp_target_size: tuple[int, int] | None = None,
    isp_scale: tuple[int, int] | None = None,
    isp_3a_fps: int | None = 15,
    input_queue_size: int = 3,
    *,
    set_interleaved: bool | None = None,
    input_reuse: bool | None = None,
    input_blocking: bool | None = None,
    input_wait_for_message: bool | None = None,
) -> dai.node.ColorCamera:
    """
    Use to create a pipeline for the color camera.

    Note:
    setVideoSize, setStillSize are both automatically called using the tuple from get_tuple_from_color_sensor_resolution.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the color camera to
    resolution : dai.ColorCameraProperties.SensorResolution, optional
        The resolution of the color camera, by default dai.ColorCameraProperties.SensorResolution.THE_1080_P
    preview_size : Tuple[int, int], optional
        The size of the preview, by default (640, 480)
    set_interleaved : bool, optional
        Whether to set the color camera to interleaved or not, by default False
    fps: int, optional
        The fps of the color camera, by default 30
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
    isp_target_size: Optional[Tuple[int, int]], optional
        Target size for scaled frames from ISP (width, height), by default None
        Allows scaling of the cameras frames on-board the OAK to any size
        Works together with the isp_scale parameter
    isp_scale: Optional[Tuple[int, int]], optional
        The isp scale of the color camera, by default None
        Allows scaling of the cameras frames on-board the OAK to any size
        not just the natively supported resolutions.
        Works together with the isp_target_size parameter
    isp_3a_fps: Optional[int], optional
        The fps of how often the 3a algorithms will run, by default 15
        Reducing this can help with performance onboard the device.
        A common value to reduce CPU usage on device is 15.
        Reference: https://docs.luxonis.com/projects/api/en/latest/tutorials/debugging/#resource-debugging
    input_queue_size: int, optional
        The size of the input queue, by default 3
    input_reuse: bool, optional
        Whether to reuse inputs or not, by default None
        If none, will be set to False
    input_blocking: bool, optional
        Whether to block the input or not, by default None
        If none, will be set to False
    input_wait_for_message: bool, optional
        Whether to wait for a message or not, by default None
        If none, will be set to False


    Returns
    -------
    dai.node.ColorCamera
        The color camera node

    Raises
    ------
    ValueError
        If the fps is not between 0 and 60
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
    if set_interleaved is None:
        set_interleaved = False

    min_fps, max_fps = 0, 60
    if fps < min_fps or fps > max_fps:
        err_msg = "fps must be between 0 and 60"
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
    if input_reuse is None:
        input_reuse = False
    if input_blocking is None:
        input_blocking = False
    if input_wait_for_message is None:
        input_wait_for_message = False

    size_tuple = get_tuple_from_color_sensor_resolution(resolution)

    # static properties
    cam: dai.node.ColorCamera = pipeline.create(dai.node.ColorCamera)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.initialControl.setBrightness(brightness)
    cam.initialControl.setSaturation(saturation)
    cam.initialControl.setContrast(contrast)
    cam.initialControl.setSharpness(sharpness)
    cam.initialControl.setLumaDenoise(luma_denoise)
    cam.initialControl.setChromaDenoise(chroma_denoise)

    # properties that the user can change
    cam.setPreviewSize(preview_size)
    cam.setResolution(resolution)
    cam.setInterleaved(set_interleaved)
    cam.setFps(fps)

    if isp_scale is not None and isp_target_size is not None:
        cam.setIspScale(*isp_scale)
        cam.setVideoSize(isp_target_size)
        cam.setStillSize(isp_target_size)
    else:
        cam.setVideoSize(size_tuple)
        cam.setStillSize(size_tuple)

    if isp_3a_fps is not None:
        cam.setIsp3aFps(isp_3a_fps)

    cam.inputConfig.setQueueSize(input_queue_size)
    cam.inputConfig.setReusePreviousMessage(input_reuse)
    cam.inputConfig.setBlocking(input_blocking)
    cam.inputConfig.setWaitForMessage(input_wait_for_message)

    return cam
