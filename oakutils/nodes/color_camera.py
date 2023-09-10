from __future__ import annotations

import depthai as dai

from oakutils.tools import get_tuple_from_color_sensor_resolution


def create_color_camera(
    pipeline: dai.Pipeline,
    resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    preview_size: tuple[int, int] = (640, 480),
    set_interleaved: bool | None = None,
    fps: int = 30,
    brightness: int = 0,
    saturation: int = 0,
    contrast: int = 0,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,
    isp_target_size: tuple[int, int] | None = None,
    isp_scale: tuple[int, int] | None = None,
    isp_3a_fps: int | None = None,
) -> dai.node.ColorCamera:
    """Creates a pipeline for the color camera.
    setVideoSize, setStillSize are both automatically called using the tuple from get_tuple_from_color_sensor_resolution.

    Parameters
    ----------
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
        The fps of how often the 3a algorithms will run, by default None
        Reducing this can help with performance onboard the device.
        A common value to reduce CPU usage on device is 15.

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

    if fps < 0 or fps > 60:
        raise ValueError("fps must be between 0 and 60")
    if brightness < -10 or brightness > 10:
        raise ValueError("brightness must be between -10 and 10")
    if saturation < -10 or saturation > 10:
        raise ValueError("saturation must be between -10 and 10")
    if contrast < -10 or contrast > 10:
        raise ValueError("contrast must be between -10 and 10")
    if sharpness < 0 or sharpness > 4:
        raise ValueError("sharpness must be between 0 and 4")
    if luma_denoise < 0 or luma_denoise > 4:
        raise ValueError("luma_denoise must be between 0 and 4")
    if chroma_denoise < 0 or chroma_denoise > 4:
        raise ValueError("chroma_denoise must be between 0 and 4")

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

    return cam
