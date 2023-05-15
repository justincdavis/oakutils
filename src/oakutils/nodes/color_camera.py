from typing import Tuple, Optional

import depthai as dai

from ..tools import get_tuple_from_color_sensor_resolution


def create_color_camera(
    pipeline: dai.Pipeline,
    resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    preview_size: Tuple[int, int] = (640, 480),
    set_interleaved: bool = False,
    fps: int = 30,
    brightness: int = 1,
    saturation: int = 1,
    contrast: int = 1,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,
    isp_target_size: Optional[Tuple[int, int]] = None,
    isp_scale: Optional[Tuple[int, int]] = None,
) -> dai.node.ColorCamera:
    """
    Creates a pipeline for the color camera.
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
        The brightness of the color camera, by default 1
    saturation: int, optional
        The saturation of the color camera, by default 1
    contrast: int, optional
        The contrast of the color camera, by default 1
    sharpness: int, optional
        The sharpness of the color camera, by default 1
    luma_denoise: int, optional
        The luma denoise of the color camera, by default 1
    chroma_denoise: int, optional
        The chroma denoise of the color camera, by default 1
    isp_target_size: Optional[Tuple[int, int]], optional
        Target size for scaled frames from ISP (width, height), by default None
        Allows scaling of the cameras frames on-board the OAK to any size
        Works together with the isp_scale parameter
    isp_scale: Optional[Tuple[int, int]], optional
        The isp scale of the color camera, by default None
        Allows scaling of the cameras frames on-board the OAK to any size
        not just the natively supported resolutions.
        Works together with the isp_target_size parameter
        
    Returns
    -------
    dai.node.ColorCamera
        The color camera node
    """
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

    return cam
