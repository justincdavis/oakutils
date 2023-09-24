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
    isp_3a_fps: int | None = None,
) -> dai.node.MonoCamera:
    """Creates a pipeline for the mono camera.

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
        The fps of how often the 3a algorithms will run, by default None
        Reducing this can help with performance onboard the device.
        A common value to reduce CPU usage on device is 15.

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
        raise ValueError("socket must be LEFT or RIGHT")

    if fps < 0 or fps > 120:
        raise ValueError("fps must be between 0 and 120")
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
    isp_3a_fps: int | None = None,
) -> tuple[dai.node.MonoCamera, dai.node.MonoCamera,]:
    """Wrapper function for creating the left and right mono cameras.

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
    )

    return left_cam, right_cam
