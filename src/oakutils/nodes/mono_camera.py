from typing import Tuple

import depthai as dai


def create_mono_camera(
    pipeline: dai.Pipeline,
    socket: dai.CameraBoardSocket,
    resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P,
    fps: int = 60,
    brightness: int = 1,
    saturation: int = 1,
    contrast: int = 1,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,        
) -> dai.node.MonoCamera:
    """
    Creates a pipeline for the mono camera.

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

    Returns
    -------
    dai.node.MonoCamera
        The mono camera node
    """
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

    return cam


def create_left_right_cameras(
    pipeline: dai.Pipeline,
    resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P,
    fps: int = 60,
    brightness: int = 1,
    saturation: int = 1,
    contrast: int = 1,
    sharpness: int = 1,
    luma_denoise: int = 1,
    chroma_denoise: int = 1,  
) -> Tuple[dai.node.MonoCamera, dai.node.MonoCamera]:
    """
    Wrapper function for creating the left and right mono cameras.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to add the two mono cameras to
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

    Returns
    -------
    dai.node.MonoCamera
        The left mono camera node
    dai.node.MonoCamera
        The right mono camera node
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
    )

    return left_cam, right_cam
