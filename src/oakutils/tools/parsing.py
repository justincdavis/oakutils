from typing import Tuple

import depthai as dai


def get_tuple_from_color_sensor_resolution(
    resolution: dai.ColorCameraProperties.SensorResolution,
) -> Tuple[int, int]:
    """
    Converts a SensorResolution to a tuple

    Parameters
    ----------
    resolution : dai.ColorCameraProperties.SensorResolution
        The resolution to convert

    Returns
    -------
    Tuple[int, int]
        The tuple of the resolution

    Raises
    ------
    ValueError
        If the resolution is invalid
    """
    # P resolutions
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_720_P:
        return (1280, 720)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_800_P:
        return (1280, 800)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_1080_P:
        return (1920, 1080)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_1200_P:
        return (1920, 1200)
    # MP resolutions
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_5_MP:
        return (2592, 1944)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_12_MP:
        return (4056, 3040)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_13_MP:
        return (4096, 3072)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_48_MP:
        return (8000, 6000)
    # K resolutions
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_4_K:
        return (3840, 2160)
    # misc resolutions
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_5312X6000:
        return (5312, 6000)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_1440X1080:
        return (1440, 1080)
    elif resolution == dai.ColorCameraProperties.SensorResolution.THE_4000X3000:
        return (4000, 3000)
    # Anything else is not valid
    else:
        raise ValueError("Invalid resolution in get_tuple_from_color_sensor_resolution")


def get_tuple_from_mono_sensor_resolution(
    resolution: dai.MonoCameraProperties.SensorResolution,
) -> Tuple[int, int]:
    """
    Converts a SensorResolution to a tuple

    Parameters
    ----------
    resolution : dai.MonoCameraProperties.SensorResolution
        The resolution to convert

    Returns
    -------
    Tuple[int, int]
        The tuple of the resolution

    Raises
    ------
    ValueError
        If the resolution is invalid
    """
    # P resolutions
    if resolution == dai.MonoCameraProperties.SensorResolution.THE_400_P:
        return (640, 400)
    elif resolution == dai.MonoCameraProperties.SensorResolution.THE_480_P:
        return (640, 480)
    elif resolution == dai.MonoCameraProperties.SensorResolution.THE_720_P:
        return (1280, 720)
    elif resolution == dai.MonoCameraProperties.SensorResolution.THE_800_P:
        return (1280, 800)
    elif resolution == dai.MonoCameraProperties.SensorResolution.THE_1200_P:
        return (1920, 1200)
    # Anything else is not valid
    else:
        raise ValueError("Invalid resolution in get_tuple_from_mono_sensor_resolution")
