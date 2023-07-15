from typing import Tuple, Optional

import depthai as dai


def get_color_sensor_resolution_from_str(
    resolution: str,
) -> dai.ColorCameraProperties.SensorResolution:
    """
    Coverts a str to a SensorResolution

    Parameters
    ----------
    resolution : str
        The resolution to convert

    Returns
    -------
    dai.ColorCameraProperties.SensorResolution
        The SensorResolution
    """
    if resolution == "720p":
        return dai.ColorCameraProperties.SensorResolution.THE_720_P
    elif resolution == "800p":
        return dai.ColorCameraProperties.SensorResolution.THE_800_P
    elif resolution == "1080p":
        return dai.ColorCameraProperties.SensorResolution.THE_1080_P
    elif resolution == "1200p":
        return dai.ColorCameraProperties.SensorResolution.THE_1200_P
    elif resolution == "5mp" or resolution == "5MP":
        return dai.ColorCameraProperties.SensorResolution.THE_5_MP
    elif resolution == "12mp" or resolution == "12MP":
        return dai.ColorCameraProperties.SensorResolution.THE_12_MP
    elif resolution == "13mp" or resolution == "13MP":
        return dai.ColorCameraProperties.SensorResolution.THE_13_MP
    elif resolution == "48mp" or resolution == "48MP":
        return dai.ColorCameraProperties.SensorResolution.THE_48_MP
    elif resolution == "4k" or resolution == "4K":
        return dai.ColorCameraProperties.SensorResolution.THE_4_K
    elif resolution == "5312x6000":
        return dai.ColorCameraProperties.SensorResolution.THE_5312X6000
    elif resolution == "1440x1080":
        return dai.ColorCameraProperties.SensorResolution.THE_1440X1080
    elif resolution == "4000x3000":
        return dai.ColorCameraProperties.SensorResolution.THE_4000X3000
    else:
        raise ValueError("Invalid resolution in get_color_sensor_resolution_from_str")


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


def get_color_sensor_info_from_str(
    resolution: str,
) -> Tuple[int, int, dai.ColorCameraProperties.SensorResolution]:
    """
    Parses a resolution string into a tuple of (width, height, SensorResolution)

    Parameters
    ----------
    resolution : str
        The resolution to parse

    Returns
    -------
    Tuple[int, int, dai.ColorCameraProperties.SensorResolution]
        The tuple of (width, height, SensorResolution)
    """
    sensor_res = get_color_sensor_resolution_from_str(resolution)
    width, height = get_tuple_from_color_sensor_resolution(sensor_res)
    return (width, height, sensor_res)


def get_mono_sensor_resolution_from_str(
    resolution: str,
) -> dai.MonoCameraProperties.SensorResolution:
    """
    Coverts a str to a SensorResolution

    Parameters
    ----------
    resolution : str
        The resolution to convert

    Returns
    -------
    dai.MonoCameraProperties.SensorResolution
        The SensorResolution
    """
    if resolution == "400p":
        return dai.MonoCameraProperties.SensorResolution.THE_400_P
    elif resolution == "480p":
        return dai.MonoCameraProperties.SensorResolution.THE_480_P
    elif resolution == "720p":
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    elif resolution == "800p":
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    elif resolution == "1200p":
        return dai.MonoCameraProperties.SensorResolution.THE_1200_P
    else:
        raise ValueError("Invalid resolution in get_mono_sensor_resolution_from_str")


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


def get_mono_sensor_info_from_str(
    resolution: str,
) -> Tuple[int, int, dai.MonoCameraProperties.SensorResolution]:
    """
    Parses a resolution string into a tuple of (width, height, SensorResolution)

    Parameters
    ----------
    resolution : str
        The resolution to parse

    Returns
    -------
    Tuple[int, int, dai.MonoCameraProperties.SensorResolution]
        The tuple of (width, height, SensorResolution)
    """
    sensor_res = get_mono_sensor_resolution_from_str(resolution)
    width, height = get_tuple_from_mono_sensor_resolution(sensor_res)
    return (width, height, sensor_res)


def get_median_filter_from_str(
    filter_size: Optional[int],
) -> dai.StereoDepthProperties.MedianFilter:
    """
    Converts a str to a MedianFilter

    Parameters
    ----------
    filter_size : Optional[int]
        The filter size to convert

    Returns
    -------
    dai.StereoDepthProperties.MedianFilter
        The MedianFilter
    """
    if filter_size is None:
        return dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
    elif filter_size == 0:
        return dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
    elif filter_size == 3:
        return dai.StereoDepthProperties.MedianFilter.MEDIAN_3x3
    elif filter_size == 5:
        return dai.StereoDepthProperties.MedianFilter.MEDIAN_5x5
    elif filter_size == 7:
        return dai.StereoDepthProperties.MedianFilter.MEDIAN_7x7
    else:
        raise ValueError(
            "Invalid filter size in get_median_filter_from_str, must be 0, 3, 5, 7 or None"
        )
