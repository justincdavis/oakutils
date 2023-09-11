from __future__ import annotations

import depthai as dai


def get_color_sensor_resolution_from_str(
    resolution: str,
) -> dai.ColorCameraProperties.SensorResolution:
    """Coverts a str to a SensorResolution.

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
    if resolution == "800p":
        return dai.ColorCameraProperties.SensorResolution.THE_800_P
    if resolution == "1080p":
        return dai.ColorCameraProperties.SensorResolution.THE_1080_P
    if resolution == "1200p":
        return dai.ColorCameraProperties.SensorResolution.THE_1200_P
    if resolution == "5mp" or resolution == "5MP":
        return dai.ColorCameraProperties.SensorResolution.THE_5_MP
    if resolution == "12mp" or resolution == "12MP":
        return dai.ColorCameraProperties.SensorResolution.THE_12_MP
    if resolution == "13mp" or resolution == "13MP":
        return dai.ColorCameraProperties.SensorResolution.THE_13_MP
    if resolution == "48mp" or resolution == "48MP":
        return dai.ColorCameraProperties.SensorResolution.THE_48_MP
    if resolution == "4k" or resolution == "4K":
        return dai.ColorCameraProperties.SensorResolution.THE_4_K
    if resolution == "5312x6000":
        return dai.ColorCameraProperties.SensorResolution.THE_5312X6000
    if resolution == "1440x1080":
        return dai.ColorCameraProperties.SensorResolution.THE_1440X1080
    if resolution == "4000x3000":
        return dai.ColorCameraProperties.SensorResolution.THE_4000X3000
    raise ValueError("Invalid resolution in get_color_sensor_resolution_from_str")


def get_tuple_from_color_sensor_resolution(
    resolution: dai.ColorCameraProperties.SensorResolution,
) -> tuple[int, int]:
    """Converts a SensorResolution to a tuple.

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
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_800_P:
        return (1280, 800)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_1080_P:
        return (1920, 1080)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_1200_P:
        return (1920, 1200)
    # MP resolutions
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_5_MP:
        return (2592, 1944)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_12_MP:
        return (4056, 3040)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_13_MP:
        return (4096, 3072)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_48_MP:
        return (8000, 6000)
    # K resolutions
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_4_K:
        return (3840, 2160)
    # misc resolutions
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_5312X6000:
        return (5312, 6000)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_1440X1080:
        return (1440, 1080)
    if resolution == dai.ColorCameraProperties.SensorResolution.THE_4000X3000:
        return (4000, 3000)
    # Anything else is not valid
    raise ValueError("Invalid resolution in get_tuple_from_color_sensor_resolution")


def get_color_sensor_info_from_str(
    resolution: str,
) -> tuple[int, int, dai.ColorCameraProperties.SensorResolution]:
    """Parses a resolution string into a tuple of (width, height, SensorResolution).

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
    """Coverts a str to a SensorResolution.

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
    if resolution == "480p":
        return dai.MonoCameraProperties.SensorResolution.THE_480_P
    if resolution == "720p":
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    if resolution == "800p":
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    if resolution == "1200p":
        return dai.MonoCameraProperties.SensorResolution.THE_1200_P
    raise ValueError("Invalid resolution in get_mono_sensor_resolution_from_str")


def get_tuple_from_mono_sensor_resolution(
    resolution: dai.MonoCameraProperties.SensorResolution,
) -> tuple[int, int]:
    """Converts a SensorResolution to a tuple.

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
    if resolution == dai.MonoCameraProperties.SensorResolution.THE_480_P:
        return (640, 480)
    if resolution == dai.MonoCameraProperties.SensorResolution.THE_720_P:
        return (1280, 720)
    if resolution == dai.MonoCameraProperties.SensorResolution.THE_800_P:
        return (1280, 800)
    if resolution == dai.MonoCameraProperties.SensorResolution.THE_1200_P:
        return (1920, 1200)
    # Anything else is not valid
    raise ValueError("Invalid resolution in get_tuple_from_mono_sensor_resolution")


def get_mono_sensor_info_from_str(
    resolution: str,
) -> tuple[int, int, dai.MonoCameraProperties.SensorResolution]:
    """Parses a resolution string into a tuple of (width, height, SensorResolution).

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
    filter_size: int | None,
) -> dai.StereoDepthProperties.MedianFilter:
    """Converts a str to a MedianFilter.

    Parameters
    ----------
    filter_size : Optional[int]
        The filter size to convert

    Returns
    -------
    dai.StereoDepthProperties.MedianFilter
        The MedianFilter
    """
    minor_version = 21
    if int(str(dai.Version(dai.__version__)).split(".")[1]) >= minor_version:
        median_off = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
        median_3 = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
        median_5 = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
        median_7 = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    else:
        median_off = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
        median_3 = dai.StereoDepthProperties.MedianFilter.MEDIAN_3x3
        median_5 = dai.StereoDepthProperties.MedianFilter.MEDIAN_5x5
        median_7 = dai.StereoDepthProperties.MedianFilter.MEDIAN_7x7
    if filter_size is None:
        return median_off
    if filter_size == 0:
        return median_off
    if filter_size == 3:
        return median_3
    if filter_size == 5:
        return median_5
    if filter_size == 7:
        return median_7
    raise ValueError(
        "Invalid filter size in get_median_filter_from_str, must be 0, 3, 5, 7 or None"
    )
