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
Module for converting between depthai types, tuples, and strings.

Functions
---------
get_color_sensor_resolution_from_str
    Use to convert a str to a SensorResolution.
get_tuple_from_color_sensor_resolution
    Use to convert a SensorResolution to a tuple.
get_color_sensor_resolution_from_tuple
    Use to convert a tuple to a SensorResolution.
get_color_sensor_info_from_str
    Use to parse a resolution string into a tuple of (width, height, SensorResolution).
get_mono_sensor_resolution_from_str
    Coverts a str to a SensorResolution.
get_mono_sensor_resolution_from_tuple
    Use to convert a tuple to a SensorResolution.
get_tuple_from_mono_sensor_resolution
    Use to convert a SensorResolution to a tuple.
get_mono_sensor_info_from_str
    Use to parse a resolution string into a tuple of (width, height, SensorResolution).
get_median_filter_from_str
    Use to convert a str to a MedianFilter.
"""
from __future__ import annotations

import depthai as dai


def get_color_sensor_resolution_from_str(
    resolution: str,
) -> dai.ColorCameraProperties.SensorResolution:
    """
    Use to convert a str to a SensorResolution.

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
    err_msg = "Invalid resolution in get_color_sensor_resolution_from_str"
    raise ValueError(err_msg)


def get_tuple_from_color_sensor_resolution(
    resolution: dai.ColorCameraProperties.SensorResolution,
) -> tuple[int, int]:
    """
    Use to convert a SensorResolution to a tuple.

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
    err_msg = "Invalid resolution in get_tuple_from_color_sensor_resolution"
    raise ValueError(err_msg)


def get_color_sensor_resolution_from_tuple(
    resolution: tuple[int, int],
) -> dai.ColorCameraProperties.SensorResolution:
    """
    Use to convert a tuple to a SensorResolution.

    Parameters
    ----------
    resolution : tuple[int, int]
        The resolution to convert

    Returns
    -------
    dai.ColorCameraProperties.SensorResolution
        The SensorResolution

    Raises
    ------
    ValueError
        If the resolution is invalid
    """
    if resolution == (1280, 720):
        return dai.ColorCameraProperties.SensorResolution.THE_720_P
    if resolution == (1280, 800):
        return dai.ColorCameraProperties.SensorResolution.THE_800_P
    if resolution == (1920, 1080):
        return dai.ColorCameraProperties.SensorResolution.THE_1080_P
    if resolution == (1920, 1200):
        return dai.ColorCameraProperties.SensorResolution.THE_1200_P
    if resolution == (2592, 1944):
        return dai.ColorCameraProperties.SensorResolution.THE_5_MP
    if resolution == (4056, 3040):
        return dai.ColorCameraProperties.SensorResolution.THE_12_MP
    if resolution == (4096, 3072):
        return dai.ColorCameraProperties.SensorResolution.THE_13_MP
    if resolution == (8000, 6000):
        return dai.ColorCameraProperties.SensorResolution.THE_48_MP
    if resolution == (3840, 2160):
        return dai.ColorCameraProperties.SensorResolution.THE_4_K
    if resolution == (5312, 6000):
        return dai.ColorCameraProperties.SensorResolution.THE_5312X6000
    if resolution == (1440, 1080):
        return dai.ColorCameraProperties.SensorResolution.THE_1440X1080
    if resolution == (4000, 3000):
        return dai.ColorCameraProperties.SensorResolution.THE_4000X3000
    err_msg = "Invalid resolution in get_color_resolution_from_tuple"
    raise ValueError(err_msg)


def get_color_sensor_info_from_str(
    resolution: str,
) -> tuple[int, int, dai.ColorCameraProperties.SensorResolution]:
    """
    Use to parse a resolution string into a tuple of (width, height, SensorResolution).

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
    Coverts a str to a SensorResolution.

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
    err_msg = "Invalid resolution in get_mono_sensor_resolution_from_str"
    raise ValueError(err_msg)


def get_mono_sensor_resolution_from_tuple(
    resolution: tuple[int, int],
) -> dai.MonoCameraProperties.SensorResolution:
    """
    Use to convert a tuple to a SensorResolution.

    Parameters
    ----------
    resolution : tuple[int, int]
        The resolution to convert

    Returns
    -------
    dai.MonoCameraProperties.SensorResolution
        The SensorResolution

    Raises
    ------
    ValueError
        If the resolution is invalid
    """
    if resolution == (640, 400):
        return dai.MonoCameraProperties.SensorResolution.THE_400_P
    if resolution == (640, 480):
        return dai.MonoCameraProperties.SensorResolution.THE_480_P
    if resolution == (1280, 720):
        return dai.MonoCameraProperties.SensorResolution.THE_720_P
    if resolution == (1280, 800):
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    if resolution == (1920, 1200):
        return dai.MonoCameraProperties.SensorResolution.THE_1200_P
    err_msg = "Invalid resolution in get_mono_sensor_resolution_from_tuple"
    raise ValueError(err_msg)


def get_tuple_from_mono_sensor_resolution(
    resolution: dai.MonoCameraProperties.SensorResolution,
) -> tuple[int, int]:
    """
    Use to convert a SensorResolution to a tuple.

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
    err_msg = "Invalid resolution in get_tuple_from_mono_sensor_resolution"
    raise ValueError(err_msg)


def get_mono_sensor_info_from_str(
    resolution: str,
) -> tuple[int, int, dai.MonoCameraProperties.SensorResolution]:
    """
    Use to parse a resolution string into a tuple of (width, height, SensorResolution).

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
) -> dai.MedianFilter:
    """
    Use to convert a str to a MedianFilter.

    Parameters
    ----------
    filter_size : Optional[int]
        The filter size to convert

    Returns
    -------
    dai.StereoDepthProperties.MedianFilter
        The MedianFilter
    """
    median_off = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
    median_3 = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
    median_5 = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
    median_7 = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
    median_filter_map = {
        None: median_off,
        0: median_off,
        3: median_3,
        5: median_5,
        7: median_7,
    }
    try:
        return median_filter_map[filter_size]
    except KeyError as err:
        err_msg = "Invalid filter size in get_median_filter_from_str, must be 0, 3, 5, 7 or None"
        raise ValueError(err_msg) from err
