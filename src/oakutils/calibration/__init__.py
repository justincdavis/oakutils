# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Module for camera calibration utilities.

Classes
-------
CalibrationData
    A class for storing calibration data.
ColorCalibrationData
    A class for storing color camera calibration data.
MonoCalibrationData
    A class for storing mono camera calibration data.
StereoCalibrationData
    A class for storing stereo camera calibration data.

Functions
---------
create_q_matrix
    Creates a Q matrix from a stereo calibration.
get_camera_calibration
    Gets the camera calibration data from any device.
get_oak1_calibration
    Gets the camera calibration data from the OAK-1 device.
get_oak1_calibration_basic
    Gets the camera calibration data from the OAK-1 device, without computed info.
get_oakd_calibration
    Gets the camera calibration data from the OAK-D device.
get_oakd_calibration_basic
    Gets the camera calibration data from the OAK-D device, without computed info.
get_oakd_calibration_primary_mono
    Gets the camera calibration data from the OAK-D device, using the primary mono camera.
"""

from __future__ import annotations

import logging

from ._classes import (
    CalibrationData,
    ColorCalibrationData,
    MonoCalibrationData,
    StereoCalibrationData,
)
from ._funcs import get_camera_calibration
from ._oak1 import get_oak1_calibration, get_oak1_calibration_basic
from ._oakd import (
    create_q_matrix,
    get_oakd_calibration,
    get_oakd_calibration_basic,
    get_oakd_calibration_primary_mono,
)

_log = logging.getLogger(__name__)

__all__ = [
    "CalibrationData",
    "ColorCalibrationData",
    "MonoCalibrationData",
    "StereoCalibrationData",
    "create_q_matrix",
    "get_camera_calibration",
    "get_oak1_calibration",
    "get_oak1_calibration_basic",
    "get_oakd_calibration",
    "get_oakd_calibration_basic",
    "get_oakd_calibration_primary_mono",
]

_log.debug("Loaded calibration")
