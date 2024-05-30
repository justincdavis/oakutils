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
