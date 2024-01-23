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
    Gets the camera calibration data from the device.
get_camera_calibration_basic
    Gets the camera calibration data from the device, using a basic resolution.
get_camera_calibration_primary_mono
    Gets the camera calibration data from the device, using the primary mono camera.
"""
from ._classes import (
    CalibrationData,
    ColorCalibrationData,
    MonoCalibrationData,
    StereoCalibrationData,
)
from ._funcs import (
    create_q_matrix,
    get_camera_calibration,
    get_camera_calibration_basic,
    get_camera_calibration_primary_mono,
)

__all__ = [
    "CalibrationData",
    "ColorCalibrationData",
    "MonoCalibrationData",
    "StereoCalibrationData",
    "create_q_matrix",
    "get_camera_calibration",
    "get_camera_calibration_basic",
    "get_camera_calibration_primary_mono",
]
