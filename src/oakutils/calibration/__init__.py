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
    "create_q_matrix",
    "get_camera_calibration_basic",
    "get_camera_calibration_primary_mono",
    "get_camera_calibration",
    "MonoCalibrationData",
    "StereoCalibrationData",
    "ColorCalibrationData",
    "CalibrationData",
]
