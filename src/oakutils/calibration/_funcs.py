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
from __future__ import annotations

from typing import TYPE_CHECKING

import depthai as dai

from ._oak1 import get_oak1_calibration, get_oak1_calibration_basic
from ._oakd import get_oakd_calibration, get_oakd_calibration_basic

if TYPE_CHECKING:
    from ._classes import CalibrationData, ColorCalibrationData


def get_camera_calibration(
    rgb_size: tuple[int, int] | None = None,
    mono_size: tuple[int, int] | None = None,
    device: dai.DeviceBase | None = None,
    *,
    basic: bool | None = None,
    is_primary_mono_left: bool | None = None,
) -> CalibrationData | ColorCalibrationData:
    """
    Get the calibration of your OAK camera.

    This function will return a different calibration datatype
    depending on which model camera is being used.
    OAK-D devices will return a CalibrationData object.
    OAK-1 devices will return a ColorCalibrationData object.

    Parameters
    ----------
    device : Optional[dai.DeviceBase], optional
        DepthAI device object.
    rgb_size : Optional[Tuple[int, int]], optional
        RGB camera resolution.
    mono_size : Optional[Tuple[int, int]], optional
        Mono camera resolution.
    basic : Optional[bool], optional
        Whether to get basic calibration data.
        If True, no computed data will be included such as cv2.remap
        or o3d.PinholeCameraIntrinsic (even if o3d is available).
    is_primary_mono_left : Optional[bool], optional
        Whether the primary mono camera is the left camera.
        This parameter is only used for cameras with depth
        imaging onboard.

    Returns
    -------
    CalibrationData | ColorCalibrationData
        Object containing all the calibration data.

    Raises
    ------
    ValueError
        If the camera type is not recognized.
        If the RGB size is not provided for OAK-1 devices.
        If the RGB and mono sizes are not provided for OAK-D devices.

    """
    if basic is None:
        basic = False

    if device is None:
        device = dai.Device()

    camtype = device.getDeviceName()

    if camtype == "OAK-D":
        if rgb_size is None or mono_size is None:
            err_msg = "RGB and mono sizes are required for OAK-D devices."
            raise ValueError(err_msg)
        if basic:
            return get_oakd_calibration_basic(
                device=device,
                rgb_size=rgb_size,
                mono_size=mono_size,
            )
        return get_oakd_calibration(
            device=device,
            rgb_size=rgb_size,
            mono_size=mono_size,
            is_primary_mono_left=is_primary_mono_left,
        )
    if camtype == "OAK-1":
        if rgb_size is None:
            err_msg = "RGB size is required for OAK-1 devices."
            raise ValueError(err_msg)
        if basic:
            return get_oak1_calibration_basic(device=device, rgb_size=rgb_size)
        return get_oak1_calibration(device=device, rgb_size=rgb_size)
    err_msg = f"Unknown camera type: {camtype}"
    raise ValueError(err_msg)
