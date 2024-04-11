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

import depthai as dai

from ._classes import CalibrationData, ColorCalibrationData
from ._oak1 import get_oakd_calibration
from ._oakd import get_oakd_calibration


def get_camera_calibration(
    device: dai.DeviceBase | None = None,
    rgb_size: tuple[int, int] | None = None,
    mono_size: tuple[int, int] | None = None,
    *,
    is_primary_left: bool | None = None,
) -> CalibrationData | ColorCalibrationData:
    