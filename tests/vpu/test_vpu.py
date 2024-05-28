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

import time

import numpy as np
import depthai as dai

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15

from ..helpers import check_device, TIME_TO_RUN


def check_model(model_path: str) -> None:
    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    vpu = VPU()
    vpu.reconfigure(model_path)
    t0 = time.perf_counter()
    while True:
        if time.perf_counter() - t0 > TIME_TO_RUN:
            break
        data = np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8)
        vpu.run(data)
    return 0

def test_vpu_basic():
    check_device(lambda: check_model(GAUSSIAN_15X15), TIME_TO_RUN)
