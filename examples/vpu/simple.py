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
"""Example showcasing how to use the VPU abstraction."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15

vpu = VPU()
vpu.reconfigure(GAUSSIAN_15X15)

fps_buffer = deque(maxlen=30)
while True:
    # generate some random data, then send to camera and wait for the result
    data = np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8)
    t0 = time.perf_counter()
    vpu.run(data)
    t1 = time.perf_counter()
    fps_buffer.append(1.0 / (t1 - t0))
    print(f"FPS: {np.mean(fps_buffer):.2f}")
