# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use the VPU abstraction."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15

vpu = VPU()
vpu.reconfigure(GAUSSIAN_15X15)
rng = np.random.Generator(np.random.PCG64())

fps_buffer = deque(maxlen=30)
while True:
    # generate some random data, then send to camera and wait for the result
    data = np.array(rng.integers(0, 255, (640, 480, 3)), dtype=np.uint8)
    t0 = time.perf_counter()
    vpu.run(data)
    t1 = time.perf_counter()
    fps_buffer.append(1.0 / (t1 - t0))
    print(f"FPS: {np.mean(fps_buffer):.2f}")
