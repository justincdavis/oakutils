# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to reconfigure the VPU on the fly."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15, LAPLACIAN_15X15

SWAP_TIME = 10


def get_model(current_model: str) -> str:
    """Get the next model to use."""
    if current_model == GAUSSIAN_15X15:
        current_model = LAPLACIAN_15X15
    else:
        current_model = GAUSSIAN_15X15
    return current_model


vpu = VPU()
current_model = GAUSSIAN_15X15
vpu.reconfigure(current_model)
rng = np.random.Generator(np.random.PCG64())
fps_buffer = deque(maxlen=SWAP_TIME)
counter = 0
while True:
    # reconfigure every 10 frames
    if counter == SWAP_TIME - 1:
        current_model = get_model(current_model)
        vpu.reconfigure(current_model)
        counter = 0
        fps_buffer.clear()
    # generate some random data, then send to camera and wait for the result
    data = np.array(rng.integers(0, 255, (640, 480, 3)), dtype=np.uint8)
    t0 = time.perf_counter()
    vpu.run(data)
    t1 = time.perf_counter()
    fps_buffer.append(1.0 / (t1 - t0))
    print(f"FPS: {np.mean(fps_buffer):.2f}")
    counter += 1
