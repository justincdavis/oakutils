from __future__ import annotations

import time
import concurrent.futures
from typing import Any

import numpy as np
import depthai as dai

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15, LAPLACIAN_15X15

from ..helpers import check_device


def check_model(model_paths: list[str], time_to_run: int) -> None:
    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    vpu = VPU()
    vpu.reconfigure_multi(model_paths)
    t0 = time.perf_counter()
    while True:
        if time.perf_counter() - t0 > time_to_run:
            break
        data = [np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8) for _ in range(len(model_paths))]
        vpu.run(data)
    return 0

def test_multi_model_reconfigure():
    check_device(lambda: check_model([GAUSSIAN_15X15, LAPLACIAN_15X15], time_to_run=15), timeout=30)
