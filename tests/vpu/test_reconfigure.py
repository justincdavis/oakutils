from __future__ import annotations

import time
import concurrent.futures
from typing import Any

import numpy as np
import depthai as dai

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15, GAUSSIAN_13X13, GAUSSIAN_11X11

from ..helpers import check_device


def check_model(model_paths: list[str], timeout: float) -> None:
    timeout += 10
    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    vpu = VPU()
    for model_path in model_paths:
        print(f"Reconfiguring to {model_path}")
        vpu.reconfigure(model_path)
        t0 = time.perf_counter()
        while True:
            if time.perf_counter() - t0 > timeout:
                break
            data = np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8)
            vpu.run(data)
    return 0

def test_reconfigure():
    time_to_run = 30
    check_device(lambda: check_model([GAUSSIAN_15X15, GAUSSIAN_13X13, GAUSSIAN_11X11], time_to_run / 3), time_to_run)
