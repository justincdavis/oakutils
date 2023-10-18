import time
import concurrent
from typing import Any

import numpy as np
import depthai as dai

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15, GAUSSIAN_13X13, GAUSSIAN_11X11


TIME_TO_RUN = 30


def check_method_timout(method: callable, name: str, timeout=5) -> Any:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(method)
        try:
            result = future.result(timeout=timeout)
            assert result == 0
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"{name}, timed out after {timeout} seconds")
    return result

def check_network(func: callable):
    check_method_timout(func, func.__name__, timeout=TIME_TO_RUN + 15)  # add 5 seconds to timeout for each model

def check_model(model_paths: list[str], timeout: float) -> None:
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

def test_vpu():
    check_network(lambda: check_model([GAUSSIAN_15X15, GAUSSIAN_13X13, GAUSSIAN_11X11], TIME_TO_RUN / 3))
