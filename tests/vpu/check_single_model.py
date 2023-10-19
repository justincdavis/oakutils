import time
import concurrent
from typing import Any

import numpy as np
import depthai as dai

from oakutils import VPU
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15


TIME_TO_RUN = 10


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
    check_method_timout(func, func.__name__, timeout=TIME_TO_RUN + 5)  # add 5 seconds to timeout to account for setup time

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

def test_vpu():
    check_network(lambda: check_model(GAUSSIAN_15X15))
