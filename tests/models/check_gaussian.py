import time
import concurrent
from typing import Any

import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame, get_nn_gray_frame
from oakutils.nodes.models import create_gaussian


TIME_TO_RUN = 10


def check_method_timout(method: callable, name: str, timeout=5) -> Any:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(method)
        try:
            result = future.result(timeout=timeout)
            assert result == 0
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"{name}, timed out after 5 seconds")
    return result

def check_network(func: callable):
    check_method_timout(func, func.__name__, timeout=TIME_TO_RUN + 5)  # add 5 seconds to timeout to account for setup time


def check_gaussian(kernel_size: int, shaves: int, grayscale_out: bool):
    """Test the gaussian node"""
    pipeline = dai.Pipeline()

    cam = create_color_camera(pipeline, preview_size=(640, 480))
    lp = create_gaussian(
        pipeline, 
        cam.preview,
        kernel_size=kernel_size,
        shaves=shaves,
        grayscale_out=grayscale_out,
    )
    _ = create_xout(pipeline, lp.out, "gaussian")

    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    with dai.Device(pipeline) as device:
        l_queue: dai.DataOutputQueue = device.getOutputQueue("gaussian")

        t0 = time.perf_counter()
        while True:
            l_data = l_queue.get()
            if not grayscale_out:
                l_frame = get_nn_bgr_frame(l_data, frame_size=(640, 480), normalization=255.0)
            else:
                l_frame = get_nn_gray_frame(l_data, frame_size=(640, 480), normalization=255.0)
            if time.perf_counter() - t0 > TIME_TO_RUN:
                break
    return 0

def test_gaussian_3x3_1_shave():
    check_network(lambda: check_gaussian(3, 1, False))

def test_gaussian_3x3_1_shave_gray():
    check_network(lambda: check_gaussian(3, 1, True))
