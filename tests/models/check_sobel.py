import time
import concurrent
from typing import Any

import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame, get_nn_gray_frame
from oakutils.nodes.models import create_sobel


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


def check_sobel(kernel_size: int, shaves: int, use_blur: bool, grayscale_out: bool):
    """Test the sobel node"""
    pipeline = dai.Pipeline()

    color_fps = 15 if use_blur else 30
    cam = create_color_camera(pipeline, fps=color_fps, preview_size=(640, 480))
    lp = create_sobel(
        pipeline, 
        cam.preview,
        blur_kernel_size=kernel_size,
        shaves=shaves,
        use_blur=use_blur,
        grayscale_out=grayscale_out,
    )
    _ = create_xout(pipeline, lp.out, "sobel")

    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    with dai.Device(pipeline) as device:
        l_queue: dai.DataOutputQueue = device.getOutputQueue("sobel")

        t0 = time.perf_counter()
        while True:
            l_data = l_queue.get()
            if grayscale_out:
                l_frame = get_nn_gray_frame(l_data, frame_size=(640, 480), normalization=255.0)
            else:
                l_frame = get_nn_bgr_frame(l_data, frame_size=(640, 480), normalization=255.0)
            if time.perf_counter() - t0 > TIME_TO_RUN:
                break
    return 0

def test_sobel_3x3_1_shave():
    check_network(lambda: check_sobel(3, 1, False, False))

def test_sobel_3x3_1_shave_gray():
    check_network(lambda: check_sobel(3, 1, False, True))

def test_sobel_3x3_1_shave_blur():
    check_network(lambda: check_sobel(3, 1, True, False))

def test_sobel_3x3_1_shave_blur_gray():
    check_network(lambda: check_sobel(3, 1, True, True))
