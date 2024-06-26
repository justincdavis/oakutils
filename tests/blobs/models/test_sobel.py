# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time

import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame, get_nn_gray_frame
from oakutils.nodes.models import create_sobel

from .utils import eval_model
from ...helpers import check_device, TIME_TO_RUN


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

def test_sobel_1_shave():
    check_device(lambda: check_sobel(3, 1, False, False), TIME_TO_RUN)

def test_sobel_1_shave_gray():
    check_device(lambda: check_sobel(3, 1, False, True), TIME_TO_RUN)

def test_sobel_1_shave_3x3_blur():
    check_device(lambda: check_sobel(3, 1, True, False), TIME_TO_RUN)

def test_sobel_1_shave_3x3_blur_gray():
    check_device(lambda: check_sobel(3, 1, True, True), TIME_TO_RUN)

def test_results():
    eval_model("sobel", (640, 480, 3))
