# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time

import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame, get_nn_gray_frame
from oakutils.nodes.models import create_gaussian

from .utils import eval_model
from ...helpers import check_device, TIME_TO_RUN


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
    check_device(lambda: check_gaussian(3, 1, False), TIME_TO_RUN)

def test_gaussian_3x3_1_shave_gray():
    check_device(lambda: check_gaussian(3, 1, True), TIME_TO_RUN)

def test_gaussian_15x15_6_shave():
    check_device(lambda: check_gaussian(15, 6, False), TIME_TO_RUN)

def test_gaussian_15x15_6_shave_gray():
    check_device(lambda: check_gaussian(15, 6, True), TIME_TO_RUN)

def test_results():
    eval_model("gaussian", (640, 480, 3))
