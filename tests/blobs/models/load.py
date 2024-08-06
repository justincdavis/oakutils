# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from collections.abc import Callable

import depthai as dai
from oakutils.nodes import create_color_camera, create_xout

from ...device import get_device_count


def create_model(modelfunc: Callable) -> int:
    pipeline = dai.Pipeline()
    cam = create_color_camera(pipeline)
    model = modelfunc(pipeline, cam.preview)
    xout_model = create_xout(pipeline, model.out, "model_out")

    all_nodes = [
        cam,
        model,
        xout_model,
    ]
    assert len(all_nodes) == 3
    for node in all_nodes:
        assert node is not None
    return 0


def run_model(modelfunc: Callable, decodefunc: Callable) -> int:
    pipeline = dai.Pipeline()
    cam = create_color_camera(pipeline)
    model = modelfunc(pipeline, cam.preview)
    xout_model = create_xout(pipeline, model.out, "model_out")

    all_nodes = [
        cam,
        model,
        xout_model,
    ]
    assert len(all_nodes) == 3
    for node in all_nodes:
        assert node is not None

    if get_device_count() == 0:
        return 0
    
    with dai.Device(pipeline) as device:
        queue: dai.DataOutputQueue = device.getOutputQueue("model_out")

        while True:
            data = queue.get()
            frame = decodefunc(data)
            assert frame is not None
            break
    return 0
