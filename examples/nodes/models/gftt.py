# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use the GFTT model."""

from __future__ import annotations

import cv2
import depthai as dai
from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
from oakutils.nodes.models import create_gftt

pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(
    pipeline,
    fps=30,
    preview_size=(640, 480),
)  # set the preview size to the input of the nn

gftt = create_gftt(
    pipeline,
    input_link=cam.preview,
    shaves=6,
)
xout_gftt = create_xout(pipeline, gftt.out, "gftt")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("gftt")

    while True:
        data = queue.get()
        frame = get_nn_bgr_frame(data)

        cv2.imshow("gftt frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
