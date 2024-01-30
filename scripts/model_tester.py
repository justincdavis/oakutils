# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
This is a testing script for building custom operations for the OAK-D.
Adjust the Custom class to your liking, and then run this script to test it out.
"""
from __future__ import annotations

import time
from collections import deque

import torch
import kornia
import cv2
import depthai as dai
import numpy as np
from typing_extensions import Self
from oakutils.blobs import compile_model
from oakutils.blobs.definitions import AbstractModel, InputType, ModelType
from oakutils.nodes import create_neural_network, create_color_camera, create_xout, get_nn_frame

SHAVES = 6
IMAGE_SIZE = (640, 480)

class Custom(AbstractModel):
    """nn.Module wrapper for a custom operation."""

    def __init__(self: Self) -> None:
        super().__init__()

    @classmethod
    def model_type(cls: Custom) -> ModelType:
        """The type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: Custom) -> list[tuple[str, InputType]]:
        """The names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: Custom) -> list[str]:
        """The names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        # TODO: Fill in with custom functionality and compile
        return image

def main():
    model_path = compile_model(
        Custom,
        {}, 
        cache=False, 
        shape_mapping={
            InputType.FP16: (*IMAGE_SIZE, 3)
        },
        shaves=SHAVES,
    )   
    pipeline = dai.Pipeline()
    cam = create_color_camera(
        pipeline,
        preview_size=IMAGE_SIZE,
    )
    custom_network = create_neural_network(
        pipeline, 
        cam.preview, 
        model_path, 
    )
    streamname = "network"
    passname = "passthrough"
    xout_nn = create_xout(pipeline, custom_network.out, streamname)
    xout_pass = create_xout(pipeline, custom_network.passthrough, passname)
    fps_buffer = deque(maxlen=60)
    with dai.Device(pipeline) as device:
        device.setLogLevel(dai.LogLevel.DEBUG)
        device.setLogOutputLevel(dai.LogLevel.DEBUG)
        queue: dai.DataOutputQueue = device.getOutputQueue(streamname)
        pass_queue: dai.DataOutputQueue = device.getOutputQueue(passname)
        t0 = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            data = queue.get()
            frame = get_nn_frame(
                data,
                channels=3,
                frame_size=IMAGE_SIZE, 
            )
            passdata = pass_queue.get()
            passimage: np.ndarray = passdata.getCvFrame()
            t1 = time.perf_counter()
            fps_buffer.append(1/(t1-t0))
            t0 = t1
            cv2.putText(frame, f"FPS: {np.mean(fps_buffer):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pshape = passimage.shape
            frame = cv2.resize(frame, (pshape[1], pshape[0]))
            sidebyside = np.hstack((frame, passimage))
            cv2.imshow(streamname, sidebyside)
            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == "__main__":
    main()
