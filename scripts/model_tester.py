"""
This is a testing script for building custom operations for the OAK-D.
Adjust the Custom class to your liking, and then run this script to test it out.
"""
from __future__ import annotations

import time

import torch
import kornia
import cv2
import depthai as dai
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
    xout_nn = create_xout(pipeline, custom_network.out, streamname)
    with dai.Device(pipeline) as device:
        device.setLogLevel(dai.LogLevel.DEBUG)
        device.setLogOutputLevel(dai.LogLevel.DEBUG)
        queue: dai.DataOutputQueue = device.getOutputQueue(streamname)
        while True:
            t0 = time.perf_counter()
            data = queue.get()
            frame = get_nn_frame(
                data,
                channels=3,
                frame_size=IMAGE_SIZE, 
            )
            t1 = time.perf_counter()
            cv2.putText(frame, f"FPS: {1/(t1-t0):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(streamname, frame)
            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == "__main__":
    main()
