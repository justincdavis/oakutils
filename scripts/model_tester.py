# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: TD002, TD003, FIX002, INP001, T201, TCH002, F401
"""
Testing script for building custom operations for the OAK-D.

Adjust the two Custom classes to your liking, and then run this script to test it out.
The FP16 and U8 models are compiled and run in a pipeline, and the output is displayed.
The FP16 will operate on the color camera image by default and the U8 will
operate on the depth data by default. You can change this by altering the pipeline
construction and create_neural_network calls.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import depthai as dai
import kornia
import numpy as np
import torch
from oakutils import set_log_level
from oakutils.blobs import compile_model
from oakutils.blobs.definitions import AbstractModel, InputType, ModelType
from oakutils.nodes import (
    create_color_camera,
    create_neural_network,
    create_stereo_depth,
    create_xout,
    get_nn_frame,
)
from typing_extensions import Self

SHAVES = 6
FP16_IMAGE_SIZE = (640, 480)
U8_IMAGE_SIZE = (640, 400)


class CustomFP16(AbstractModel):
    """nn.Module wrapper for a custom operation."""

    def __init__(self: Self) -> None:
        """Create a new instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: type[CustomFP16]) -> ModelType:
        """Type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: type[CustomFP16]) -> list[tuple[str, InputType]]:
        """Names of the input tensors."""
        return [("input", InputType.FP16)]

    @classmethod
    def output_names(cls: type[CustomFP16]) -> list[str]:
        """Names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # TODO: Fill in with custom functionality and compile
        return image


class CustomU8(AbstractModel):
    """nn.Module wrapper for a custom operation."""

    def __init__(self: Self) -> None:
        """Create a new instance of the model."""
        super().__init__()

    @classmethod
    def model_type(cls: type[CustomU8]) -> ModelType:
        """Type of input this model takes."""
        return ModelType.NONE

    @classmethod
    def input_names(cls: type[CustomU8]) -> list[tuple[str, InputType]]:
        """Names of the input tensors."""
        return [("input", InputType.U8)]

    @classmethod
    def output_names(cls: type[CustomU8]) -> list[str]:
        """Names of the output tensors."""
        return ["output"]

    def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # TODO: Fill in with custom functionality and compile
        # When compiling with the version of openvino used by default for U8 inputs
        # the network must not be an identity network (i.e. it must do something to the input)
        return image + 1


def main() -> None:
    """Test the custom operations."""
    set_log_level("DEBUG")
    fp16_model_path = compile_model(
        CustomFP16,
        {},  # Add any arguments here as a dictionary
        cache=False,
        shape_mapping={
            InputType.FP16: (*FP16_IMAGE_SIZE, 3),
        },
        shaves=SHAVES,
        verbose=True,
    )
    u8_model_path = compile_model(
        CustomU8,
        {},  # Add any arguments here as a dictionary
        cache=False,
        shape_mapping={
            InputType.U8: (*U8_IMAGE_SIZE, 1),
        },
        shaves=SHAVES,
        verbose=True,
    )
    pipeline = dai.Pipeline()
    cam = create_color_camera(
        pipeline,
        fps=15,
        preview_size=FP16_IMAGE_SIZE,
    )
    stereo, left, right = create_stereo_depth(
        pipeline,
        fps=15,
        resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P,
    )
    fp16_network = create_neural_network(
        pipeline,
        cam.preview,
        fp16_model_path,
    )
    u8_network = create_neural_network(
        pipeline,
        stereo.depth,
        u8_model_path,
    )
    streamname_fp16 = "network_fp16"
    passname_fp16 = "passthrough_fp16"
    streamname_u8 = "network_u8"
    passname_u8 = "passthrough_u8"
    xout_fp16_nn = create_xout(pipeline, fp16_network.out, streamname_fp16)
    xout_fp16_pass = create_xout(pipeline, fp16_network.passthrough, passname_fp16)
    xout_u8_nn = create_xout(pipeline, u8_network.out, streamname_u8)
    xout_u8_pass = create_xout(pipeline, u8_network.passthrough, passname_u8)
    all_nodes = [
        cam,
        stereo,
        left,
        right,
        fp16_network,
        u8_network,
        xout_fp16_nn,
        xout_fp16_pass,
        xout_u8_nn,
        xout_u8_pass,
    ]
    print(f"Created pipeline with {len(all_nodes)} nodes")
    fps_buffer = deque(maxlen=60)
    with dai.Device(pipeline) as device:
        device.setLogLevel(dai.LogLevel.DEBUG)
        device.setLogOutputLevel(dai.LogLevel.DEBUG)
        fp16_queue: dai.DataOutputQueue = device.getOutputQueue(streamname_fp16)
        pass_fp16_queue: dai.DataOutputQueue = device.getOutputQueue(passname_fp16)
        u8_queue: dai.DataOutputQueue = device.getOutputQueue(streamname_u8)
        pass_u8_queue: dai.DataOutputQueue = device.getOutputQueue(passname_u8)
        t0 = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            fp16_data = fp16_queue.get()
            fp16_passdata = pass_fp16_queue.get()
            u8_data = u8_queue.get()
            u8_passdata = pass_u8_queue.get()
            fp16_frame = get_nn_frame(
                fp16_data,
                channels=3,
                frame_size=FP16_IMAGE_SIZE,
            )
            u8_frame = get_nn_frame(
                u8_data,
                channels=1,
                frame_size=U8_IMAGE_SIZE,
            )
            passimage_fp16: np.ndarray = fp16_passdata.getCvFrame()
            passimage_u8: np.ndarray = u8_passdata.getCvFrame()
            t1 = time.perf_counter()
            fps_buffer.append(1 / (t1 - t0))
            t0 = t1
            for frame, passimage, streamname in zip(
                [fp16_frame, u8_frame],
                [passimage_fp16, passimage_u8],
                [streamname_fp16, streamname_u8],
            ):
                cv2.putText(
                    frame,
                    f"FPS: {np.mean(fps_buffer):.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                pshape = passimage.shape
                resized_frame = cv2.resize(frame, (pshape[1], pshape[0]))
                sidebyside = np.hstack((resized_frame, passimage))
                cv2.imshow(streamname, sidebyside)
            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
