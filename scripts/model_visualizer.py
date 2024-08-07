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

import argparse
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
from oakutils.nodes import models
from typing_extensions import Self


def main() -> None:
    """Test the custom operations."""
    set_log_level("WARNING")
    parser = argparse.ArgumentParser("Visualize the output of custom models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "gaussian",
            "laplacian",
            "sobel",
            "gftt",
            "harris",
            "hessian",
        ],
        help="The model you want to visualize output for.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=7,
        help="The kernel size to use for the model. Laplacian and gaussian use this.",
    )
    parser.add_argument(
        "--shaves",
        type=int,
        default=6,
        help="The number of shaves to use for the model.",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Use a blur operation in the pipeline.",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=3,
        help="The kernel size to use for the blur operation (on models which utilize this).",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Output a grayscale image.",
    )
    args = parser.parse_args()

    # create the base pipeline
    pipeline = dai.Pipeline()
    cam = create_color_camera(pipeline)

    # create the custom operation according to the model type
    if args.model == "gaussian":
        model = models.create_gaussian(pipeline, cam.preview, kernel_size=args.kernel_size, grayscale_out=args.grayscale)
    elif args.model == "laplacian":
        model = models.create_laplacian(pipeline, cam.preview, kernel_size=args.kernel_size, blur_kernel_size=args.blur_kernel_size, use_blur=args.blur, grayscale_out=args.grayscale)
    elif args.model == "sobel":
        model = models.create_sobel(pipeline, cam.preview, blur_kernel_size=args.blur_kernel_size, use_blur=args.blur, grayscale_out=args.grayscale)
    elif args.model == "gftt":
        model = models.create_gftt(pipeline, cam.preview, blur_kernel_size=args.blur_kernel_size, use_blur=args.blur, grayscale_out=args.grayscale)
    elif args.model == "harris":
        model = models.create_harris(pipeline, cam.preview, blur_kernel_size=args.blur_kernel_size, use_blur=args.blur, grayscale_out=args.grayscale)
    elif args.model == "hessian":
        model = models.create_hessian(pipeline, cam.preview, blur_kernel_size=args.blur_kernel_size, use_blur=args.blur, grayscale_out=args.grayscale)
    else:
        err_msg = f"Unknown model type: {args.model}"
        raise ValueError(err_msg)

    streamname_fp16 = "network_fp16"
    passname_fp16 = "passthrough_fp16"
    xout_nn = create_xout(pipeline, model.out, streamname_fp16)
    xout_pass = create_xout(pipeline, model.passthrough, passname_fp16)
    all_nodes = [
        cam,
        model,
        xout_nn,
        xout_pass,
    ]
    print(f"Created pipeline with {len(all_nodes)} nodes")
    fps_buffer = deque(maxlen=60)
    channels = 1 if args.grayscale else 3
    with dai.Device(pipeline) as device:
        device.setLogLevel(dai.LogLevel.DEBUG)
        device.setLogOutputLevel(dai.LogLevel.DEBUG)
        fp16_queue: dai.DataOutputQueue = device.getOutputQueue(streamname_fp16)
        pass_fp16_queue: dai.DataOutputQueue = device.getOutputQueue(passname_fp16)
        t0 = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            fp16_data = fp16_queue.get()
            fp16_passdata = pass_fp16_queue.get()
            fp16_frame = get_nn_frame(
                fp16_data,
                channels=channels,
            )
            if channels == 1:
                fp16_frame = cv2.cvtColor(fp16_frame, cv2.COLOR_GRAY2BGR)
            passimage_fp16: np.ndarray = fp16_passdata.getCvFrame()
            t1 = time.perf_counter()
            fps_buffer.append(1 / (t1 - t0))
            t0 = t1
            for frame, passimage, streamname in zip(
                [fp16_frame],
                [passimage_fp16],
                [streamname_fp16],
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
                cv2.putText(
                    frame,
                    f"{args.model}",
                    (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                )
                cv2.putText(
                    passimage,
                    "Input",
                    (700, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                )
                pshape = passimage.shape
                resized_frame = cv2.resize(frame, (pshape[1], pshape[0]))
                sidebyside = np.hstack((resized_frame, passimage))
                cv2.imshow(streamname, sidebyside)
            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
