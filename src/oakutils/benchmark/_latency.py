# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from threading import Thread

import depthai as dai
import numpy as np

from oakutils.core import create_device


def measure_latency(
    device_id: str | None = None,
    iterations: int = 100,
) -> tuple[float, float, list[float]]:
    """
    Measure the latency of the DepthAI device.

    Parameters
    ----------
    device_id : str, optional
        The id of the device to use, by default None
        This can be a MXID, IP address, or USB port name.
        Examples: "14442C108144F1D000", "
    iterations : int, optional
        The number of iterations to run, by default 100

    Returns
    -------
    tuple[float, float, list[float]]
        The average latency, the standard deviation of latency, and the raw data.

    Notes
    -----
    Modified from script at:
    https://github.com/luxonis/depthai-experiments/blob/master/random-scripts/oak_latency_test.py

    """

    def _send_buff(queue: dai.DataInputQueue, ts: list[float]) -> None:
        for _ in range(iterations):
            buffer = dai.Buffer()
            buffer.setData([1])
            queue.send(buffer)
            ts.append(time.time())
            time.sleep(0.2)

    pipeline = dai.Pipeline()

    # create input buffer
    xin = pipeline.create(dai.node.XLinkIn)
    xin.setMaxDataSize(10)
    xin.setNumFrames(1)
    xin.setStreamName("xin")

    # create output buffer
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("xout")
    xout.input.setBlocking(True)
    xout.input.setQueueSize(1)

    # link
    xin.out.link(xout.input)

    timestamps: list[float] = []

    with create_device(pipeline, device_id) as device:
        qin = device.getInputQueue("xin", 1, blocking=True)  # type: ignore[attr-defined]
        qout = device.getOutputQueue("xout", 1, blocking=True)  # type: ignore[attr-defined]

        thread = Thread(target=_send_buff, args=(qin, timestamps), daemon=True)
        thread.start()

        latencies = []
        for i in range(iterations):
            qout.get()
            latency = time.time() - timestamps[i]

            # skip first buffer
            if i == 0:
                continue

            latencies.append(latency * 1000)

        thread.join()

    return float(np.mean(latencies)), float(np.std(latencies)), latencies
