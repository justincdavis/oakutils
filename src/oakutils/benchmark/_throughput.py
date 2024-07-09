# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time

import depthai as dai
import numpy as np

from oakutils.core import create_device


def measure_throughput(
    device_id: str | None = None,
    max_usb_speed: dai.UsbSpeed | None = None,
    iterations: int = 50,
    discard: int = 10,
    size: int = 10_000_000,
) -> tuple[float, float]:
    """
    Measure the throughput of the DepthAI device.

    Parameters
    ----------
    device_id : str, optional
        The id of the device to use, by default None
        This can be a MXID, IP address, or USB port name.
        Examples: "14442C108144F1D000", "
    max_usb_speed : dai.UsbSpeed, optional
        The maximum USB speed to use, by default None
        Options are available in dai.UsbSpeed
    iterations : int, optional
        The number of iterations to run, by default 50
    discard : int, optional
        The number of iterations to discard for warmup,
        by default 10
    size : int, optional
        The size of the data to send,
        by default 10_000_000 or 20MB

    Returns
    -------
    tuple[float, float]
        The downlink and uplink throughput in MB/s.

    Notes
    -----
    Modified from script at:
    https://github.com/luxonis/depthai-experiments/blob/master/random-scripts/oak_bandwidth_test.py

    """
    pipeline = dai.Pipeline()

    script = pipeline.create(dai.node.Script)
    script.setScript(f"""
    import time

    # Measure downlink first
    sent_ts = []
    buff = Buffer({size})
    for i in range({iterations}):
        node.io['xout'].send(buff)
        sent_ts.append(time.time())
        if i == {discard - 1}:
            # node.warn('{discard - 1}th buffer sent at' + str(time.time()))
            pass
        # node.warn('Sent buffer ' + str(i))
    # node.warn('{iterations}th buffer sent at' + str(time.time()))
    total_time = sent_ts[-1] - sent_ts[{discard - 1}]
    total_bits = ({iterations - discard}) * {size} * 8
    downlink = total_bits / total_time
    downlink_mbps = downlink / (1000 * 1000)
    # node.warn('Downlink ' + str(downlink_mbps) + ' mbps')

    # Measure uplink
    receive_ts = []
    for i in range({iterations}):
        node.io['xin'].get()
        receive_ts.append(time.time())
        if i == {discard - 1}:
            # node.warn('{discard - 1}th buffer received at' + str(time.time()))
            pass
        # node.warn('Received buffer ' + str(i))
    # node.warn('{iterations}th buffer received at' + str(time.time()))

    total_time = receive_ts[-1] - receive_ts[{discard - 1}]
    total_bits = ({iterations - discard}) * {size} * 8
    uplink = total_bits / total_time
    uplink_mbps = uplink / (1000 * 1000)
    # node.warn('Uplink ' + str(uplink_mbps) + ' mbps')
    """)

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setNumFrames(2)
    xin.setMaxDataSize(size * 2)
    xin.setStreamName("xin")
    xin.out.link(script.inputs["xin"])

    xout = pipeline.create(dai.node.XLinkOut)
    xout.input.setBlocking(True)
    xout.input.setQueueSize(2)
    xout.setStreamName("xout")
    script.outputs["xout"].link(xout.input)

    downlink_mbps = 0.0
    uplink_mbps = 0.0

    with create_device(pipeline, device_id, max_usb_speed) as device:
        qin = device.getInputQueue("xin", 2, blocking=True)  # type: ignore[attr-defined]
        qout = device.getOutputQueue("xout", 2, blocking=True)  # type: ignore[attr-defined]

        # Downlink
        receive_ts = []
        for i in range(iterations):
            qout.get()
            receive_ts.append(time.time())
            if i == discard - 1:
                pass
        total_time = receive_ts[-1] - receive_ts[discard - 1]
        total_bits = (iterations - discard) * size * 8
        downlink = total_bits / total_time
        downlink_mbps = downlink / (1000 * 1000)

        buffer = dai.Buffer()
        buffer.setData(np.zeros(size, dtype=np.uint8))
        sent_ts = []
        for i in range(iterations):
            qin.send(buffer)
            sent_ts.append(time.time())
            if i == discard - 1:
                pass
        total_time = sent_ts[-1] - sent_ts[discard - 1]
        total_bits = (iterations - discard) * size * 8
        uplink = total_bits / total_time
        uplink_mbps = uplink / (1000 * 1000)

    return downlink_mbps, uplink_mbps
