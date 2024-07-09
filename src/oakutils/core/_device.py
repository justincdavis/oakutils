# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import depthai as dai


def create_device(
    pipeline: dai.Pipeline,
    device_id: str | None = None,
    max_usb_speed: dai.UsbSpeed | None = None,
) -> dai.DeviceBase:
    """
    Create a DepthAI device object from a pipeline.

    Parameters
    ----------
    pipeline : dai.Pipeline
        The pipeline to use
    device_id : str, optional
        The id of the device to use, by default None
        This can be a MXID, IP address, or USB port name.
        Examples: "14442C108144F1D000", "192.168.1.44", "3.3.3"
    max_usb_speed : dai.UsbSpeed, optional
        The maximum USB speed to use, by default None
        Options are available in dai.UsbSpeed

    Returns
    -------
    dai.Device
        The DepthAI device object

    """
    if device_id is not None and max_usb_speed is not None:
        device_info = dai.DeviceInfo(device_id)
        return dai.Device(
            pipeline=pipeline,
            deviceInfo=device_info,
            maxUsbSpeed=max_usb_speed,
        )
    if device_id is not None:
        device_info = dai.DeviceInfo(device_id)
        return dai.Device(pipeline=pipeline, devInfo=device_info)
    if max_usb_speed is not None:
        return dai.Device(pipeline=pipeline, maxUsbSpeed=max_usb_speed)
    return dai.Device(pipeline)
