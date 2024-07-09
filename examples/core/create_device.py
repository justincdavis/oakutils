# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing getting the creation of a device."""

from __future__ import annotations

import depthai as dai
from oakutils import create_device

pipeline = dai.Pipeline()

# basic creation
device1 = create_device(pipeline)

# using an ip address, mxid, or usb ip
# device2 = create_device(pipeline, "192.168.1.44")
device3 = create_device(pipeline, "14442C108144F1D000")
# device4 = create_device(pipeline, "3.3.3")

# adding the usb speed parameters
# device5 = create_device(pipeline, max_usb_speed=dai.UsbSpeed.SUPER_PLUS)
device6 = create_device(pipeline, "3.3.3", dai.UsbSpeed.SUPER_PLUS)
