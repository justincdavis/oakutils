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
"""Example showcasing how to make a mono_camera node."""
from __future__ import annotations

import cv2
import depthai as dai

from oakutils.nodes import create_mono_camera, create_xout

pipeline = dai.Pipeline()

# create the color camera node
left = create_mono_camera(pipeline, dai.CameraBoardSocket.LEFT)
right = create_mono_camera(pipeline, dai.CameraBoardSocket.RIGHT)
xout_left = create_xout(pipeline, left.out, "left")
xout_right = create_xout(pipeline, right.out, "right")

with dai.Device(pipeline) as device:
    lq: dai.DataOutputQueue = device.getOutputQueue("left")
    rq: dai.DataOutputQueue = device.getOutputQueue("right")

    while True:
        left = lq.get()
        right = rq.get()
        cv2.imshow("left", left.getCvFrame())
        cv2.imshow("right", right.getCvFrame())
        if cv2.waitKey(1) == ord("q"):
            break
