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
"""Example showcasing how to start rgb and depth streams."""
from __future__ import annotations

import cv2
import depthai as dai

from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_color_camera, create_stereo_depth, create_xout


pipeline = dai.Pipeline()
calibration = get_camera_calibration(
    rgb_size=(1920, 1080),
    mono_size=(640, 400),
    is_primary_mono_left=True,  # make sure to set primary to same as align_socket
)

# create the color camera node
cam = create_color_camera(pipeline, preview_size=(640, 480))
stereo, left, right = create_stereo_depth(pipeline)

xout_rgb = create_xout(pipeline, cam.video, "rgb")
xout_depth = create_xout(pipeline, stereo.depth, "depth")

with dai.Device(pipeline) as device:
    device.setLogLevel(dai.LogLevel.DEBUG)
    device.setLogOutputLevel(dai.LogLevel.DEBUG)
    
    rgb_q: dai.DataOutputQueue = device.getOutputQueue("rgb")
    depth_q: dai.DataOutputQueue = device.getOutputQueue("depth")

    while True:
        in_rgb = rgb_q.get()
        in_depth = depth_q.get()
        rgb_frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        cv2.imshow("rgb", rgb_frame)
        cv2.imshow("depth", depth_frame)
        if cv2.waitKey(1) == ord("q"):
            break
