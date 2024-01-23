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
"""Example showcasing how to use the ArucoFinder."""
import cv2
import depthai as dai

from oakutils.aruco import ArucoFinder
from oakutils.calibration import get_camera_calibration_basic
from oakutils.nodes import create_color_camera, create_xout

calibration = get_camera_calibration_basic()
finder = ArucoFinder(cv2.aruco.DICT_4X4_100, 0.05, calibration.rgb)

pipeline = dai.Pipeline()
cam = create_color_camera(pipeline)
xout_cam = create_xout(pipeline, cam.video, "rgb")

with dai.Device(pipeline) as device:
    cam_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = cam_queue.get()
        frame = in_rgb.getCvFrame()
        markers = finder.find(frame)
        for marker in markers:
            print(marker)
        cv2.imshow("frame", finder.draw(frame, markers))
        if cv2.waitKey(1) == ord("q"):
            break
