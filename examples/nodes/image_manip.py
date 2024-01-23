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
import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, create_image_manip, create_xout

pipeline = dai.Pipeline()

# create the color camera
cam = create_color_camera(pipeline)
xout_cam = create_xout(pipeline, cam.video, "rgb")

# create the image manip node
manip = create_image_manip(
    pipeline=pipeline,
    input_link=cam.preview,
    frame_type=dai.RawImgFrame.Type.GRAY8,
)
xout_manip = create_xout(pipeline, manip.out, "gray")

with dai.Device(pipeline) as device:
    rgb_queue: dai.DataOutputQueue = device.getOutputQueue("rgb")
    queue: dai.DataOutputQueue = device.getOutputQueue("gray")

    while True:
        rgb_data = rgb_queue.get()
        cv2.imshow("rgb", rgb_data.getCvFrame())

        lp_data = queue.get()
        frame = lp_data.getCvFrame()

        cv2.imshow("gray frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
