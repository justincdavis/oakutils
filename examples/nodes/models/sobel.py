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

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
from oakutils.nodes.models import create_sobel

pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(
    pipeline, preview_size=(640, 480)
)  # set the preview size to the input of the nn

sobel = create_sobel(
    pipeline,
    input_link=cam.preview,
    shaves=1,
)
xout_sobel = create_xout(pipeline, sobel.out, "sobel")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("sobel")

    while True:
        data = queue.get()
        frame = get_nn_bgr_frame(data, normalization=255.0)

        cv2.imshow("sobel frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
