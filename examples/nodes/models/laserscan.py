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
"""Example showcasing how to use the Sobel model."""
from __future__ import annotations

import depthai as dai

from oakutils.nodes import create_stereo_depth, create_xout, get_nn_data
from oakutils.nodes.models import create_laserscan

pipeline = dai.Pipeline()

# create the color camera node
stereo, left, right = create_stereo_depth(
    pipeline,
    preview_size=(640, 480),
)  # set the preview size to the input of the nn

laser = create_laserscan(
    pipeline,
    input_link=stereo.depth,
    width=10,
    shaves=1,
)
xout_sobel = create_xout(pipeline, laser.out, "sobel")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("sobel")

    while True:
        data = queue.get()
        scan = get_nn_data(data, reshape_to=(400))

        print(f"Scan shape: {scan.shape}, max: {scan.max()}, min: {scan.min()}")
