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
import depthai as dai

from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, create_xout, get_nn_point_cloud_buffer
from oakutils.nodes.models.point_cloud import create_point_cloud, create_xyz_matrix
from oakutils.point_clouds import (
    PointCloudVisualizer,
    get_point_cloud_from_np_buffer,
    filter_point_cloud,
)

pipeline = dai.Pipeline()
pcv = PointCloudVisualizer(use_threading=False)

# get the calibration
calibration = get_camera_calibration(
    rgb_size=(1920, 1080),
    mono_size=(640, 400),
    is_primary_mono_left=True,  # make sure to set primary to same as align_socket
)

# create the color camera node
out_nodes = create_stereo_depth(
    pipeline,
    align_socket=dai.CameraBoardSocket.LEFT,  # make sure this is same as primary for calibration
)
depth = out_nodes[0]  # the first node from create_stero_depth is the depth node

pcl, xin_pcl, start_pcl = create_point_cloud(
    pipeline,
    depth_link=depth.depth,
    calibration=calibration,
)
xout_pcl = create_xout(pipeline, pcl.out, "pcl")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("pcl")

    start_pcl(device)

    while True:
        data = queue.get()

        counter += 1  # visualizer is super slow
        if counter == 3:
            np_pcl = get_nn_point_cloud_buffer(data)
            pcl = get_point_cloud_from_np_buffer(np_pcl)
            pcl = filter_point_cloud(pcl)
            pcv.update(pcl)
            counter = 0
