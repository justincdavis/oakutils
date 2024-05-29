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
from __future__ import annotations

import time

import depthai as dai

from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, create_xout, get_nn_point_cloud_buffer
from oakutils.nodes.models import create_point_cloud

from .utils import eval_model
from ...helpers import check_device, TIME_TO_RUN


def check_pointcloud(shaves: int):
    """Test the pointcloud node"""
    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    
    pipeline = dai.Pipeline()

    calibration = get_camera_calibration(
        (1920, 1080),
        (640, 400),
    )
    stereo, left, right = create_stereo_depth(pipeline)
    pcl, xin_xyz, start_pcl = create_point_cloud(
        pipeline, 
        stereo.depth,
        calibration,
        shaves=shaves,
    )
    _ = create_xout(pipeline, pcl.out, "pcl")

    with dai.Device(pipeline) as device:
        start_pcl(device)
        l_queue: dai.DataOutputQueue = device.getOutputQueue("pcl")

        t0 = time.perf_counter()
        while True:
            l_data = l_queue.get()
            pcl = get_nn_point_cloud_buffer(l_data)
            if time.perf_counter() - t0 > TIME_TO_RUN:
                break
    return 0

def test_pointcloud_1_shave():
    check_device(lambda: check_pointcloud(1), TIME_TO_RUN)

def test_pointcloud_2_shave():
    check_device(lambda: check_pointcloud(2), TIME_TO_RUN)

def test_pointcloud_3_shave():
    check_device(lambda: check_pointcloud(3), TIME_TO_RUN)

def test_pointcloud_4_shave():
    check_device(lambda: check_pointcloud(4), TIME_TO_RUN)

def test_pointcloud_5_shave():
    check_device(lambda: check_pointcloud(5), TIME_TO_RUN)

def test_pointcloud_6_shave():
    check_device(lambda: check_pointcloud(6), TIME_TO_RUN)

def test_results():
    eval_model("pointcloud", (640, 400, 1))
