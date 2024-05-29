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

from oakutils.nodes import create_stereo_depth, create_xout
from oakutils.nodes.models import create_laserscan, get_laserscan

from .utils import eval_model
from ...helpers import check_device, TIME_TO_RUN


def check_laserscan(shaves: int, scans: int):
    """Test the pointcloud node"""
    if len(dai.Device.getAllAvailableDevices()) == 0:
        return 0  # no device found
    
    pipeline = dai.Pipeline()

    stereo, left, right = create_stereo_depth(pipeline)
    laser = create_laserscan(pipeline, stereo.depth, shaves=shaves, scans=scans)
    xout = create_xout(pipeline, laser.out, "laser")

    with dai.Device(pipeline) as device:
        l_queue: dai.DataOutputQueue = device.getOutputQueue("laser")

        t0 = time.perf_counter()
        while True:
            l_data = l_queue.get()
            scan = get_laserscan(l_data)
            if time.perf_counter() - t0 > TIME_TO_RUN:
                break
    return 0

def test_laserscan_1_shave_1_scan():
    check_device(lambda: check_laserscan(1, 1), TIME_TO_RUN)

def test_laserscan_1_shave_3_scans():
    check_device(lambda: check_laserscan(1, 3), TIME_TO_RUN)

def test_laserscan_1_shave_5_scans():
    check_device(lambda: check_laserscan(1, 5), TIME_TO_RUN)

def test_laserscan_2_shaves_1_scan():
    check_device(lambda: check_laserscan(2, 1), TIME_TO_RUN)

def test_laserscan_3_shaves_1_scan():
    check_device(lambda: check_laserscan(3, 1), TIME_TO_RUN)

def test_laserscan_4_shaves_1_scans():
    check_device(lambda: check_laserscan(4, 1), TIME_TO_RUN)

def test_laserscan_5_shaves_1_scans():
    check_device(lambda: check_laserscan(5, 1), TIME_TO_RUN)

def test_laserscan_6_shaves_1_scans():
    check_device(lambda: check_laserscan(6, 1), TIME_TO_RUN)

def test_results():
    eval_model("laserscan", (640, 400, 1))
