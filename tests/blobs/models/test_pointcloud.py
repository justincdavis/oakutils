# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import depthai as dai
from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, create_xout
from oakutils.nodes.models import create_point_cloud, get_point_cloud_buffer

from ...device import get_device_count


def test_create_and_run() -> None:
    if get_device_count() == 0:
        return
    calib_data = get_camera_calibration()
    for shave in [1, 2, 3, 4, 5, 6]:
        pipeline = dai.Pipeline()
        stereo, left, right = create_stereo_depth(pipeline)
        pcl, xin_pcl, device_call = create_point_cloud(
            pipeline, stereo.depth, calib_data, shaves=shave
        )
        xout_pcl = create_xout(pipeline, pcl.out, "pcl_out")

        all_nodes = [
            stereo,
            left,
            right,
            pcl,
            xin_pcl,
            xout_pcl,
        ]
        assert len(all_nodes) == 6
        for node in all_nodes:
            assert node is not None

        with dai.Device(pipeline) as device:
            device_call(device)
            queue: dai.DataOutputQueue = device.getOutputQueue("pcl_out")

            while True:
                data = queue.get()
                pcl_buffer = get_point_cloud_buffer(data)
                assert pcl_buffer is not None
                break
