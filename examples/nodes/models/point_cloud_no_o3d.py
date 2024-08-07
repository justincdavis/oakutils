# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use the onboard point cloud model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import depthai as dai
from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, create_xout, get_nn_point_cloud_buffer
from oakutils.nodes.models import create_point_cloud

if TYPE_CHECKING:
    import numpy as np
    import open3d as o3d

pipeline = dai.Pipeline()

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
        np_pcl: np.ndarray = get_nn_point_cloud_buffer(data)
        print(np_pcl.shape)
