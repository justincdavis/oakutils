.. _examples_nodes/models/point_cloud:

Example: nodes/models/point_cloud.py
====================================

.. code-block:: python

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
	
	# Both get_nn_point_cloud_buffer and get_point_cloud_buffer are the same
	# from oakutils.nodes.models import get_point_cloud_buffer  # optionally use this to get the point cloud buffer
	from oakutils.point_clouds import (
	    PointCloudVisualizer,
	    filter_point_cloud,
	    get_point_cloud_from_np_buffer,
	)
	
	if TYPE_CHECKING:
	    import numpy as np
	    import open3d as o3d
	
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
	
	    counter = 0
	    update_rate = 3
	    while True:
	        data = queue.get()
	
	        counter += 1  # visualizer is super slow
	        if counter == update_rate:
	            np_pcl: np.ndarray = get_nn_point_cloud_buffer(data)
	            # after getting buffer, all point clouds after are Open3d
	            # if not using Open3d, then the NumPy buffer can be used
	            pcl: o3d.geometry.PointCloud = get_point_cloud_from_np_buffer(np_pcl)
	            pcl = filter_point_cloud(pcl)
	            pcv.update(pcl)
	            counter = 0

