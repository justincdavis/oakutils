.. _examples_api_camera:

Example: api_camera.py
======================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""File showcasing an example usecase of the ApiCamera class."""
	
	from __future__ import annotations
	
	from typing import TYPE_CHECKING
	
	from oakutils import ApiCamera
	from oakutils.nodes import (
	    create_color_camera,
	    create_stereo_depth,
	    create_xout,
	    get_nn_point_cloud_buffer,
	)
	from oakutils.nodes.models import create_point_cloud
	from oakutils.point_clouds import get_point_cloud_from_np_buffer
	from oakutils.tools.parsing import (
	    get_color_sensor_resolution_from_tuple,
	    get_mono_sensor_resolution_from_tuple,
	)
	
	if TYPE_CHECKING:
	    import depthai as dai
	
	rgb_resolution = (1920, 1080)
	mono_resolution = (640, 400)
	use_left_mono = True
	print("Starting API Camera")
	oak = ApiCamera(
	    primary_mono_left=use_left_mono,
	    color_size=rgb_resolution,
	    mono_size=mono_resolution,
	)
	print("Camera Initialized")
	
	cam = create_color_camera(
	    oak.pipeline,
	    fps=15,
	    resolution=get_color_sensor_resolution_from_tuple(rgb_resolution),
	)
	xout_cam = create_xout(oak.pipeline, cam.video, "color")
	stereo, left, right = create_stereo_depth(
	    oak.pipeline,
	    resolution=get_mono_sensor_resolution_from_tuple(mono_resolution),
	)
	point_cloud, xin_xyz, start_pcl = create_point_cloud(
	    oak.pipeline,
	    stereo.depth,
	    oak.calibration,
	)
	xout_point_cloud = create_xout(oak.pipeline, point_cloud.out, "point_cloud")
	
	# add the basic display
	oak.add_display("color")
	
	
	# adding outputs from onboard neural networks is easy, but requires the correct calls
	def pcl_callback(pcl: dai.NNData) -> None:
	    """Use as callback for processing the pointcloud. Need a callback for api cam processing."""
	    pcl = get_nn_point_cloud_buffer(pcl)
	    pcl = get_point_cloud_from_np_buffer(pcl)
	    oak.pcv.update(pcl)
	
	
	oak.add_callback("point_cloud", pcl_callback)
	
	# # alternatively, you can do it with partials
	# # shown below will be commented out since both calls do the same thing
	# # the second approach will work for when the function is predefined
	# # such as imported from another library or created elsewhere
	# def pcl_callback_2(viz, pcl):
	#     pcl = get_nn_point_cloud(pcl)
	#     pcl = create_point_cloud_from_np(pcl)
	#     viz.update(pcl)
	# oak.add_callback("point_cloud", partial(pcl_callback_2, oak.pcv))
	
	oak.add_device_call(start_pcl)  # start_pcl takes only an dai.Device, queue as such
	
	oak.start(blocking=True)

