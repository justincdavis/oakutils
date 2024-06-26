.. _examples_nodes/rgb_depth:

Example: nodes/rgb_depth.py
===========================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to start rgb and depth streams."""
	
	from __future__ import annotations
	
	import cv2
	import depthai as dai
	from oakutils.calibration import get_camera_calibration
	from oakutils.nodes import create_color_camera, create_stereo_depth, create_xout
	
	pipeline = dai.Pipeline()
	calibration = get_camera_calibration(
	    rgb_size=(1920, 1080),
	    mono_size=(640, 400),
	    is_primary_mono_left=True,  # make sure to set primary to same as align_socket
	)
	
	# create the color camera node
	cam = create_color_camera(pipeline, preview_size=(640, 480))
	stereo, left, right = create_stereo_depth(pipeline)
	
	xout_rgb = create_xout(pipeline, cam.video, "rgb")
	xout_depth = create_xout(pipeline, stereo.depth, "depth")
	
	with dai.Device(pipeline) as device:
	    device.setLogLevel(dai.LogLevel.DEBUG)
	    device.setLogOutputLevel(dai.LogLevel.DEBUG)
	
	    rgb_q: dai.DataOutputQueue = device.getOutputQueue("rgb")
	    depth_q: dai.DataOutputQueue = device.getOutputQueue("depth")
	
	    while True:
	        in_rgb = rgb_q.get()
	        in_depth = depth_q.get()
	        rgb_frame = in_rgb.getCvFrame()
	        depth_frame = in_depth.getFrame()
	
	        cv2.imshow("rgb", rgb_frame)
	        cv2.imshow("depth", depth_frame)
	        if cv2.waitKey(1) == ord("q"):
	            break

