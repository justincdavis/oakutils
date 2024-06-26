.. _examples_nodes/stereo_depth:

Example: nodes/stereo_depth.py
==============================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to make a stereo_depth node."""
	from __future__ import annotations
	
	import cv2
	import depthai as dai
	
	from oakutils.nodes import create_stereo_depth, create_xout
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	stereo, left_cam, right_cam = create_stereo_depth(pipeline)
	_ = create_xout(pipeline, stereo.depth, "depth")
	_ = create_xout(pipeline, stereo.disparity, "disparity")
	
	with dai.Device(pipeline) as device:
	    depthq: dai.DataOutputQueue = device.getOutputQueue("depth")
	    disparityq: dai.DataOutputQueue = device.getOutputQueue("disparity")
	
	    while True:
	        depth = depthq.get()
	        disparity = disparityq.get()
	        cv2.imshow("depth", depth.getFrame())
	        cv2.imshow("disparity", disparity.getFrame())
	        if cv2.waitKey(1) == ord("q"):
	            break

