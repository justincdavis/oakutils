.. _examples_nodes/models/hessian:

Example: nodes/models/hessian.py
================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the HarrisCornerDetector model."""
	from __future__ import annotations
	
	import cv2
	import depthai as dai
	
	from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
	from oakutils.nodes.models import create_hessian
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	cam = create_color_camera(
	    pipeline,
	    fps=30,
	    preview_size=(640, 480),
	)  # set the preview size to the input of the nn
	
	hessian = create_hessian(
	    pipeline,
	    input_link=cam.preview,
	    shaves=1,
	)
	xout_hessian = create_xout(pipeline, hessian.out, "hessian")
	
	with dai.Device(pipeline) as device:
	    queue: dai.DataOutputQueue = device.getOutputQueue("hessian")
	
	    while True:
	        data = queue.get()
	        frame = get_nn_bgr_frame(data, normalization=255.0)
	
	        cv2.imshow("hessian frame", frame)
	        if cv2.waitKey(1) == ord("q"):
	            break

