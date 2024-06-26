.. _examples_nodes/models/laserscan:

Example: nodes/models/laserscan.py
==================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example showcasing how to use the Sobel model."""
	from __future__ import annotations
	
	import depthai as dai
	
	from oakutils.nodes import create_stereo_depth, create_xout
	from oakutils.nodes.models import create_laserscan, get_laserscan
	
	pipeline = dai.Pipeline()
	
	# create the stereo node
	stereo, left, right = create_stereo_depth(
	    pipeline,
	)
	
	laser = create_laserscan(
	    pipeline,
	    input_link=stereo.depth,
	    width=10,
	    scans=1,
	    shaves=1,
	)
	xout_sobel = create_xout(pipeline, laser.out, "sobel")
	
	with dai.Device(pipeline) as device:
	    queue: dai.DataOutputQueue = device.getOutputQueue("sobel")
	
	    while True:
	        data = queue.get()
	        scan = get_laserscan(data)
	
	        print(f"Scan shape: {scan.shape}, max: {scan.max()}, min: {scan.min()}")

