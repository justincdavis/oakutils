.. _examples_nodes/image_manip:

Example: nodes/image_manip.py
=============================

.. code-block:: python

	import cv2
	import depthai as dai
	
	from oakutils.nodes import create_color_camera, create_image_manip, create_xout
	
	pipeline = dai.Pipeline()
	
	# create the color camera
	cam = create_color_camera(pipeline)
	xout_cam = create_xout(pipeline, cam.video, "rgb")
	
	# create the image manip node
	manip = create_image_manip(
	    pipeline=pipeline,
	    input_link=cam.preview,
	    frame_type=dai.RawImgFrame.Type.GRAY8,
	)
	xout_manip = create_xout(pipeline, manip.out, "gray")
	
	with dai.Device(pipeline) as device:
	    rgb_queue: dai.DataOutputQueue = device.getOutputQueue("rgb")
	    queue: dai.DataOutputQueue = device.getOutputQueue("gray")
	
	    while True:
	        rgb_data = rgb_queue.get()
	        cv2.imshow("rgb", rgb_data.getCvFrame())
	
	        lp_data = queue.get()
	        frame = lp_data.getCvFrame()
	
	        cv2.imshow("gray frame", frame)
	        if cv2.waitKey(1) == ord("q"):
	            break

