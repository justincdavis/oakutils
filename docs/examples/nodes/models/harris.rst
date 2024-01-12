.. _examples_nodes/models/harris:

Example: nodes/models/harris.py
===============================

.. code-block:: python

	import cv2
	import depthai as dai
	
	from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
	from oakutils.nodes.models import create_harris
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	cam = create_color_camera(
	    pipeline, 
	    fps=15, # some "advanced" kornia based models require lower fps
	    preview_size=(640, 480),
	)  # set the preview size to the input of the nn
	
	harris = create_harris(
	    pipeline,
	    input_link=cam.preview,
	    shaves=1,
	)
	xout_harris = create_xout(pipeline, harris.out, "harris")
	
	with dai.Device(pipeline) as device:
	    queue: dai.DataOutputQueue = device.getOutputQueue("harris")
	
	    while True:
	        data = queue.get()
	        frame = get_nn_bgr_frame(data, normalization=255.0)
	
	        cv2.imshow("harris frame", frame)
	        if cv2.waitKey(1) == ord("q"):
	            break

