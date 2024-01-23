.. _examples_nodes/neural_network:

Example: nodes/neural_network.py
================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	# This program is free software: you can redistribute it and/or modify
	# it under the terms of the GNU General Public License as published by
	# the Free Software Foundation, either version 3 of the License, or
	# (at your option) any later version.
	#
	# This program is distributed in the hope that it will be useful,
	# but WITHOUT ANY WARRANTY; without even the implied warranty of
	# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	# GNU General Public License for more details.
	#
	# You should have received a copy of the GNU General Public License
	# along with this program. If not, see <https://www.gnu.org/licenses/>.
	import cv2
	import depthai as dai
	
	from oakutils.blobs import models
	from oakutils.nodes import (
	    create_color_camera,
	    create_neural_network,
	    create_xout,
	    get_nn_bgr_frame,
	    get_nn_gray_frame,
	)
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	cam = create_color_camera(
	    pipeline,
	    resolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P,
	    preview_size=(640, 480),
	)
	
	# create the neural network node
	lp = create_neural_network(pipeline, cam.preview, models.LAPLACIAN_15X15)
	xout_lp = create_xout(pipeline, lp.out, "laplacian")
	
	# create another neural network node
	lp_gray = create_neural_network(pipeline, lp.passthrough, models.LAPLACIANGRAY_15X15)
	xout_lp_gray = create_xout(pipeline, lp_gray.out, "laplacian_gray")
	
	with dai.Device(pipeline) as device:
	    lp_queue: dai.DataOutputQueue = device.getOutputQueue("laplacian")
	    lp_gray_queue: dai.DataOutputQueue = device.getOutputQueue("laplacian_gray")
	
	    while True:
	        lp_data = lp_queue.get()
	        lp_gray_data = lp_gray_queue.get()
	
	        lp_frame = get_nn_bgr_frame(lp_data, normalization=255.0)
	
	        # also do this
	        lp_gray_data = lp_gray_data.getData()
	
	        lp_gray_frame = get_nn_gray_frame(lp_gray_data, normalization=255.0)
	
	        cv2.imshow("lp frame", lp_frame)
	        if cv2.waitKey(1) == ord("q"):
	            break
	        cv2.imshow("lp gray frame", lp_gray_frame)
	        if cv2.waitKey(1) == ord("q"):
	            break

