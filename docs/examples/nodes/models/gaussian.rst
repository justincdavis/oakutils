.. _examples_nodes/models/gaussian:

Example: nodes/models/gaussian.py
=================================

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
	"""Example showcasing how to use the Gaussian model."""
	from __future__ import annotations
	
	import cv2
	import depthai as dai
	from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
	from oakutils.nodes.models import create_gaussian
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	cam = create_color_camera(pipeline, preview_size=(640, 480))
	
	# create neural network node
	lp = create_gaussian(pipeline, cam.preview, kernel_size=5)
	xout_lp = create_xout(pipeline, lp.out, "gaussian")
	
	with dai.Device(pipeline) as device:
	    l_queue: dai.DataOutputQueue = device.getOutputQueue("gaussian")
	
	    while True:
	        l_data = l_queue.get()
	        l_frame = get_nn_bgr_frame(l_data, frame_size=(640, 480), normalization=255.0)
	
	        cv2.imshow("gaussian frame", l_frame)
	        if cv2.waitKey(1) == ord("q"):
	            break

