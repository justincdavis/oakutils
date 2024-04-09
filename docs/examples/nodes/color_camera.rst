.. _examples_nodes/color_camera:

Example: nodes/color_camera.py
==============================

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
	"""Example showcasing making a color_camera node."""
	from __future__ import annotations
	
	import cv2
	import depthai as dai
	
	from oakutils.nodes import create_color_camera, create_xout
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	cam = create_color_camera(pipeline, preview_size=(640, 480))
	xout_cam = create_xout(pipeline, cam.video, "rgb")
	
	with dai.Device(pipeline) as device:
	    queue: dai.DataOutputQueue = device.getOutputQueue("rgb")
	
	    while True:
	        in_rgb = queue.get()
	        cv2.imshow("rgb", in_rgb.getCvFrame())
	        if cv2.waitKey(1) == ord("q"):
	            break

