.. _examples_nodes/imu:

Example: nodes/imu.py
=====================

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
	"""Example showcasing how to make an imu node."""
	from __future__ import annotations
	
	import depthai as dai
	
	from oakutils.nodes import create_imu, create_xout
	
	pipeline = dai.Pipeline()
	
	# create the color camera node
	imu = create_imu(
	    pipeline,
	    accelerometer_rate=400,
	    gyroscope_rate=400,
	    enable_accelerometer=True,
	    enable_gyroscope_calibrated=True,
	    enable_game_rotation_vector=True,
	)
	xout_imu = create_xout(pipeline, imu.out, "imu")
	
	with dai.Device(pipeline) as device:
	    queue: dai.DataOutputQueue = device.getOutputQueue("imu")
	
	    while True:
	        left = queue.get()
	
	        print(dir(left))

