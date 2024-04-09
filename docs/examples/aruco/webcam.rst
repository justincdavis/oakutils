.. _examples_aruco/webcam:

Example: aruco/webcam.py
========================

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
	"""Example showcasing how to use ArucoStream on the Webcam abstraction."""
	import cv2
	
	from oakutils import Webcam
	from oakutils.aruco import ArucoStream
	
	cam = Webcam()
	stream = ArucoStream(
	    aruco_dict=cv2.aruco.DICT_5X5_100,
	    marker_size=0.2,
	    calibration=cam.calibration,
	)
	
	while True:
	    _, frame = cam.read()
	    markers = stream.find(frame)
	    cv2.imshow("frame", stream.draw(frame, markers))
	    if cv2.waitKey(1) & 0xFF == ord("q"):
	        break

