.. _examples_webcam:

Example: webcam.py
==================

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
	"""Example showcasing how to use the Webcam abstraction."""
	from __future__ import annotations
	
	import cv2
	from oakutils import Webcam
	
	cam = Webcam()
	while True:
	    ret, frame = cam.read()
	    if not ret:
	        continue
	    cv2.imshow("frame", frame)
	    if cv2.waitKey(1) == ord("q"):
	        break

