.. _examples_webcam:

Example: webcam.py
==================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
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

