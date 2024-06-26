.. _examples_calibration/oakd_calibration:

Example: calibration/oakd_calibration.py
========================================

.. code-block:: python

	# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
	#
	# MIT License
	"""Example on getting calibration data from the OAK-D."""
	
	import depthai_sdk as sdk
	from oakutils.calibration import CalibrationData, get_camera_calibration
	
	
	def print_calibration(calibration: CalibrationData) -> None:
	    """Print out basic calibration information about the camera."""
	    # print out the K matrices
	    print(f"K matrix for rgb: {calibration.rgb.K}")
	    print(f"K matrix for left: {calibration.left.K}")
	    print(f"K matrix for right: {calibration.right.K}")
	    print(f"K matrix for primary: {calibration.primary.K}")
	
	    # print out the distortion coefficients
	    print(f"Distortion coefficients for rgb: {calibration.rgb.D}")
	    print(f"Distortion coefficients for left: {calibration.left.D}")
	    print(f"Distortion coefficients for right: {calibration.right.D}")
	    print(f"Distortion coefficients for primary: {calibration.primary.D}")
	
	    # print out the stereo information
	    print(f"Q matrix: {calibration.stereo.Q_cv2}")
	    print(f"Manual Left Q matrix: {calibration.stereo.Q_left}")
	    print(f"Manual Right Q matrix: {calibration.stereo.Q_right}")
	
	
	# get it with the sdk
	with sdk.OakCamera() as oak:
	    calibration = get_camera_calibration(
	        device=oak.device,
	        rgb_size=(1920, 1080),
	        mono_size=(640, 400),
	        is_primary_mono_left=True,
	    )
	    print_calibration(calibration)
	
	# Create a CalibrationData object for the camera
	# create_camera_calibration requires an open device through depthai
	# this function will also pre-create the primary mono camera
	calibration = get_camera_calibration(
	    rgb_size=(1920, 1080),
	    mono_size=(640, 400),
	    is_primary_mono_left=True,
	)
	print_calibration(calibration)

