.. _examples_calibration/calibration:

Example: calibration/calibration.py
===================================

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
	import depthai_sdk as sdk
	
	from oakutils.calibration import get_camera_calibration
	
	
	def print_calibration(calibration):
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
	    print(f"Q matrix: {calibration.stereo.cv2_Q}")
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

