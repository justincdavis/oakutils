.. _examples_calibration/oak1_calibration:

Example: calibration/oak1_calibration.py
========================================

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
	"""Example on getting calibration data from the OAK-D."""
	import depthai_sdk as sdk
	
	from oakutils.calibration import ColorCalibrationData, get_camera_calibration
	
	
	def print_calibration(calibration: ColorCalibrationData) -> None:
	    """Print out basic calibration information about the camera."""
	    # print out the K matrices
	    print(f"K matrix for rgb: {calibration.K}")
	    # print out the distortion coefficients
	    print(f"Distortion coefficients for rgb: {calibration.D}")
	
	
	# get it with the sdk
	with sdk.OakCamera() as oak:
	    calibration = get_camera_calibration(
	        device=oak.device,
	        rgb_size=(1920, 1080),
	    )
	    print_calibration(calibration)
	
	# Create a ColorCalibration object for the camera
	calibration = get_camera_calibration(
	    rgb_size=(1920, 1080),
	)
	print_calibration(calibration)

