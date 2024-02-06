.. _examples_aruco/localizer:

Example: aruco/localizer.py
===========================

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
	"""Example showcasing how to use the ArucoLocalizer."""
	
	import cv2
	import depthai as dai
	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib.axes import Axes
	
	from oakutils.aruco import ArucoLocalizer, ArucoStream
	from oakutils.calibration import get_camera_calibration_basic
	from oakutils.nodes import create_color_camera, create_xout
	from oakutils.tools.transform import create_transform
	
	
	def draw_coordinate_axes(ax: Axes, h: np.ndarray, label: str) -> None:
	    """Draws the coordinate axes of a given transform."""
	    p = h[0:3, 3]  # Origin of the coordinate frame
	    ux = h @ np.array([1, 0, 0, 1])  # Tip of the x axis
	    uy = h @ np.array([0, 1, 0, 1])  # Tip of the y axis
	    uz = h @ np.array([0, 0, 1, 1])  # Tip of the z axis
	    ax.plot(xs=[p[0], ux[0]], ys=[p[1], ux[1]], zs=[p[2], ux[2]], c="r")  # x axis
	    ax.plot(xs=[p[0], uy[0]], ys=[p[1], uy[1]], zs=[p[2], uy[2]], c="g")  # y axis
	    ax.plot(xs=[p[0], uz[0]], ys=[p[1], uz[1]], zs=[p[2], uz[2]], c="b")  # z axis
	    ax.text(p[0], p[1], p[2], label)  # Also draw the label of the coordinate frame
	
	
	calibration = get_camera_calibration_basic()
	stream = ArucoStream(
	    cv2.aruco.DICT_4X4_100,
	    0.05,
	    calibration.rgb,
	    5,
	    5,
	    0.95,
	)
	# uses these markers: http://jevois.org/moddoc/DemoArUco/screenshot2.png
	marker_locations = {
	    42: create_transform(0.0, 0.0, 0.0, -1.0, 0.5, 0.0),
	    18: create_transform(0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
	    12: create_transform(0.0, 0.0, 0.0, 1.0, 0.5, 0.0),
	    23: create_transform(0.0, 0.0, 0.0, -1.0, -0.5, 0.0),
	    43: create_transform(0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
	    5: create_transform(0.0, 0.0, 0.0, 1.0, -0.5, 0.0),
	}
	localizer = ArucoLocalizer(
	    marker_locations,
	    5,
	    5,
	    0.95,
	)
	
	pipeline = dai.Pipeline()
	cam = create_color_camera(pipeline)
	xout_cam = create_xout(pipeline, cam.video, "rgb")
	
	# create plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.ion()
	plt.show()
	counter = 0
	
	with dai.Device(pipeline) as device:
	    cam_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
	
	    while True:
	        in_rgb = cam_queue.get()
	        frame = in_rgb.getCvFrame()
	        markers = stream.find(frame)
	        transform = localizer.localize(markers)
	
	        for tag, H in marker_locations.items():
	            draw_coordinate_axes(ax, H, str(tag))
	        draw_coordinate_axes(ax, transform, "Camera")
	
	        plt.draw()
	        plt.pause(0.001)
	
	        cv2.imshow("frame", stream.draw(frame, markers))
	        if cv2.waitKey(1) == ord("q"):
	            break

