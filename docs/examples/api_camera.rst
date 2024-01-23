.. _examples_api_camera:

Example: api_camera.py
======================

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
	from oakutils import ApiCamera
	from oakutils.nodes import (
	    create_color_camera,
	    create_stereo_depth,
	    create_xout,
	    get_nn_point_cloud,
	)
	from oakutils.nodes.models import create_point_cloud
	from oakutils.tools.parsing import (
	    get_color_sensor_resolution_from_tuple,
	    get_mono_sensor_resolution_from_tuple,
	)
	from oakutils.point_clouds import create_point_cloud_from_np
	
	
	def main():
	    rgb_resolution = (1920, 1080)
	    mono_resolution = (640, 400)
	    use_left_mono = True
	    print("Starting API Camera")
	    oak = ApiCamera(
	        primary_mono_left=use_left_mono,
	        color_size=rgb_resolution,
	        mono_size=mono_resolution,
	    )
	    print("Camera Initialized")
	
	    cam = create_color_camera(
	        oak.pipeline, resolution=get_color_sensor_resolution_from_tuple(rgb_resolution)
	    )
	    xout_cam = create_xout(oak.pipeline, cam.video, "color")
	    stereo, left, right = create_stereo_depth(
	        oak.pipeline, resolution=get_mono_sensor_resolution_from_tuple(mono_resolution)
	    )
	    xout_left = create_xout(oak.pipeline, stereo.rectifiedLeft, "left")
	    xout_right = create_xout(oak.pipeline, stereo.rectifiedRight, "right")
	    xout_depth = create_xout(oak.pipeline, stereo.depth, "depth")
	    xout_disparity = create_xout(oak.pipeline, stereo.disparity, "disparity")
	    point_cloud, xin_xyz, start_pcl = create_point_cloud(
	        oak.pipeline, stereo.depth, oak.calibration
	    )
	    xout_point_cloud = create_xout(oak.pipeline, point_cloud.out, "point_cloud")
	
	    # add the basic displays
	    oak.add_display("color")
	    oak.add_display("left")
	    oak.add_display("right")
	    oak.add_display("depth")
	    oak.add_display("disparity")
	
	    # adding outputs from onboard neural networks is easy, but requires the correct calls
	    def pcl_callback(pcl):
	        pcl = get_nn_point_cloud(pcl)
	        pcl = create_point_cloud_from_np(pcl)
	        oak.pcv.update(pcl)
	
	    oak.add_callback("point_cloud", pcl_callback)
	
	    # # alternatively, you can do it with partials
	    # # shown below will be commented out since both calls do the same thing
	    # # the second approach will work for when the function is predefined
	    # # such as imported from another library or created elsewhere
	    # def pcl_callback_2(viz, pcl):
	    #     pcl = get_nn_point_cloud(pcl)
	    #     pcl = create_point_cloud_from_np(pcl)
	    #     viz.update(pcl)
	    # oak.add_callback("point_cloud", partial(pcl_callback_2, oak.pcv))
	
	    oak.add_device_call(start_pcl)  # start_pcl takes only an dai.Device, queue as such
	
	    oak.start(blocking=True)
	
	
	if __name__ == "__main__":
	    main()

