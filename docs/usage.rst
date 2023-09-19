Usage
=====

Using the camera
----------------

The oakutils module supports using the camera in a multitude of methods:

- Using the `oakutils.ApiCamera` object:

   .. code-block:: python
   
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

      oak = ApiCamera(
         primary_mono_left=True,
         color_size=(1920, 1080),
         mono_size=(640, 400),
      )

      cam = create_color_camera(
         oak.pipeline, resolution=get_color_sensor_resolution_from_tuple(rgb_resolution)
      )
      xout_cam = create_xout(oak.pipeline, cam.video, "color")
      stereo, left, right = create_stereo_depth(
         oak.pipeline, resolution=get_mono_sensor_resolution_from_tuple(mono_resolution)
      )
      point_cloud, xin_xyz, start_pcl = create_point_cloud(
         oak.pipeline, stereo.depth, oak.calibration
      )
      xout_point_cloud = create_xout(oak.pipeline, point_cloud.out, "point_cloud")

      oak.add_display("color")

      def pcl_callback(pcl):
         pcl = get_nn_point_cloud(pcl)
         pcl = create_point_cloud_from_np(pcl)
         oak.pcv.update(pcl)

      oak.add_callback("point_cloud", pcl_callback)
      oak.add_device_call(start_pcl) 

      oak.start(blocking=True)


- Using the `depthai_sdk` library:

   .. code-block:: python
   
      import depthai_sdk as sdk
      from oakutils.nodes import create_color_camera

      with sdk.OakCamera() as oak:
         oak.build()
         camera = create_color_camera(oak.pipeline)
         oak.visualize(camera.video)
         oak.start(block=True)

   .. code-block:: python
   
      import depthai_sdk as sdk
      from oakutils.nodes import create_stereo_depth

      with sdk.OakCamera() as oak:
         oak.build()
         stereo = create_stereo_depth(oak.pipeline)
         oak.visualize(stereo.disparity)
         oak.visualize(stereo.depth)
         oak.start(block=True)

- Using the `depthai` library:

   .. code-block:: python
   
      import cv2
      import depthai as dai
      from oakutils.nodes import create_stereo_depth, create_xout

      pipeline = dai.Pipeline()
      stereo = create_stereo_depth(pipeline)
      xout_disparity = create_xout(pipeline, stereo.disparity, "disparity")
      xout_depth = create_xout(pipeline, stereo.depth, "depth")

      with dai.Device(pipeline) as device:
         disparity_q = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
         depth_q = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

         while True:
            in_disparity = disparity_q.get()
            in_depth = depth_q.get()

            cv2.imshow("disparity", in_disparity.getFrame())
            cv2.imshow("depth", in_depth.getFrame())

            if cv2.waitKey(1) == ord('q'):
               break
