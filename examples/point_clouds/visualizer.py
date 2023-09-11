import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, create_mono_camera
from oakutils.point_clouds import PointCloudVisualizer

pipeline = dai.Pipeline()
pcv = PointCloudVisualizer()

# create the color camera node
cam, xout_rgb = create_color_camera(pipeline, preview_size=(640, 480))
left, xout_left = create_mono_camera(pipeline, dai.CameraBoardSocket.LEFT)

with dai.Device(pipeline) as device:
    rgb_q: dai.DataOutputQueue = device.getOutputQueue("rgb")
    left_q: dai.DataOutputQueue = device.getOutputQueue("left")

    while True:
        in_rgb = rgb_q.get()
        cv2.imshow("rgb", in_rgb.getCvFrame())
        if cv2.waitKey(1) == ord("q"):
            break
