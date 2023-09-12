import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, create_xout

pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(pipeline, preview_size=(640, 480))
xout_cam = create_xout(pipeline, cam.video, "rgb")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("rgb")

    while True:
        in_rgb = queue.get()
        cv2.imshow("rgb", in_rgb.getCvFrame())
        if cv2.waitKey(1) == ord("q"):
            break