import cv2
import depthai as dai

from oakutils.nodes import create_mono_camera

pipeline = dai.Pipeline()

# create the color camera node
left, xout_left = create_mono_camera(pipeline, dai.CameraBoardSocket.LEFT)
right, xout_right = create_mono_camera(pipeline, dai.CameraBoardSocket.RIGHT)

with dai.Device(pipeline) as device:
    lq: dai.DataOutputQueue = device.getOutputQueue("left")
    rq: dai.DataOutputQueue = device.getOutputQueue("right")

    while True:
        left = lq.get()
        right = rq.get()
        cv2.imshow("left", left.getCvFrame())
        cv2.imshow("right", right.getCvFrame())
        if cv2.waitKey(1) == ord("q"):
            break
