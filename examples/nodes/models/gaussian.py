import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, get_nn_bgr_frame, create_xout
from oakutils.nodes.models import create_gaussian


pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(pipeline, preview_size=(640, 480))

# create neural network node
lp = create_gaussian(pipeline, cam.preview, kernel_size=5)
xout_lp = create_xout(pipeline, lp.out, "gaussian")

with dai.Device(pipeline) as device:
    l_queue: dai.DataOutputQueue = device.getOutputQueue("gaussian")

    while True:
        l_data = l_queue.get()
        l_frame = get_nn_bgr_frame(l_data, frame_size=(640, 480))

        cv2.imshow("gaussian frame", l_frame)
        if cv2.waitKey(1) == ord("q"):
            break
