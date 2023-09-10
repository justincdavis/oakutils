import cv2
import depthai as dai

from oakutils.blobs import models
from oakutils.nodes import create_color_camera, get_nn_gray_frame
from oakutils.nodes.models import create_gaussian


pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(pipeline)

# create neural network node
lp, xout_lp, name = create_gaussian(pipeline, cam.preview, kernel_size=7, grayscale_out=True)

with dai.Device(pipeline) as device:
    l_queue: dai.DataOutputQueue = device.getOutputQueue(name)

    while True:
        l_data = l_queue.get()

        l_frame = get_nn_gray_frame(l_data)
        # convert to binary with otsu
        _, l_frame = cv2.threshold(l_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow("l frame", l_frame)
        if cv2.waitKey(1) == ord('q'):
            break
