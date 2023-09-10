import cv2
import depthai as dai

from oakutils.blobs import models
from oakutils.nodes import create_neural_network, get_nn_gray_frame, create_stereo_depth, create_color_camera, get_nn_bgr_frame


pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(pipeline)

# create the stereo camera node
stereo, left, right, xout_left, xout_right, xout_depth, xout_disparity, xout_rect_left, xout_rect_right = create_stereo_depth(pipeline, resolution=dai.MonoCameraProperties.SensorResolution.THE_480_P)

# create neural network node
lp, xout_lp = create_neural_network(pipeline, cam.preview, models.LAPLACIANGRAY_7X7, stream_name="l")

with dai.Device(pipeline) as device:
    l_queue: dai.DataOutputQueue = device.getOutputQueue("l")

    while True:
        l_data = l_queue.get()

        l_frame = get_nn_gray_frame(l_data)
        # # convert to binary with otsu
        # _, l_frame = cv2.threshold(l_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow("l frame", l_frame)
        if cv2.waitKey(1) == ord('q'):
            break
